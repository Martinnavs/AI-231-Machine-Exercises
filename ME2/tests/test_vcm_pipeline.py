"""Fast, CPU-only, checkpoint-free tests for `vcm.pipeline`. Uses Task 02's
shared `vcm_stub_model_factory` fixture (see tests/conftest.py) -- no real
trained checkpoint is ever loaded here, so this file stays in the cheap
default test run.
"""

from __future__ import annotations

import pytest
import torch

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.decoder import decode
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.optiona.grammar import SPEC_GRAMMAR, TOY_GRAMMAR
from me2_voicegen.vcm.pipeline import (
    SlidingWindowPipeline,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    infer_waveform,
    logp_for_waveform,
)

SAMPLE_RATE = 16000

CALL_IDS = alphabet.encode("call")  # [3, 1, 12, 12]


def _tone(duration_s: float, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    n = max(1, int(duration_s * sample_rate))
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return 0.1 * torch.sin(2 * 3.14159265 * 220.0 * t)


# ---------------------------------------------------------------------------
# Whole-clip mode.
# ---------------------------------------------------------------------------


def test_logp_for_waveform_shape_matches_alphabet(vcm_stub_model_factory):
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()
    waveform = _tone(0.2)
    logp = logp_for_waveform(model, fe, waveform)
    assert logp.ndim == 2
    assert logp.shape[1] == alphabet.ALPHABET_SIZE


def test_infer_waveform_accepts_grammar_phrase_at_permissive_threshold(vcm_stub_model_factory):
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()
    waveform = _tone(0.1)
    result = infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=-50.0)
    assert result.no_match is False
    assert result.intent == "CALL"


def test_infer_waveform_rejects_when_confidence_below_threshold(vcm_stub_model_factory):
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()
    waveform = _tone(0.1)
    result = infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=10.0)
    assert result.no_match is True
    assert result.intent is None


def test_infer_waveform_no_match_at_a_realistic_threshold_for_an_uncommitted_phrase(
    vcm_stub_model_factory,
):
    """"dimmer" forced through SPEC_GRAMMAR (which has no bare "dimmer"
    alias -- only the toy-only alias does) has no clean forced alignment
    to any SPEC_GRAMMAR terminal; the beam search's best reachable
    terminal is only reachable via near-negligible-probability
    off-forced-path letters (score far below the forced-phrase-confidence
    range seen elsewhere in this file, e.g. ~-4.2 to -4.8 for an actual
    forced match) -- a realistic (non-extreme) threshold correctly rejects
    it as no_match rather than accepting a low-confidence coincidental
    match."""
    model = vcm_stub_model_factory(forced_ids=alphabet.encode("dimmer"))
    fe = LogMelFeatureExtractor()
    waveform = _tone(0.1)
    result = infer_waveform(model, fe, waveform, SPEC_GRAMMAR, threshold=-6.0)
    assert result.no_match is True
    assert result.intent is None


def test_infer_waveform_matches_direct_decode_call(vcm_stub_model_factory):
    """infer_waveform is exactly logp_for_waveform + vcm.decoder.decode --
    proves the pipeline doesn't silently change decode behavior."""
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()
    waveform = _tone(0.1)
    logp = logp_for_waveform(model, fe, waveform)
    direct = decode(logp, TOY_GRAMMAR, threshold=-50.0)
    via_pipeline = infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=-50.0)
    assert direct.intent == via_pipeline.intent
    assert direct.confidence == via_pipeline.confidence


# ---------------------------------------------------------------------------
# Sliding-window mode: ring buffer + debounce/refractory.
# ---------------------------------------------------------------------------


def _make_pipeline(vcm_stub_model_factory, **overrides):
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()
    kwargs = dict(
        model=model,
        feature_extractor=fe,
        grammar=TOY_GRAMMAR,
        threshold=-50.0,
        window_s=0.1,
        stride_s=0.02,
        refractory_s=0.06,
    )
    kwargs.update(overrides)
    return SlidingWindowPipeline(**kwargs)


def test_sliding_window_pipeline_window_and_stride_samples_from_seconds(vcm_stub_model_factory):
    pipe = _make_pipeline(vcm_stub_model_factory)
    assert pipe.window_samples == int(0.1 * SAMPLE_RATE)
    assert pipe.stride_samples == int(0.02 * SAMPLE_RATE)


def test_sliding_window_pipeline_default_window_and_stride_match_module_constants():
    assert WINDOW_SAMPLES == int(round(1.5 * SAMPLE_RATE))
    assert STRIDE_SAMPLES == int(round(0.1 * SAMPLE_RATE))


def test_sliding_window_pipeline_ring_buffer_never_exceeds_window_samples(vcm_stub_model_factory):
    pipe = _make_pipeline(vcm_stub_model_factory)
    chunk = torch.zeros(int(0.02 * SAMPLE_RATE))
    for _ in range(20):
        pipe.feed(chunk)
        assert pipe._buffer.numel() <= pipe.window_samples


def test_sliding_window_pipeline_no_trigger_before_first_full_window(vcm_stub_model_factory):
    pipe = _make_pipeline(vcm_stub_model_factory)
    chunk = torch.zeros(int(0.02 * SAMPLE_RATE))
    events = pipe.feed(chunk)
    assert events == []


def test_sliding_window_pipeline_debounce_suppresses_trigger_inside_cooldown(vcm_stub_model_factory):
    """The acceptance-criterion test: feed enough identical accepted
    windows in a row that, without debounce, every stride would re-trigger
    -- assert the cooldown suppresses every trigger that falls inside the
    refractory window after the first one, then correctly re-arms once the
    cooldown has fully elapsed."""
    pipe = _make_pipeline(vcm_stub_model_factory)
    chunk = torch.zeros(int(0.02 * SAMPLE_RATE))

    all_events: list = []
    for _ in range(10):
        all_events.extend(pipe.feed(chunk))

    trigger_windows = [e.window_index for e in all_events]
    assert len(trigger_windows) >= 2, "expected multiple triggers across 10 strides once re-armed"

    # refractory_s=0.06s == 3 strides (0.02s each): consecutive triggers
    # must never be closer together than that, i.e. debounce actually
    # suppressed the strides in between rather than firing every stride.
    for a, b in zip(trigger_windows, trigger_windows[1:]):
        assert b - a >= 3

    assert pipe.suppressed_count > 0, "expected at least one suppressed in-cooldown trigger"


def test_sliding_window_pipeline_debounce_disabled_when_refractory_zero(vcm_stub_model_factory):
    """refractory_s=0 means every accepted window fires -- confirms the
    cooldown, not something else, is what suppresses repeats above."""
    pipe = _make_pipeline(vcm_stub_model_factory, refractory_s=0.0)
    chunk = torch.zeros(int(0.02 * SAMPLE_RATE))

    all_events: list = []
    for _ in range(10):
        all_events.extend(pipe.feed(chunk))

    # First full window fills at stride index 5 (0.1s window / 0.02s
    # stride); every stride after that should trigger with no suppression.
    assert pipe.suppressed_count == 0
    assert len(all_events) >= 4


def test_sliding_window_pipeline_reset_clears_state(vcm_stub_model_factory):
    pipe = _make_pipeline(vcm_stub_model_factory)
    chunk = torch.zeros(int(0.02 * SAMPLE_RATE))
    for _ in range(10):
        pipe.feed(chunk)
    assert pipe._buffer.numel() > 0

    pipe.reset()
    assert pipe._buffer.numel() == 0
    assert pipe._window_index == 0
    assert pipe.suppressed_count == 0
    assert pipe._cooldown_samples_remaining == 0


def test_sliding_window_pipeline_feed_empty_chunk_is_noop(vcm_stub_model_factory):
    pipe = _make_pipeline(vcm_stub_model_factory)
    events = pipe.feed(torch.zeros(0))
    assert events == []
    assert pipe._buffer.numel() == 0


# ---------------------------------------------------------------------------
# load_checkpoint: only exercised against the real checkpoint if present on
# disk (slow/optional -- never required for the fast suite, per this
# ticket's acceptance criteria).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_load_checkpoint_real_artifact_if_present():
    from pathlib import Path

    from me2_voicegen.vcm.pipeline import load_checkpoint

    ckpt_path = Path(__file__).resolve().parents[1] / "out" / "vcm" / "checkpoints" / "checkpoint.pt"
    if not ckpt_path.exists():
        pytest.skip("out/vcm/checkpoints/checkpoint.pt not present (Task 04 training artifact)")

    model, checkpoint = load_checkpoint(ckpt_path, device="cpu")
    assert model.training is False
    assert "CC-BY-NC-SA-4.0" in checkpoint["license"]
    waveform = _tone(0.1)
    fe = LogMelFeatureExtractor()
    result = infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=-1e6)
    assert result.text is not None


# ---------------------------------------------------------------------------
# load_checkpoint: weights_only / allow_unsafe_load fallback behavior
# (SEC-2). `torch.load` is monkeypatched -- no real checkpoint needed, and
# no real malicious pickle is ever constructed; "failure" is simulated.
# ---------------------------------------------------------------------------


def _fake_checkpoint_dict():
    from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC

    config = MatchboxNetConfig(
        n_mels=40, n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8, epilogue_channels=8
    )
    model = MatchboxNetCTC(config)
    return {
        "model_state_dict": model.state_dict(),
        "config": {
            "n_mels": config.n_mels,
            "n_blocks": config.n_blocks,
            "channels": config.channels,
            "kernel_sizes": config.kernel_sizes,
            "prologue_channels": config.prologue_channels,
            "epilogue_channels": config.epilogue_channels,
            "alphabet_size": config.alphabet_size,
        },
        "preset": "test-fixture",
    }


def test_load_checkpoint_default_call_never_requests_weights_only(monkeypatch):
    import me2_voicegen.vcm.pipeline as pipeline_module

    calls = []

    def fake_torch_load(path, map_location=None, **kwargs):
        calls.append(kwargs)
        return _fake_checkpoint_dict()

    monkeypatch.setattr(pipeline_module.torch, "load", fake_torch_load)
    pipeline_module.load_checkpoint("fake/path.pt")
    # Existing callers (evaluate.py/benchmark.py/export_onnx.py) never pass
    # weights_only=True, so this call shape (no weights_only kwarg at all)
    # must remain exactly what they get.
    assert calls == [{}]


def test_load_checkpoint_weights_only_failure_raises_without_allow_unsafe_load(monkeypatch):
    import me2_voicegen.vcm.pipeline as pipeline_module

    def fake_torch_load(path, map_location=None, weights_only=False):
        if weights_only:
            raise RuntimeError("simulated weights_only allow-list rejection")
        return _fake_checkpoint_dict()

    monkeypatch.setattr(pipeline_module.torch, "load", fake_torch_load)
    with pytest.raises(RuntimeError, match="allow_unsafe_load"):
        pipeline_module.load_checkpoint("fake/path.pt", weights_only=True)


def test_load_checkpoint_weights_only_failure_never_silently_falls_back_by_default(monkeypatch):
    """The core SEC-2 requirement: no automatic, silent retry with
    weights_only=False when the safe path fails and the caller didn't
    explicitly opt into allow_unsafe_load."""
    import me2_voicegen.vcm.pipeline as pipeline_module

    unsafe_load_calls = []

    def fake_torch_load(path, map_location=None, weights_only=False):
        if weights_only:
            raise RuntimeError("simulated weights_only allow-list rejection")
        unsafe_load_calls.append(1)
        return _fake_checkpoint_dict()

    monkeypatch.setattr(pipeline_module.torch, "load", fake_torch_load)
    with pytest.raises(RuntimeError):
        pipeline_module.load_checkpoint(
            "fake/path.pt", weights_only=True, allow_unsafe_load=False
        )
    assert unsafe_load_calls == []


def test_load_checkpoint_weights_only_failure_falls_back_with_explicit_allow_unsafe_load(
    monkeypatch, capsys
):
    import me2_voicegen.vcm.pipeline as pipeline_module

    def fake_torch_load(path, map_location=None, weights_only=False):
        if weights_only:
            raise RuntimeError("simulated weights_only allow-list rejection")
        return _fake_checkpoint_dict()

    monkeypatch.setattr(pipeline_module.torch, "load", fake_torch_load)
    model, checkpoint = pipeline_module.load_checkpoint(
        "fake/path.pt", weights_only=True, allow_unsafe_load=True
    )
    assert checkpoint["preset"] == "test-fixture"
    assert model.training is False
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "allow_unsafe_load" in captured.err


def test_infer_waveform_threads_score_mode(vcm_stub_model_factory):
    """per_char confidence == command_raw_score / len(phrase); the default
    (mean_frame) is unchanged and differs from it (dense-d2-loose-impl)."""
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()
    waveform = _tone(0.1)
    default = infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=-50.0)
    per_char = infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=-50.0,
                              score_mode="per_char")
    assert per_char.intent == default.intent == "CALL"
    assert per_char.confidence == pytest.approx(per_char.command_raw_score / len(per_char.grammar_text))
    assert default.confidence != per_char.confidence
    assert default == infer_waveform(model, fe, waveform, TOY_GRAMMAR, threshold=-50.0,
                                     score_mode="mean_frame")


def test_sliding_window_pipeline_threads_score_mode(vcm_stub_model_factory):
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    pipe = SlidingWindowPipeline(model, LogMelFeatureExtractor(), TOY_GRAMMAR, threshold=-50.0,
                                 score_mode="per_char")
    assert pipe.score_mode == "per_char"
    assert SlidingWindowPipeline(model, LogMelFeatureExtractor(), TOY_GRAMMAR,
                                 threshold=-50.0).score_mode == "mean_frame"
