"""Unit tests for `vcm.optionb.incomplete_probes`. No checkpoint, no WAV
file: log-posteriors are built directly via the repo's `make_posterior`
idiom (duplicated here per convention -- see
`tests/test_vcm_decoder_incomplete_prefix.py`'s own copy, `tests/` is not a
package) and `ForcedAlignment` is sometimes constructed by hand to exercise
edge cases `force_align`'s own transition rules cannot organically produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from me2_voicegen.vcm import alphabet as vcm_alphabet
from me2_voicegen.vcm.optionb import incomplete_probes as ip
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.segment_scorer import ForcedAlignment, force_align

PEAK = 20.0


def make_posterior(text: str, peak: float = PEAK) -> np.ndarray:
    frame_ids: list[int] = []
    prev: int | None = None
    for ch in text:
        cid = vcm_alphabet.CHAR_TO_ID[ch]
        if cid == prev:
            frame_ids.append(vcm_alphabet.BLANK_ID)
            prev = None
        frame_ids.append(cid)
        prev = cid

    T = len(frame_ids)
    logits = np.full((T, vcm_alphabet.ALPHABET_SIZE), -peak, dtype=np.float64)
    for t, cid in enumerate(frame_ids):
        logits[t, cid] = peak
    m = logits.max(axis=-1, keepdims=True)
    return logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))


def _dummy_alignment(
    state_path: tuple[int, ...], start_frame: int = 0, end_frame: int | None = None
) -> ForcedAlignment:
    return ForcedAlignment(
        token_ids=(0,),
        state_path=state_path,
        frame_token_ids=tuple(0 for _ in state_path),
        log_probability=-1.0,
        start_frame=start_frame,
        end_frame=len(state_path) - 1 if end_frame is None else end_frame,
    )


# ---------------------------------------------------------------------------
# Word-boundary prefix matching / char-overlap exclusion.
# ---------------------------------------------------------------------------


def test_is_word_boundary_prefix_requires_trailing_space():
    assert ip.is_word_boundary_prefix("remind me to study", "remind")
    assert not ip.is_word_boundary_prefix("reminders", "remind")
    assert not ip.is_word_boundary_prefix("remind", "remind")


def test_has_char_overlap_flags_the_false_match():
    assert ip.has_char_overlap("reminders", "remind")
    assert not ip.has_char_overlap("remind me to study", "remind")
    assert not ip.has_char_overlap("remind", "remind")
    assert not ip.has_char_overlap("call", "remind")


# ---------------------------------------------------------------------------
# select_target_rows: deterministic seeded selection, split/source filtering,
# and exclusion of character-overlap false matches.
# ---------------------------------------------------------------------------


def _row(filename, split="val", source_dataset="optionb", transcript=""):
    return {
        "filename": filename,
        "split": split,
        "source_dataset": source_dataset,
        "group_id": "s1",
        "path": f"audio/{filename}",
        "transcript": transcript,
    }


def _resolve(row):
    return row["transcript"] or None


def test_select_target_rows_excludes_char_overlap_false_match():
    rows = [
        _row("a.wav", transcript="reminders"),
        _row("b.wav", transcript="remind me to study"),
    ]
    specs = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve)
    assert [s.source_row["filename"] for s in specs] == ["b.wav"]


def test_select_target_rows_filters_split_and_source_dataset():
    rows = [
        _row("a.wav", split="test", transcript="remind me to study"),
        _row("b.wav", split="val", source_dataset="babble", transcript="remind me to study"),
        _row("c.wav", split="val", transcript="remind me to study"),
    ]
    specs = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve)
    assert [s.source_row["filename"] for s in specs] == ["c.wav"]


def test_select_target_rows_caps_and_is_deterministic():
    rows = [_row(f"r{i:02d}.wav", transcript="remind me to study") for i in range(50)]
    specs_a = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve, cap_per_prefix=5, seed=7)
    specs_b = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve, cap_per_prefix=5, seed=7)
    assert len(specs_a) == 5
    assert [s.source_row["filename"] for s in specs_a] == [s.source_row["filename"] for s in specs_b]


def test_select_target_rows_different_seed_can_differ():
    rows = [_row(f"r{i:02d}.wav", transcript="remind me to study") for i in range(50)]
    specs_a = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve, cap_per_prefix=5, seed=1)
    specs_b = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve, cap_per_prefix=5, seed=2)
    names_a = [s.source_row["filename"] for s in specs_a]
    names_b = [s.source_row["filename"] for s in specs_b]
    assert names_a != names_b


def test_select_target_rows_none_resolved_transcript_skipped():
    rows = [_row("a.wav", transcript="")]
    specs = ip.select_target_rows(rows, frozenset({"remind"}), "val", _resolve)
    assert specs == []


def test_probe_spec_prefix_word_count():
    spec = ip.ProbeSpec(source_row=_row("a.wav"), prefix="set the lights to")
    assert spec.prefix_word_count == 4


# ---------------------------------------------------------------------------
# compute_crop: frame-to-sample mapping, never crossing the next word.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grace_frames", [0, 1, 3, 5, 50, 1000])
def test_compute_crop_never_crosses_next_word(grace_frames):
    text = "set the lights to red"
    prefix = "set the lights to"
    alignment = force_align(make_posterior(text), text)

    crop = ip.compute_crop(alignment, text, prefix, grace_frames)

    assert isinstance(crop, ip.CropResult)
    assert crop.crop_end_frame <= crop.next_word_first_char_frame
    assert crop.crop_end_frame >= crop.last_char_frame + 1
    assert crop.crop_end_sample == crop.crop_end_frame * 160


def test_compute_crop_maps_frames_to_samples():
    text = "turn the volume up"
    prefix = "turn the volume"
    alignment = force_align(make_posterior(text), text)

    crop = ip.compute_crop(alignment, text, prefix, grace_frames=2)

    assert crop.crop_end_sample == crop.crop_end_frame * 160
    last_char_index = len(prefix) - 1
    assert alignment.state_path[crop.last_char_frame] == 2 * last_char_index + 1


def test_compute_crop_rejects_non_word_boundary_prefix():
    text = "reminders"
    alignment = force_align(make_posterior(text), text)
    with pytest.raises(ValueError):
        ip.compute_crop(alignment, text, "remind", grace_frames=0)


def test_compute_crop_no_gap_failure():
    # text = "ab cd" -> labels: [_, a, _, b, _, ' ', _, c, _, d, _] (state 2j+1
    # is char j). prefix "ab" -> last_char_index=1 (state 3), next word's
    # first char index=3 (state 7). Crafted so state 7 is visited the frame
    # right after the last frame at state 3 -- zero frame gap.
    state_path = (0, 1, 2, 3, 7, 7, 7, 7, 8)
    alignment = _dummy_alignment(state_path)
    result = ip.compute_crop(alignment, "ab cd", "ab", grace_frames=0)
    assert isinstance(result, ip.CropFailure)
    assert result.reason == ip.FAILURE_NO_GAP


def test_compute_crop_too_short_failure():
    # Gap is 1 frame (state 3 at frame 2, state 7 first at frame 4), but
    # crop_end_sample = 3 * 160 = 480 samples < the 0.15s (2400-sample)
    # floor.
    state_path = (0, 1, 2, 3, 3, 5, 7, 7, 8)
    alignment = _dummy_alignment(state_path)
    result = ip.compute_crop(alignment, "ab cd", "ab", grace_frames=0)
    assert isinstance(result, ip.CropFailure)
    assert result.reason == ip.FAILURE_CROP_TOO_SHORT


def test_compute_crop_frame_hop_stride2_doubles_sample_offset():
    # quartznet-promotion ticket 01: for the SAME alignment (same frame
    # indices), a stride-2 frame_hop (320 samples/frame) must give exactly 2x
    # the stride-1 crop_end_sample, and the frame-level crop is unchanged.
    text = "turn the volume up"
    prefix = "turn the volume"
    alignment = force_align(make_posterior(text), text)

    stride1 = ip.compute_crop(alignment, text, prefix, grace_frames=2)
    stride2 = ip.compute_crop(alignment, text, prefix, grace_frames=2, frame_hop=2 * 160)

    assert isinstance(stride1, ip.CropResult)
    assert isinstance(stride2, ip.CropResult)
    assert stride2.crop_end_frame == stride1.crop_end_frame
    assert stride1.crop_end_sample == stride1.crop_end_frame * 160
    assert stride2.crop_end_sample == stride2.crop_end_frame * 320
    assert stride2.crop_end_sample == 2 * stride1.crop_end_sample


# ---------------------------------------------------------------------------
# find_quiet_window: R2-1 (excludes onset/tail-adjacent frames) + R2-5
# (existence of a candidate region is not enough -- it must also measure at
# least 20dB below the aligned speech span's RMS, and both lead-in and tail
# are searched, not just lead-in).
# ---------------------------------------------------------------------------

SPEECH_AMPLITUDE = 1.0  # RMS 1.0 for a constant-amplitude array
QUIET_AMPLITUDE = 0.001  # RMS 0.001 -- 60dB below SPEECH_AMPLITUDE, clears the 20dB bar


def test_find_quiet_window_uses_lead_in_when_available():
    lead_in_quiet = np.full(4800, QUIET_AMPLITUDE, dtype=np.float32)
    pre_speech_buffer = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)
    speech = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)
    waveform = np.concatenate([lead_in_quiet, pre_speech_buffer, speech])
    # start_frame = 8000/160 = 50; end_frame = 50 + 20 - 1 = 69; no tail exists.
    alignment = _dummy_alignment(tuple(range(70)), start_frame=50, end_frame=69)

    window, source = ip.find_quiet_window(waveform, alignment)

    assert source == ip.SILENCE_SOURCE_LEAD_IN
    assert window is not None
    assert window.shape == (1600,)
    assert float(np.sqrt(np.mean(np.square(window)))) == pytest.approx(QUIET_AMPLITUDE, abs=1e-6)


def test_find_quiet_window_uses_tail_when_lead_in_unavailable():
    speech = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)  # start_frame=0, end_frame=19
    post_speech_buffer = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)
    tail_quiet = np.full(4800, QUIET_AMPLITUDE, dtype=np.float32)
    waveform = np.concatenate([speech, post_speech_buffer, tail_quiet])
    alignment = _dummy_alignment(tuple(range(20)), start_frame=0, end_frame=19)

    window, source = ip.find_quiet_window(waveform, alignment)

    assert source == ip.SILENCE_SOURCE_TAIL
    assert window is not None
    assert window.shape == (1600,)
    assert float(np.sqrt(np.mean(np.square(window)))) == pytest.approx(QUIET_AMPLITUDE, abs=1e-6)


def test_find_quiet_window_rejects_a_region_that_is_not_quiet_enough():
    # A "quiet" candidate exists (the length check would have passed under
    # the R2-1 fix), but it is only ~6dB below speech level -- far short of
    # the 20dB bar -- so R2-5 must reject it rather than tile it.
    speech = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)
    post_speech_buffer = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)
    almost_quiet_tail = np.full(4800, 0.5, dtype=np.float32)
    waveform = np.concatenate([speech, post_speech_buffer, almost_quiet_tail])
    alignment = _dummy_alignment(tuple(range(20)), start_frame=0, end_frame=19)

    window, source = ip.find_quiet_window(waveform, alignment)

    assert window is None
    assert source == ip.SILENCE_SOURCE_DIGITAL_ZERO


def test_find_quiet_window_none_when_neither_region_is_long_enough():
    waveform = np.zeros(500, dtype=np.float32)
    alignment = _dummy_alignment((0,), start_frame=1, end_frame=0)
    window, source = ip.find_quiet_window(waveform, alignment)
    assert window is None
    assert source == ip.SILENCE_SOURCE_DIGITAL_ZERO


def test_find_quiet_window_none_when_speech_span_itself_is_silent():
    # A relative dB threshold is meaningless against a zero reference --
    # must not accept any candidate in this case.
    silence = np.zeros(3200, dtype=np.float32)
    quiet_tail = np.full(4800, QUIET_AMPLITUDE, dtype=np.float32)
    waveform = np.concatenate([silence, np.zeros(3200, dtype=np.float32), quiet_tail])
    alignment = _dummy_alignment(tuple(range(20)), start_frame=0, end_frame=19)
    window, source = ip.find_quiet_window(waveform, alignment)
    assert window is None
    assert source == ip.SILENCE_SOURCE_DIGITAL_ZERO


def test_find_quiet_window_frame_hop_scales_region_boundaries():
    # quartznet-promotion ticket 01: start_frame/end_frame are posterior
    # frames, so with frame_hop=320 (stride 2) every region boundary doubles
    # versus the stride-1 default. Same waveform + alignment:
    #   stride 1: speech span [0, 3200), backoff 3200, tail search from
    #             20*160 + 3200 = 6400 -- the quiet region [6400, 11200) is
    #             found.
    #   stride 2: speech span [0, 6400), backoff 6400, tail search from
    #             20*320 + 6400 = 12800 -- past the end of the waveform, so
    #             no region is long enough and it falls back to digital zero.
    speech = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)  # frames 0..19
    loud_buffer = np.full(3200, SPEECH_AMPLITUDE, dtype=np.float32)
    quiet_tail = np.full(4800, QUIET_AMPLITUDE, dtype=np.float32)
    waveform = np.concatenate([speech, loud_buffer, quiet_tail])
    alignment = _dummy_alignment(tuple(range(20)), start_frame=0, end_frame=19)

    window, source = ip.find_quiet_window(waveform, alignment)
    assert source == ip.SILENCE_SOURCE_TAIL
    assert window is not None
    assert float(np.sqrt(np.mean(np.square(window)))) == pytest.approx(QUIET_AMPLITUDE, abs=1e-6)

    window_s2, source_s2 = ip.find_quiet_window(waveform, alignment, frame_hop=2 * 160)
    assert window_s2 is None
    assert source_s2 == ip.SILENCE_SOURCE_DIGITAL_ZERO


# ---------------------------------------------------------------------------
# Trailing-silence bucket padding.
# ---------------------------------------------------------------------------


def test_build_trailing_silence_zero_duration():
    silence, source = ip.build_trailing_silence(np.zeros(5000, dtype=np.float32), ip.SILENCE_SOURCE_TAIL, 0.0)
    assert silence.shape == (0,)
    assert source == ip.SILENCE_SOURCE_NONE


@pytest.mark.parametrize("quiet_source", [ip.SILENCE_SOURCE_LEAD_IN, ip.SILENCE_SOURCE_TAIL])
def test_build_trailing_silence_tiles_quiet_window_and_passes_through_source(quiet_source):
    quiet_window = np.arange(1600, dtype=np.float32)  # a 100ms window at 16kHz
    silence, source = ip.build_trailing_silence(quiet_window, quiet_source, 0.3, sample_rate=16000)
    assert silence.shape == (4800,)
    assert source == quiet_source
    np.testing.assert_array_equal(silence[:1600], quiet_window)


def test_build_trailing_silence_falls_back_to_digital_zero_when_no_quiet_window():
    silence, source = ip.build_trailing_silence(None, ip.SILENCE_SOURCE_DIGITAL_ZERO, 1.0, sample_rate=16000)
    assert silence.shape == (16000,)
    assert source == ip.SILENCE_SOURCE_DIGITAL_ZERO
    np.testing.assert_array_equal(silence, np.zeros(16000, dtype=np.float32))


@pytest.mark.parametrize("bucket_s,expected_samples", [(0.3, 4800), (1.0, 16000), (2.0, 32000)])
def test_build_trailing_silence_bucket_lengths(bucket_s, expected_samples):
    quiet_window = np.ones(1600, dtype=np.float32)
    silence, _source = ip.build_trailing_silence(quiet_window, ip.SILENCE_SOURCE_LEAD_IN, bucket_s, sample_rate=16000)
    assert silence.shape == (expected_samples,)


# ---------------------------------------------------------------------------
# Output-path refusal.
# ---------------------------------------------------------------------------


def test_refuses_training_manifest_output_path(tmp_path):
    forbidden = tmp_path / "optionb" / "manifest.csv"
    forbidden.parent.mkdir()
    forbidden.write_text("x")
    with pytest.raises(ValueError):
        ip.assert_safe_manifest_output_path(forbidden, forbidden_path=forbidden)


def test_allows_a_different_output_path(tmp_path):
    forbidden = tmp_path / "optionb" / "manifest.csv"
    allowed = tmp_path / "probes" / "manifest.csv"
    ip.assert_safe_manifest_output_path(allowed, forbidden_path=forbidden)


def test_default_forbidden_path_is_the_optionb_manifest():
    with pytest.raises(ValueError):
        ip.assert_safe_manifest_output_path(ip.OPTIONB_MANIFEST)


# ---------------------------------------------------------------------------
# generate_probes: end-to-end with synthetic logp, plus every failure reason
# counted (never raised).
# ---------------------------------------------------------------------------


def _identity_resolve(row):
    return row["transcript"]


def test_generate_probes_success_path():
    text = "set the lights to red"
    prefix = "set the lights to"
    row = _row("a.wav", transcript=text)
    specs = [ip.ProbeSpec(source_row=row, prefix=prefix)]

    def load_waveform(_row):
        return np.zeros(32000, dtype=np.float32)

    def compute_logp(_waveform):
        return make_posterior(text)

    result = ip.generate_probes(
        specs,
        split="val",
        grace_frames=2,
        silence_buckets=(0.0, 0.3),
        aligner_checkpoint_path="ckpt.pt",
        aligner_checkpoint_sha256="deadbeef",
        load_waveform=load_waveform,
        compute_logp=compute_logp,
        resolve_and_normalize=_identity_resolve,
    )

    assert not result.failure_counts
    assert len(result.manifest_rows) == 2  # one per silence bucket
    assert len(result.audio) == 2
    buckets = sorted(r["trailing_silence_s"] for r in result.manifest_rows)
    assert buckets == [0.0, 0.3]
    for manifest_row in result.manifest_rows:
        assert manifest_row["prefix"] == prefix
        assert manifest_row["source_filename"] == "a.wav"
        assert manifest_row["char_overlap"] is False
        assert manifest_row["aligner_checkpoint"] == "ckpt.pt"
        assert manifest_row["aligner_checkpoint_sha256"] == "deadbeef"
        assert manifest_row["label"] == ""
        assert manifest_row["bucket"] == "incomplete_prefix"
        assert manifest_row["source_dataset"] == ip.SOURCE_DATASET


def test_generate_probes_counts_missing_audio_without_raising():
    row = _row("missing.wav", transcript="set the lights to red")
    specs = [ip.ProbeSpec(source_row=row, prefix="set the lights to")]

    def load_waveform(_row):
        raise FileNotFoundError("no such file")

    result = ip.generate_probes(
        specs,
        split="val",
        grace_frames=2,
        silence_buckets=(0.0,),
        aligner_checkpoint_path="ckpt.pt",
        aligner_checkpoint_sha256="deadbeef",
        load_waveform=load_waveform,
        compute_logp=lambda _w: make_posterior("x"),
        resolve_and_normalize=_identity_resolve,
    )

    assert result.failure_counts[ip.FAILURE_MISSING_AUDIO] == 1
    assert result.manifest_rows == []


def test_generate_probes_counts_force_align_error_without_raising():
    text = "set the lights to red"
    row = _row("a.wav", transcript=text)
    specs = [ip.ProbeSpec(source_row=row, prefix="set the lights to")]

    def load_waveform(_row):
        return np.zeros(320, dtype=np.float32)

    def compute_logp(_waveform):
        # Only 2 frames -- far too few for `text`'s length: force_align
        # must raise ValueError ("impossible CTC alignment").
        return np.zeros((2, vcm_alphabet.ALPHABET_SIZE))

    result = ip.generate_probes(
        specs,
        split="val",
        grace_frames=2,
        silence_buckets=(0.0,),
        aligner_checkpoint_path="ckpt.pt",
        aligner_checkpoint_sha256="deadbeef",
        load_waveform=load_waveform,
        compute_logp=compute_logp,
        resolve_and_normalize=_identity_resolve,
    )

    assert result.failure_counts[ip.FAILURE_FORCE_ALIGN] == 1
    assert result.manifest_rows == []


def test_generate_probes_counts_no_gap_and_crop_too_short_without_raising(monkeypatch):
    text = "set the lights to red"
    prefix = "set the lights to"
    rows = [_row("a.wav", transcript=text), _row("b.wav", transcript=text)]
    specs = [ip.ProbeSpec(source_row=r, prefix=prefix) for r in rows]

    outcomes = iter([ip.CropFailure(ip.FAILURE_NO_GAP), ip.CropFailure(ip.FAILURE_CROP_TOO_SHORT)])
    monkeypatch.setattr(ip, "compute_crop", lambda *a, **k: next(outcomes))

    result = ip.generate_probes(
        specs,
        split="val",
        grace_frames=2,
        silence_buckets=(0.0,),
        aligner_checkpoint_path="ckpt.pt",
        aligner_checkpoint_sha256="deadbeef",
        load_waveform=lambda _row: np.zeros(32000, dtype=np.float32),
        compute_logp=lambda _w: make_posterior(text),
        resolve_and_normalize=_identity_resolve,
    )

    assert result.failure_counts[ip.FAILURE_NO_GAP] == 1
    assert result.failure_counts[ip.FAILURE_CROP_TOO_SHORT] == 1
    assert result.manifest_rows == []
    assert len(result.failures) == 2


def test_generate_probes_missing_resolved_transcript_counts_as_missing_audio():
    row = _row("a.wav", transcript="")
    specs = [ip.ProbeSpec(source_row=row, prefix="set the lights to")]

    result = ip.generate_probes(
        specs,
        split="val",
        grace_frames=2,
        silence_buckets=(0.0,),
        aligner_checkpoint_path="ckpt.pt",
        aligner_checkpoint_sha256="deadbeef",
        load_waveform=lambda _row: np.zeros(32000, dtype=np.float32),
        compute_logp=lambda _w: make_posterior("x"),
        resolve_and_normalize=lambda _row: None,
    )

    assert result.failure_counts[ip.FAILURE_MISSING_AUDIO] == 1


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------


def test_build_generation_report_shape():
    result = ip.GenerationResult()
    result.failure_counts[ip.FAILURE_NO_GAP] = 2
    report = ip.build_generation_report([ip.ProbeSpec(_row("a.wav"), "remind")], result, "val")
    assert report["split"] == "val"
    assert report["source_rows_selected"] == 1
    assert report["failure_counts"] == {
        ip.FAILURE_FORCE_ALIGN: 0,
        ip.FAILURE_NO_GAP: 2,
        ip.FAILURE_CROP_TOO_SHORT: 0,
        ip.FAILURE_MISSING_AUDIO: 0,
    }


def test_build_generation_report_zero_fills_all_four_reasons_when_no_failures():
    report = ip.build_generation_report([], ip.GenerationResult(), "test")
    assert report["failure_counts"] == {reason: 0 for reason in ip.ALL_FAILURE_REASONS}


def test_build_audit_sample_caps_per_prefix_and_dedupes_source_rows():
    rows = []
    for i in range(10):
        for bucket in (0.0, 0.3):
            rows.append(
                {
                    "prefix": "remind",
                    "source_filename": f"r{i:02d}.wav",
                    "trailing_silence_s": bucket,
                }
            )
    sample = ip.build_audit_sample(rows, seed=0, per_prefix=3)
    assert len(sample) == 3
    assert len({r["source_filename"] for r in sample}) == 3
    for row in sample:
        assert row["trailing_silence_s"] == 0.3  # largest bucket chosen (padding must be audible)


# ---------------------------------------------------------------------------
# R2-5 real-data regression: every accepted padding window must actually
# measure >=20dB below speech level on real audio, and the digital-zero
# fallback must not have silently regressed to "always" (e.g. the tail
# search accidentally disabled). Skipped when the cluster checkpoint/
# manifest aren't present locally -- this is an integration check over real
# artifacts, not part of the synthetic-only unit suite above.
# ---------------------------------------------------------------------------

REAL_OPTIONB_MANIFEST = ip.OPTIONB_MANIFEST
REAL_CHECKPOINT = ip.DEFAULT_CHECKPOINT
_REAL_ARTIFACTS_MISSING = not (REAL_OPTIONB_MANIFEST.exists() and REAL_CHECKPOINT.exists())

# Ticket 05's own smoke runs (R2-1 fix, then this R2-5 fix) measured a
# digital-zero share of 81-84% over the full val split (798-row sample) --
# most 1.5-2s Option B clips simply don't have a lead-in *or* tail long
# enough to clear the 20-frame backoff plus the 100ms window and the 20dB
# bar at once. 0.95 is a loose smoke ceiling: it would catch a regression
# that disables the tail search entirely (reverting to R2-1's lead-in-only
# behavior pushed the share to ~99%+ on this same kind of sample), without
# being tight enough to flake on the small fixed sample this test uses.
MAX_DIGITAL_ZERO_SHARE = 0.95
REAL_SAMPLE_CAP_PER_PREFIX = 3


@pytest.mark.skipif(_REAL_ARTIFACTS_MISSING, reason="real optionb manifest/checkpoint not present locally")
def test_real_val_padding_windows_clear_the_20db_bar_and_digital_zero_is_bounded():
    import torch
    import torchaudio

    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform
    from me2_voicegen.vcm.text import normalize_text, resolve_transcript

    with REAL_OPTIONB_MANIFEST.open(newline="", encoding="utf-8") as f:
        import csv

        rows = list(csv.DictReader(f))

    def _resolve_and_normalize(row):
        transcript = resolve_transcript(dict(row))
        return None if transcript is None else normalize_text(transcript)

    specs = ip.select_target_rows(
        rows,
        OPTIONB_GRAMMAR.incomplete_prefixes,
        "val",
        _resolve_and_normalize,
        cap_per_prefix=REAL_SAMPLE_CAP_PER_PREFIX,
        seed=0,
    )
    assert specs, "expected at least one real val row matching a designated prefix"

    model, _checkpoint = load_checkpoint(REAL_CHECKPOINT, weights_only=True)
    feature_extractor = LogMelFeatureExtractor()
    audio_root = REAL_OPTIONB_MANIFEST.parent

    non_digital_zero = 0
    total = 0
    for spec in specs:
        wav_path = audio_root / spec.source_row["path"]
        waveform_t, _sr = torchaudio.load(str(wav_path))
        if waveform_t.dim() == 2:
            waveform_t = waveform_t.mean(dim=0)
        waveform = waveform_t.numpy().astype(np.float32)

        text = _resolve_and_normalize(spec.source_row)
        logp = logp_for_waveform(model, feature_extractor, torch.as_tensor(waveform))
        try:
            alignment = force_align(logp, text)
        except ValueError:
            continue

        total += 1
        window, source = ip.find_quiet_window(waveform, alignment)

        if source == ip.SILENCE_SOURCE_DIGITAL_ZERO:
            assert window is None
            continue

        non_digital_zero += 1
        speech_start = alignment.start_frame * 160
        speech_end = (alignment.end_frame + 1) * 160
        speech_rms = float(np.sqrt(np.mean(np.square(waveform[speech_start:speech_end]))))
        window_rms = float(np.sqrt(np.mean(np.square(window))))
        threshold = speech_rms * (10.0 ** (-ip.QUIET_RELATIVE_DB / 20.0))
        assert window_rms <= threshold + 1e-12, (
            f"{spec.source_row['filename']} ({source}): window RMS {window_rms:.6g} "
            f"exceeds the {ip.QUIET_RELATIVE_DB}dB-below-speech threshold {threshold:.6g}"
        )

    assert total > 0
    digital_zero_share = 1.0 - (non_digital_zero / total)
    assert digital_zero_share <= MAX_DIGITAL_ZERO_SHARE, (
        f"digital-zero share {digital_zero_share:.2%} exceeds the {MAX_DIGITAL_ZERO_SHARE:.0%} "
        f"smoke ceiling -- looks like the lead-in/tail quiet-window search regressed"
    )
