"""Attention-pooled intent/slot heads on the QuartzNet encoder (feature ctc-attention, ticket 01)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from me2_voicegen.vcm import semantic_labels
from me2_voicegen.vcm.model import PRESETS, build_model, build_model_from_config, model_type_for_config, param_count
from me2_voicegen.vcm.quartznet import AttentivePool, QuartzNetConfig, QuartzNetCTC
from me2_voicegen.vcm.semantic_labels import INTENT_CLASSES, SLOTS, slot_head_name

SMALL = dict(channels=16, epilogue_channels=16, head_dim=8)


def _small(heads: bool, **kw) -> QuartzNetCTC:
    return QuartzNetCTC(QuartzNetConfig(heads=heads, **{**SMALL, **kw})).eval()


# --- label tables: pinned, so a grammar change fails loudly --------------------

def test_label_tables_match_grammar():
    assert len(INTENT_CLASSES) == 21
    assert INTENT_CLASSES[-2:] == ("unknown", "silence")
    assert set(INTENT_CLASSES[:-2]) == {
        "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS",
        "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN",
        "VOLUME_UP", "WEATHER",
    }
    assert SLOTS == {
        "ALARM": ("ALARM_TIME", ("6 AM", "8 AM", "9 PM")),
        "BRIGHTNESS": ("PERCENT", ("100 percent", "20 percent", "60 percent")),
        "COLOR": ("COLOR", ("blue", "green", "red")),
        "CREATE_REMINDER": ("TASK", ("drink water", "exercise", "study")),
        "TEMPERATURE": ("DEGREES", ("18 degrees", "22 degrees", "26 degrees")),
        "TIMER": ("DURATION", ("1 minute", "10 seconds", "30 seconds")),
    }


def test_derivation_rejects_two_slots_per_intent(monkeypatch):
    class FakeGrammar:
        @staticmethod
        def all_phrases():
            return [("a", "X", {"A": "1", "B": "2"})]

    monkeypatch.setattr(semantic_labels, "OPTIONB_GRAMMAR", FakeGrammar)
    with pytest.raises(ValueError, match="exactly one"):
        semantic_labels._derive()


# --- back compatibility ---------------------------------------------------------

def test_old_config_dict_still_builds():
    old = dataclasses.asdict(PRESETS["quartznet5x3"])
    for key in ("heads", "head_dim", "pooling"):
        old.pop(key)
    model = build_model_from_config(old, "quartznet")
    assert not model.config.heads and not hasattr(model, "pool")


def test_heads_off_adds_no_parameters():
    assert param_count(build_model("quartznet5x3")) == 911_189


def test_ctc_logits_identical_with_heads_on_and_off():
    off = _small(False)
    on = _small(True)
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected and all(k.split(".")[0] in {"pool", "intent_head", "slot_heads"} for k in missing)
    x = torch.randn(3, 40, 97)
    assert torch.equal(off(x), on(x))
    assert torch.equal(on(x), on.forward_heads(x).ctc_logits)


# --- shapes ---------------------------------------------------------------------

def test_forward_heads_shapes():
    model = _small(True)
    out = model.forward_heads(torch.randn(2, 40, 151))
    assert out.ctc_logits.shape == (2, 76, 29)
    assert out.intent_logits.shape == (2, 21)
    assert out.attention.shape == (2, 76)
    assert set(out.slot_logits) == {slot_head_name(i) for i in SLOTS} and len(out.slot_logits) == 6
    assert all(v.shape == (2, 3) for v in out.slot_logits.values())


def test_forward_heads_requires_heads():
    with pytest.raises(RuntimeError):
        _small(False).forward_heads(torch.randn(1, 40, 50))


# --- masking --------------------------------------------------------------------

def test_attention_sums_to_one_and_zero_on_padding():
    model = _small(True)
    lengths = torch.tensor([151, 80, 31])
    out = model.forward_heads(torch.randn(3, 40, 151), lengths)
    valid = model.output_lengths(lengths)
    for i, n in enumerate(valid.tolist()):
        assert out.attention[i, :n].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.all(out.attention[i, n:] == 0)


@pytest.mark.parametrize("attention", [True, False])
def test_pool_ignores_padded_frames(attention):
    """Pool of a clip alone == pool of the same clip inside a padded batch, whatever the padding holds."""
    pool = AttentivePool(16, 8, attention=attention).eval()
    clip = torch.randn(1, 20, 16)
    alone, _ = pool(clip)
    padded = torch.cat([clip, 100 * torch.randn(1, 15, 16)], dim=1)
    in_batch, _ = pool(padded, torch.tensor([20]))
    assert torch.allclose(alone, in_batch, atol=1e-6)


def test_model_pools_only_valid_encoder_frames():
    """With the same encoder output, garbage past `output_lengths` cannot reach the intent logits.
    (Whole-model equality of padded vs unpadded is not expected: the conv encoder's receptive
    field sees the padding near the clip end.)"""
    model = _small(True)
    x = torch.randn(2, 40, 120)
    lengths = torch.tensor([120, 60])
    out = model.forward_heads(x, lengths)
    enc = model._encode(x)
    n = int(model.output_lengths(lengths)[1])
    enc_garbage = enc.clone()
    enc_garbage[1, n:] = 1e3
    pooled_a, _ = model.pool(enc, model.output_lengths(lengths))
    pooled_b, _ = model.pool(enc_garbage, model.output_lengths(lengths))
    assert torch.equal(pooled_a, pooled_b)
    assert torch.allclose(model.intent_head(pooled_a), out.intent_logits, atol=1e-6)


def test_mean_pooling_is_uniform_over_valid_frames():
    model = _small(True, pooling="mean")
    lengths = torch.tensor([100, 40])
    out = model.forward_heads(torch.randn(2, 40, 100), lengths)
    n = int(model.output_lengths(lengths)[1])
    assert torch.allclose(out.attention[1, :n], torch.full((n,), 1.0 / n))
    assert model.pool.score is None


def test_none_lengths_means_all_frames_valid():
    model = _small(True)
    x = torch.randn(2, 40, 100)
    full = torch.full((2,), 100)
    a, b = model.forward_heads(x), model.forward_heads(x, full)
    assert torch.equal(a.intent_logits, b.intent_logits)


# --- config / preset / checkpoint -----------------------------------------------

def test_config_validation():
    with pytest.raises(ValueError):
        QuartzNetConfig(pooling="max")
    with pytest.raises(ValueError):
        QuartzNetConfig(head_dim=0)


def test_preset_and_factory_round_trip():
    config = PRESETS["quartznet5x3-heads"]
    assert config.heads and model_type_for_config(config) == "quartznet"
    model = build_model("quartznet5x3-heads")
    assert isinstance(model, QuartzNetCTC) and model.config.heads
    rebuilt = build_model_from_config(dataclasses.asdict(config), "quartznet")
    assert param_count(rebuilt) == param_count(model)


def test_heads_config_weights_only_round_trip(tmp_path):
    model = _small(True)
    path = tmp_path / "c.pt"
    torch.save(
        {"config": dataclasses.asdict(model.config), "model_type": "quartznet", "model_state_dict": model.state_dict()},
        path,
    )
    ckpt = torch.load(path, weights_only=True)
    rebuilt = build_model_from_config(ckpt["config"], ckpt["model_type"]).eval()
    rebuilt.load_state_dict(ckpt["model_state_dict"])
    x = torch.randn(1, 40, 60)
    assert torch.equal(model.forward_heads(x).intent_logits, rebuilt.forward_heads(x).intent_logits)


def test_added_parameter_budget():
    added = param_count(build_model("quartznet5x3-heads")) - param_count(build_model("quartznet5x3"))
    assert added == 43_048  # pool 33,025 + intent 5,397 + 6 slot heads x 771; a change here is deliberate


# --- streaming-shaped windows (docs/STREAMING-CONTRACT.md): batch 1, no padding, 0.3 s .. past window_s ----

@pytest.mark.parametrize("seconds", [0.3, 0.5, 1.0, 2.5, 4.0, 6.0])
def test_unpadded_streaming_windows(seconds):
    model = _small(True)
    out = model.forward_heads(torch.randn(1, 40, int(seconds * 100) + 1))  # lengths=None, as the streaming path calls it
    n = out.ctc_logits.shape[1]
    assert n >= 15 if seconds == 0.3 else n >= 1
    assert out.attention.shape == (1, n) and out.attention.sum().item() == pytest.approx(1.0, abs=1e-5)
    for t in (out.ctc_logits, out.intent_logits, *out.slot_logits.values()):
        assert torch.isfinite(t).all()


def test_wide_heads_preset_is_3_to_5_mb_int8_and_runs():
    from me2_voicegen.vcm.model import PRESETS, build_model, estimated_int8_bytes

    assert PRESETS["quartznet5x3-wide-heads"].heads is True
    model = build_model("quartznet5x3-wide-heads").eval()
    assert 3.0e6 <= estimated_int8_bytes(model) <= 5.0e6
    assert estimated_int8_bytes(model) > 3 * estimated_int8_bytes(build_model("quartznet5x3-heads"))
    feats = torch.randn(2, 40, 120)
    out = model.forward_heads(feats, torch.tensor([120, 90]))
    assert out.ctc_logits.shape[0] == 2 and out.intent_logits.shape == (2, 21)


def test_xl_heads_preset_is_about_10_mb_int8_and_runs():
    from me2_voicegen.vcm.model import PRESETS, build_model, estimated_int8_bytes

    assert PRESETS["quartznet5x3-xl-heads"].heads is True
    model = build_model("quartznet5x3-xl-heads").eval()
    assert 9.5e6 <= estimated_int8_bytes(model) <= 10.0e6
    out = model.forward_heads(torch.randn(2, 40, 120), torch.tensor([120, 90]))
    assert out.ctc_logits.shape[0] == 2 and out.intent_logits.shape == (2, 21)
