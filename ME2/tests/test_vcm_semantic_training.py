"""Head targets, joint loss and the heads training path (feature ctc-attention, ticket 02)."""

from __future__ import annotations

import json
from unittest import mock

import pytest
import torch

from me2_voicegen.vcm import train as train_mod
from me2_voicegen.vcm.dataset import VCMDataset, collate_fn
from me2_voicegen.vcm.model import build_model
from me2_voicegen.vcm.quartznet import QuartzNetConfig, QuartzNetCTC
from me2_voicegen.vcm.semantic_labels import IGNORE, INTENT_CLASSES, SLOT_INTENTS, SLOTS, row_targets
from me2_voicegen.vcm.train import HeadsMeter, heads_ce, joint_loss, parse_heads_loss_weights

ALARM_K = SLOT_INTENTS.index("ALARM")


# --- label mapping --------------------------------------------------------------

def test_phrase_row_gets_intent_and_slot():
    intent_id, slots, source = row_targets("ALARM", "alarm six am")
    assert INTENT_CLASSES[intent_id] == "ALARM" and source == "phrase"
    assert slots[ALARM_K] == SLOTS["ALARM"][1].index("6 AM")
    assert [v for k, v in enumerate(slots) if k != ALARM_K] == [IGNORE] * 5


def test_paraphrase_uses_manifest_slot_value_else_ignored():
    _, slots, source = row_targets("TIMER", "set the timer for one minute", "1 minute")
    assert source == "manifest_slot_value" and slots[SLOT_INTENTS.index("TIMER")] == SLOTS["TIMER"][1].index("1 minute")
    _, slots, source = row_targets("TIMER", "set the timer for one minute", "")
    assert source == "no_slot_target" and set(slots) == {IGNORE}


def test_nonslotted_nontarget_unmapped_and_mismatch():
    assert row_targets("STOP", "stop")[2] == "nonslotted"
    assert row_targets("unknown", "what is the name")[2] == "nontarget"
    assert row_targets("silence", None)[2] == "nontarget"
    iid, slots, source = row_targets("MYSTERY", "stop")
    assert (iid, source) == (IGNORE, "intent_unmapped") and set(slots) == {IGNORE}
    iid, slots, source = row_targets("ALARM", "stop")  # transcript is another intent's phrase
    assert INTENT_CLASSES[iid] == "ALARM" and source == "phrase_label_mismatch" and set(slots) == {IGNORE}


# --- dataset / collate ----------------------------------------------------------

@pytest.fixture
def labelled_manifest(vcm_fake_manifest_factory):
    def spec(label, transcript, split="train", slot_value="", bucket="target_commands"):
        return {"bucket": bucket, "source_dataset": "optionb", "label": label, "split": split,
                "transcript": transcript, "duration_s": 1.0, "slot_value": slot_value}
    rows = [
        spec("ALARM", "Alarm 6 AM"), spec("STOP", "Stop"), spec("TIMER", "Set the timer for one minute", slot_value="1 minute"),
        spec("TIMER", "Set the timer for a bit"), spec("unknown", "what is the name", bucket="babble"),
    ]
    rows += [spec("ALARM", "Alarm 8 AM", "val"), spec("STOP", "Stop", "val")]
    return vcm_fake_manifest_factory(rows)


def test_dataset_labels_and_counts(labelled_manifest):
    ds = VCMDataset(labelled_manifest, split="train", semantic_labels=True)
    assert dict(ds.semantic_label_counts) == {
        "phrase": 1, "nonslotted": 1, "manifest_slot_value": 1, "no_slot_target": 1, "nontarget": 1,
    }
    ex = ds[0]
    assert INTENT_CLASSES[ex.intent_id] == "ALARM" and len(ex.slot_targets) == len(SLOT_INTENTS)
    batch = collate_fn([ds[i] for i in range(5)], with_labels=True)
    assert batch["intent_id"].shape == (5,) and batch["slot_targets"].shape == (5, len(SLOT_INTENTS))
    assert batch["slot_targets"][1].tolist() == [IGNORE] * len(SLOT_INTENTS)  # STOP


def test_default_dataset_and_collate_unchanged(labelled_manifest):
    ds = VCMDataset(labelled_manifest, split="train")
    ex = ds[0]
    waveform, target_ids, target_len = ex  # the 3-tuple contract every existing caller relies on
    assert not hasattr(ex, "intent_id") and not ds.semantic_label_counts
    assert set(collate_fn([ex])) == {"features", "input_lengths", "target_ids", "target_len"}


# --- loss -----------------------------------------------------------------------

def _heads_model():
    return QuartzNetCTC(QuartzNetConfig(channels=16, epilogue_channels=16, head_dim=8, heads=True))


def test_loss_terms_finite_and_zero_when_unlabelled():
    model = _heads_model()
    out = model.forward_heads(torch.randn(3, 40, 60))
    ids = torch.tensor([INTENT_CLASSES.index("ALARM"), INTENT_CLASSES.index("STOP"), IGNORE])
    slots = torch.full((3, len(SLOT_INTENTS)), IGNORE)
    slots[0, ALARM_K] = 1
    terms = heads_ce(out, ids, slots)
    assert all(torch.isfinite(v) and v > 0 for v in terms.values())
    none = heads_ce(out, torch.full((3,), IGNORE), torch.full_like(slots, IGNORE))
    assert all(torch.isfinite(v) and v == 0 for v in none.values())


def test_ctc_weight_zero_gives_no_gradient_to_ctc_linear():
    model = _heads_model()
    out = model.forward_heads(torch.randn(2, 40, 60))
    ids = torch.tensor([INTENT_CLASSES.index("ALARM"), INTENT_CLASSES.index("STOP")])
    slots = torch.full((2, len(SLOT_INTENTS)), IGNORE)
    slots[0, ALARM_K] = 0
    terms = heads_ce(out, ids, slots)
    ctc = out.ctc_logits.log_softmax(-1).sum()  # stand-in; must stay out of the graph at weight 0
    joint_loss(ctc, terms, 0.0, {"intent": 0.3, "slot": 0.1}).backward()
    assert model.classifier.weight.grad is None
    assert model.intent_head.weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    out = model.forward_heads(torch.randn(2, 40, 60))
    joint_loss(out.ctc_logits.log_softmax(-1).sum(), heads_ce(out, ids, slots), 1.0, {"intent": 0.3, "slot": 0.1}).backward()
    assert model.classifier.weight.grad.abs().sum() > 0


def test_parse_heads_loss_weights():
    assert parse_heads_loss_weights("intent=0.5,slot=0") == {"intent": 0.5, "slot": 0.0}
    assert parse_heads_loss_weights("intent=1") == {"intent": 1.0, "slot": 0.1}
    with pytest.raises(ValueError):
        parse_heads_loss_weights("ctc=1")


def test_heads_meter_accuracies():
    class Out:
        pass

    out = Out()
    n_i = len(INTENT_CLASSES)
    out.intent_logits = torch.zeros(3, n_i)
    alarm, stop = INTENT_CLASSES.index("ALARM"), INTENT_CLASSES.index("STOP")
    out.intent_logits[0, alarm] = 5   # right
    out.intent_logits[1, stop] = 5    # right
    out.intent_logits[2, stop] = 5    # wrong (true: ALARM)
    from me2_voicegen.vcm.semantic_labels import slot_head_name
    out.slot_logits = {slot_head_name(n): torch.zeros(3, 3) for n in SLOT_INTENTS}
    out.slot_logits[slot_head_name("ALARM")][0, 2] = 5  # row 0 slot pred 2 (right), row 2 pred 0 (wrong)
    ids = torch.tensor([alarm, stop, alarm])
    slots = torch.full((3, len(SLOT_INTENTS)), IGNORE)
    slots[0, ALARM_K] = 2
    slots[2, ALARM_K] = 1
    meter = HeadsMeter()
    meter.add(out, ids, slots)
    s = meter.summary()
    assert s["n_intent"] == 3 and s["intent_acc"] == pytest.approx(2 / 3)
    assert s["n_slot"] == 2 and s["slot_acc"] == pytest.approx(1 / 2)
    assert s["n_exact"] == 3 and s["exact_acc"] == pytest.approx(2 / 3)  # row0 right, row1 STOP right, row2 wrong intent
    assert s["intent_ce"] > 0


# --- the training loop ----------------------------------------------------------

def _run(manifest, out_dir, *extra):
    train_mod.main(
        ["--manifest", str(manifest), "--out-dir", str(out_dir), "--max-epochs", "1", "--device", "cpu",
         "--num-workers", "0", "--batch-size", "2", "--max-minutes", "5", *extra]
    )
    return json.loads((out_dir / "metadata" / "loss_history.json").read_text())


def test_heads_training_logs_and_checkpoint(labelled_manifest, tmp_path):
    hist = _run(labelled_manifest, tmp_path / "a", "--preset", "quartznet5x3-heads")
    row = hist["history"][0]
    for key in ("ctc_loss_train", "intent_loss_train", "slot_loss_train", "intent_acc_val", "slot_acc_val",
                "exact_acc_val", "ctc_loss_val", "intent_loss_val", "slot_loss_val"):
        assert key in row and torch.isfinite(torch.tensor(row[key]))
    assert hist["heads"] and hist["selected_on"] == "ctc_val_loss" and hist["ctc_weight"] == 1.0
    assert hist["semantic_label_counts"]["train"]["no_slot_target"] == 1
    ckpt = torch.load(tmp_path / "a" / "checkpoints" / "checkpoint.pt", weights_only=True)
    assert ckpt["config"]["heads"] is True and ckpt["model_type"] == "quartznet"


def test_classifier_only_run_selects_on_heads_loss(labelled_manifest, tmp_path):
    hist = _run(labelled_manifest, tmp_path / "b", "--preset", "quartznet5x3-heads", "--ctc-weight", "0")
    assert hist["selected_on"] == "heads_val_loss" and hist["ctc_weight"] == 0.0
    row = hist["history"][0]
    assert row["select_loss"] == pytest.approx(0.3 * row["intent_loss_val"] + 0.1 * row["slot_loss_val"])


def test_head_flags_rejected_on_other_presets(labelled_manifest, tmp_path):
    with pytest.raises(SystemExit):
        _run(labelled_manifest, tmp_path / "x", "--preset", "quartznet5x3", "--ctc-weight", "0")


def test_non_heads_preset_never_touches_heads_path(labelled_manifest, tmp_path):
    with mock.patch.object(train_mod, "joint_loss") as jl, mock.patch.object(QuartzNetCTC, "forward_heads") as fh:
        hist = _run(labelled_manifest, tmp_path / "c", "--preset", "quartznet5x3")
    assert not jl.called and not fh.called and "heads" not in hist and "intent_acc_val" not in hist["history"][0]


def test_paraphrase_slot_value_match_ignores_case():
    _, slots, source = row_targets("COLOR", "make it a bit more red please", "Red")
    assert source == "manifest_slot_value" and slots[SLOT_INTENTS.index("COLOR")] == SLOTS["COLOR"][1].index("red")
    _, slots, source = row_targets("CREATE_REMINDER", "remind me about study time", "Study")
    assert source == "manifest_slot_value" and slots[SLOT_INTENTS.index("CREATE_REMINDER")] == SLOTS["CREATE_REMINDER"][1].index("study")
