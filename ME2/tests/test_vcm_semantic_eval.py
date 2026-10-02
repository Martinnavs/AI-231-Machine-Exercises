"""Metric code of the ctc-attention scoring (feature ctc-attention, ticket 03): known rows -> known counts."""

from __future__ import annotations

import csv

import pytest
import torch

from me2_voicegen.vcm import semantic_eval as se
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.quartznet import QuartzNetConfig, QuartzNetCTC
from me2_voicegen.vcm.semantic_labels import INTENT_CLASSES, SLOT_INTENTS, SLOTS


def probs(top: str, conf: float = 0.9, second: str | None = None):
    p = [(1 - conf) / (len(INTENT_CLASSES) - 1 - (1 if second else 0))] * len(INTENT_CLASSES)
    p[INTENT_CLASSES.index(top)] = conf
    if second:
        p[INTENT_CLASSES.index(second)] = (1 - conf) / 2
    total = sum(p)
    return [x / total for x in p]


def row(bucket="target_commands", label="STOP", ctc=None, p=None, slot_value="", slot_pred=None):
    return {"i": 0, "bucket": bucket, "label": label, "source_dataset": "optionb", "variation_match": "exact",
            "slot_value": slot_value, "ctc": ctc, "probs": p, "slot_pred": slot_pred}


def ctc(intent, conf=-0.05, slots=None):
    return {"intent": intent, "confidence": conf, "slots": slots or {}}


def test_ctc_metrics_counts():
    rows = [
        row(label="STOP", ctc=ctc("STOP")),                                                  # right
        row(label="STOP", ctc=ctc("PAUSE")),                                                 # wrong intent
        row(label="STOP", ctc=ctc("STOP", conf=-0.5)),                                       # below threshold: rejected
        row(label="STOP", ctc=ctc(None, conf=None)),                                         # gate rejected
        row(label="ALARM", slot_value="6 AM", ctc=ctc("ALARM", slots={"ALARM_TIME": "6 AM"})),   # intent+slot right
        row(label="ALARM", slot_value="6 AM", ctc=ctc("ALARM", slots={"ALARM_TIME": "8 AM"})),   # intent right, slot wrong
        row("babble", "unknown", ctc=ctc("STOP")),                                           # false accept
        row("babble", "unknown", ctc=ctc(None, None)),
        row("silence", "silence", ctc=ctc("TIME", conf=-0.5)),                               # below threshold
    ]
    m = se.ctc_metrics(rows, -0.1)
    assert m == {"n_targets": 6, "intent_ok": 3, "exact_ok": 2, "n_babble": 2, "babble_fa": 1, "n_silence": 1, "silence_fa": 0}


def test_cls_metrics_and_slot_head():
    rows = [
        row(label="STOP", p=probs("STOP", 0.9)),
        row(label="STOP", p=probs("STOP", 0.4)),                                              # below threshold, right by argmax
        row(label="STOP", p=probs("PAUSE", 0.9)),
        row(label="ALARM", slot_value="6 AM", p=probs("ALARM"), slot_pred={"ALARM": "6 AM"}),
        row(label="ALARM", slot_value="6 AM", p=probs("ALARM"), slot_pred={"ALARM": "9 PM"}),
        row("babble", "unknown", p=probs("STOP", 0.95)),                                     # false accept
        row("babble", "unknown", p=probs("unknown", 0.99)),                                  # classifier says unknown: not an accept
        row("silence", "silence", p=probs("silence", 0.99)),
    ]
    m = se.cls_metrics(rows, 0.5)
    assert (m["n_targets"], m["intent_ok"], m["exact_ok"]) == (5, 3, 2)
    assert (m["argmax_intent_ok"], m["argmax_exact_ok"]) == (4, 3)
    assert (m["babble_fa"], m["silence_fa"]) == (1, 0)


def test_choose_threshold_matches_budget():
    nt = [row("babble", "unknown", p=probs("STOP", c)) for c in (0.9, 0.8, 0.5, 0.3)]
    nt.append(row("silence", "silence", p=probs("silence", 0.99)))  # never an accept; still counts in the denominator (5 rows)
    t = se.choose_cls_threshold(nt, budget_rate=0.4)                  # floor(0.4*5) = 2 allowed
    assert sum(se.cls_accepted(r, t) for r in nt) == 2
    assert sum(se.cls_accepted(r, se.choose_cls_threshold(nt, 0.0)) for r in nt) == 0
    assert se.choose_cls_threshold(nt, 1.0) == 0.0


def test_topk_confusion_agreement():
    rows = [
        row(label="TIME", p=probs("TIMER", 0.6, second="TIME"), ctc=ctc("TIME")),      # cls wrong top-1, right in top-3; CTC right
        row(label="TIMER", p=probs("TIMER", 0.9), ctc=ctc("TIMER")),
        row(label="PAUSE", p=probs("STOP", 0.9), ctc=ctc(None, None)),                 # CTC rejected
    ]
    assert se.topk_coverage(rows, 1) == 1 and se.topk_coverage(rows, 3) == 2
    cm = se.confusion(rows, se.cls_predict(0.3))
    assert cm["TIME"] == {"TIME": 0, "TIMER": 1, "other": 0, "reject": 0}
    assert cm["PAUSE"] == {"PAUSE": 0, "STOP": 1, "other": 0, "reject": 0}
    assert se.confusion(rows, se.ctc_predict(-0.1))["PAUSE"]["reject"] == 1
    a = se.agreement(rows, -0.1)
    assert (a["n_ctc_accepted"], a["agree_when_ctc_accepted"]) == (2, 1)
    assert (a["n_ctc_correct"], a["cls_top1_on_ctc_correct"], a["cls_top3_on_ctc_correct"]) == (2, 1, 2)


def test_gate_a_pass_and_fail():
    def m(exact, babble_fa):
        return {"n_targets": 1000, "exact_ok": exact, "babble_fa": babble_fa, "n_babble": 47}
    base = {"clean": m(980, 0), "noisy": m(950, 0)}
    ok = se.gate_a(base, {"clean": m(976, 0), "noisy": m(946, 1)}, 47)
    assert ok["pass"] and ok["noisy_babble_fa"]["allowed"] == 1 and ok["clean_exact"]["floor"] == 975
    bad = se.gate_a(base, {"clean": m(970, 0), "noisy": m(950, 2)}, 47)
    assert not bad["pass"] and not bad["clean_exact"]["pass"] and not bad["noisy_babble_fa"]["pass"]


def test_build_eval_manifest(tmp_path):
    ai = tmp_path / "ai231"
    (tmp_path / "noise" / "audio").mkdir(parents=True)
    ai.mkdir()
    fields = ["filename", "path", "bucket", "label", "source_dataset", "split", "variation_match", "slot_value"]
    def write(path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    base = dict(filename="x", source_dataset="optionb", slot_value="")
    write(ai / "manifest.csv", [
        dict(base, path="a/1.wav", bucket="target_commands", label="STOP", split="test", variation_match="exact"),
        dict(base, path="a/2.wav", bucket="target_commands", label="STOP", split="test", variation_match="close 0.8"),
        dict(base, path="a/3.wav", bucket="babble", label="unknown", split="test", variation_match=""),
        dict(base, path="a/4.wav", bucket="target_commands", label="STOP", split="train", variation_match="exact"),
    ])
    write(tmp_path / "noise" / "manifest.csv", [
        dict(base, path="audio/n.wav", bucket="silence", label="silence", split="test", source_dataset="background_noise"),
        dict(base, path="audio/m.wav", bucket="silence", label="silence", split="train", source_dataset="background_noise"),
    ])
    counts = se.build_eval_manifest(ai / "manifest.csv", tmp_path / "noise" / "manifest.csv", ai / "manifest.eval.csv")
    assert counts == {"ai231_rows": 2, "noise_rows": 1}
    rows = list(csv.DictReader((ai / "manifest.eval.csv").open()))
    assert [r["path"] for r in rows] == ["a/1.wav", "a/3.wav", "../noise/audio/n.wav"]


@pytest.fixture
def eval_dataset(vcm_fake_manifest_factory):
    def spec(bucket, label, source="optionb", split="test", transcript="stop"):
        return {"bucket": bucket, "source_dataset": source, "label": label, "split": split, "transcript": transcript, "duration_s": 1.0}
    rows = [spec("target_commands", "STOP"), spec("target_commands", "TIME", transcript="time"), spec("babble", "unknown", transcript="hello there"),
            spec("silence", "silence", source="background_noise", transcript="")]
    return VCMDataset(vcm_fake_manifest_factory(rows), split="test", augmenter=None)


def test_score_rows_clean_shards_and_fields(eval_dataset):
    from me2_voicegen.common.features import LogMelFeatureExtractor
    model = QuartzNetCTC(QuartzNetConfig(channels=16, epilogue_channels=16, head_dim=8, heads=True)).eval()
    ex = LogMelFeatureExtractor()
    run = lambda **kw: list(se.score_rows(model, ex, eval_dataset, noisy_seed=None, margin=4.0, beam_width=5, **kw))
    full = run()
    assert [r["i"] for r in full] == [0, 1, 2, 3]
    r = full[0]
    assert len(r["probs"]) == len(INTENT_CLASSES) and abs(sum(r["probs"]) - 1) < 1e-5
    assert set(r["slot_pred"]) == set(SLOT_INTENTS) and all(v in SLOTS[k][1] for k, v in r["slot_pred"].items())
    assert r["ctc"] is not None and set(r["ctc"]) == {"intent", "confidence", "slots"}
    s0, s1 = run(shard=(0, 2)), run(shard=(1, 2))
    assert sorted(x["i"] for x in s0 + s1) == [0, 1, 2, 3] and not {x["i"] for x in s0} & {x["i"] for x in s1}
    assert run(shard=(0, 1), do_ctc=False)[0]["ctc"] is None and len(run(limit=2)) == 2


def test_score_rows_noisy_is_deterministic_and_differs_from_clean(eval_dataset):
    from me2_voicegen.common.features import LogMelFeatureExtractor
    model = QuartzNetCTC(QuartzNetConfig(channels=16, epilogue_channels=16, head_dim=8, heads=True)).eval()
    ex = LogMelFeatureExtractor()
    noisy = lambda seed: list(se.score_rows(model, ex, eval_dataset, noisy_seed=seed, margin=None, beam_width=5, do_ctc=False, rir_pool_size=3))
    a, b = noisy(0), noisy(0)
    clean = list(se.score_rows(model, ex, eval_dataset, noisy_seed=None, margin=None, beam_width=5, do_ctc=False))
    assert [x["probs"] for x in a] == [x["probs"] for x in b]
    assert [x["probs"] for x in a] != [x["probs"] for x in clean]


def test_report_end_to_end_and_markdown_leads_with_clean(tmp_path):
    import json

    def dump(run, split, cond, rows):
        d = tmp_path / run
        d.mkdir(exist_ok=True)
        (d / f"{split}_{cond}.shard0of1.jsonl").write_text("\n".join(json.dumps(dict(r, i=i)) for i, r in enumerate(rows)))

    def rows_for(joint: bool, wrong: int):
        rs = [row(label="STOP", ctc=ctc("STOP"), p=probs("STOP")) for _ in range(10 - wrong)]
        rs += [row(label="STOP", ctc=ctc("PAUSE"), p=probs("PAUSE")) for _ in range(wrong)]
        rs += [row("babble", "unknown", ctc=ctc(None, None), p=probs("unknown", 0.99))]
        return rs
    for run, wrong in (("base", 0), ("joint", 0)):
        for split in ("val", "test"):
            for cond in ("clean", "noisy"):
                dump(run, split, cond, rows_for(True, wrong))
    rep = se.build_report({"base": tmp_path / "base", "joint": tmp_path / "joint"}, -0.1, "joint")
    assert rep["ctc"]["joint"]["clean"]["exact_ok"] == 10 and rep["cls"]["joint"]["clean"]["intent_ok"] == 10
    gates = se.gate_a(rep["ctc"]["base"], rep["ctc"]["joint"], 1)
    md = se.render_markdown(rep, "base", "joint", gates)
    assert md.index("Headline: unperturbed") < md.index("Robustness: perturbed") and "Gate A: PASS" in md


def test_slot_value_case_is_ignored():
    """Manifest slot values are capitalised ('Red'); the grammar/heads use lowercase ('red')."""
    r = row(label="COLOR", slot_value="Red", p=probs("COLOR"), slot_pred={"COLOR": "red"})
    assert se.cls_correct(r, 0.5, with_slot=True)
    assert not se.cls_correct(dict(r, slot_pred={"COLOR": "blue"}), 0.5, with_slot=True)


def test_slot_truth_parsed_from_transcript_when_no_slot_value_column():
    base = {"bucket": "target_commands", "source_dataset": "optionb"}
    assert se._slot_truth(dict(base, label="ALARM", transcript="Alarm 6 AM")) == ("6 AM", False)
    assert se._slot_truth(dict(base, label="ALARM", transcript="gibberish")) == ("", True)   # slotted, truth unknown: counted
    assert se._slot_truth(dict(base, label="STOP", transcript="Stop")) == ("", False)        # nothing to parse
    assert se._slot_truth(dict(base, label="COLOR", slot_value="Red")) == ("Red", False)     # column wins when present


def test_fixed_thresholds_report_and_by_source(tmp_path):
    import json
    d = tmp_path / "m"
    d.mkdir()
    rows = [dict(row(label="STOP", ctc=ctc("STOP"), p=probs("STOP")), source_dataset="vcm_balanced", i=0),
            dict(row(label="STOP", ctc=ctc("PAUSE"), p=probs("PAUSE")), source_dataset="optionb", i=1),
            dict(row("babble", "unknown", ctc=ctc(None, None), p=probs("unknown", 0.99)), i=2)]
    for cond in ("clean", "noisy"):
        (d / f"test_{cond}.shard0of1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    rep = se.build_report({"m": d}, -0.1, fixed_cls_thresholds={"m": 0.5})   # no val rows anywhere: nothing is chosen
    assert rep["budget"] is None and rep["cls"]["m"]["threshold"] == 0.5
    assert rep["by_source"]["m"]["clean"]["vcm_balanced"]["ctc_exact"] == 1 and rep["by_source"]["m"]["clean"]["optionb"]["ctc_exact"] == 0
    assert "By source" in se.render_markdown(rep, "m", "m", None)
