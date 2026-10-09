"""Whole-clip scoring of the CTC path and the attention heads on the same clips (feature ctc-attention, ticket 03).

Three steps, each a subcommand:

  build-manifest  ai231 manifest -> eval manifest: `exact` target rows + babble rows, plus the ESC-50
                  `background_noise` rows of an older manifest. `vcm.noisy_eval` draws its noise from the
                  split's own `background_noise` rows and the ai231 manifest has none.
  score           one checkpoint -> per-row JSONL (CTC decode at the most permissive threshold, and for a
                  heads checkpoint the intent softmax + slot argmaxes), clean or fixed-seed noisy, shardable.
                  The noisy perturbation is `vcm.noisy_eval`'s own (same seed => same signal for every model).
  report          JSONL of several checkpoints -> metrics tables, the gates, JSON + Markdown.

Thresholds are never swept on noisy data: the CTC threshold is a fixed argument (-0.1) and the classifier's
max-softmax threshold is chosen on clean val to match the CTC path's clean-val false-accept rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterator

import torch

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.evaluate import NEG_INF_THRESHOLD, _true_slots_for_row
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.noisy_eval import apply_perturbation, build_perturbations, build_rir_pool_for_seed
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.pipeline import infer_waveform, load_checkpoint
from me2_voicegen.vcm.semantic_labels import EXTRA_INTENT_CLASSES, INTENT_CLASSES, SLOT_INTENTS, SLOTS

CONFUSABLE_PAIRS = (("TIME", "TIMER"), ("PAUSE", "STOP"), ("LIGHT_ON", "LIGHT_OFF"))
EVAL_SPLITS = ("val", "test", "holdout")
NOISE_SPLITS = ("val", "test")


# --------------------------------------------------------------------------- manifest

def build_eval_manifest(ai231_manifest: Path, noise_manifest: Path | None, out: Path) -> dict[str, int]:
    """Write `out` (must sit next to `ai231_manifest` so its relative paths stay valid).

    With `noise_manifest=None` the noisy gate's noise pool is the dataset's own `background_noise` rows (the
    `noise_only` synthetic negatives of the val/test splits); no external noise is added."""
    with ai231_manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [
            r
            for r in reader
            if r["split"] in EVAL_SPLITS
            and (r["bucket"] != "target_commands" or r.get("variation_match", "").startswith("exact"))
        ]
    noise: list[dict] = []
    if noise_manifest is not None:
        noise_dir = noise_manifest.parent
        with noise_manifest.open(newline="", encoding="utf-8") as f:
            noise = [r for r in csv.DictReader(f) if r["source_dataset"] == "background_noise" and r["split"] in NOISE_SPLITS]
        for r in noise:
            r["path"] = os.path.relpath(os.path.realpath(noise_dir / r["path"]), os.path.realpath(out.parent))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + noise)
    return {"ai231_rows": len(rows), "noise_rows": len(noise)}


# --------------------------------------------------------------------------- scoring

def _slot_truth(row: dict) -> tuple[str, bool]:
    """`(slot value, unparseable)`. ai231 rows carry `slot_value`; older manifests do not, so it is parsed from the
    row's own transcript through the grammar (as `vcm.evaluate` does). `unparseable` marks a slotted-intent row whose
    slot truth could not be established; such rows are counted, and `report` shows how many there are."""
    if row.get("slot_value"):
        return row["slot_value"], False
    label = row["label"]
    if row["bucket"] != "target_commands" or label not in SLOTS:
        return "", False
    slots = _true_slots_for_row(row, OPTIONB_GRAMMAR)
    value = (slots or {}).get(SLOTS[label][0])
    return (value, False) if value else ("", True)


@torch.no_grad()
def score_rows(
    model,
    feature_extractor: LogMelFeatureExtractor,
    dataset: VCMDataset,
    *,
    noisy_seed: int | None,
    margin: float | None,
    beam_width: int = 50,
    shard: tuple[int, int] = (0, 1),
    do_ctc: bool = True,
    limit: int | None = None,
    rir_pool_size: int | None = None,
) -> Iterator[dict]:
    """One JSONL-ready dict per dataset row of this shard. `noisy_seed=None` = clean."""
    has_heads = bool(getattr(model.config, "heads", False))
    perturbations = rir_pool = noise_pool = None
    if noisy_seed is not None:
        kwargs = {} if rir_pool_size is None else {"pool_size": rir_pool_size}
        rir_pool = build_rir_pool_for_seed(noisy_seed, **kwargs)
        perturbations = build_perturbations(dataset, rir_pool, noisy_seed)
        noise_pool = dataset._noise_pool()
    index, n_shards = shard
    for i in range(len(dataset)):
        if i % n_shards != index or (limit is not None and i >= limit):
            continue
        row = dataset.rows[i]
        waveform = dataset[i].waveform
        if perturbations is not None:
            rir_i, noise_i, snr = perturbations[i]
            waveform = apply_perturbation(waveform, rir_pool[rir_i], noise_pool[noise_i], snr)
        slot_value, slot_unparseable = _slot_truth(row)
        record = {
            "i": i,
            "bucket": row["bucket"],
            "label": row["label"],
            "source_dataset": row["source_dataset"],
            "variation_match": row.get("variation_match", ""),
            "slot_value": slot_value,
            "slot_unparseable": slot_unparseable,
            "ctc": None,
            "probs": None,
            "slot_pred": None,
        }
        if do_ctc:
            d = infer_waveform(
                model, feature_extractor, waveform, OPTIONB_GRAMMAR, threshold=NEG_INF_THRESHOLD,
                beam_width=beam_width, required_command_margin=margin,
            )
            record["ctc"] = {
                "intent": d.intent,
                "confidence": None if d.intent is None else float(d.confidence),
                "slots": {k: str(v) for k, v in d.slots.items()},
            }
        if has_heads:
            out = model.forward_heads(feature_extractor(waveform).unsqueeze(0))
            record["probs"] = out.intent_logits[0].softmax(-1).tolist()
            record["slot_pred"] = {
                name: SLOTS[name][1][int(out.slot_logits[f"slot_{name}_{SLOTS[name][0]}"][0].argmax())]
                for name in SLOT_INTENTS
            }
        yield record


# --------------------------------------------------------------------------- metrics (pure functions over row dicts)

def is_target(r: dict) -> bool:
    return r["bucket"] == "target_commands"


def is_babble(r: dict) -> bool:
    return r["bucket"] == "babble"


def is_noise_silence(r: dict) -> bool:
    return r["bucket"] == "silence"


def ctc_accepted(r: dict, threshold: float) -> bool:
    c = r["ctc"]
    return bool(c and c["intent"] is not None and c["confidence"] is not None and c["confidence"] >= threshold)


def ctc_correct(r: dict, threshold: float, *, with_slot: bool) -> bool:
    if not ctc_accepted(r, threshold) or r["ctc"]["intent"] != r["label"]:
        return False
    if not with_slot or not r["slot_value"]:
        return True
    return r["slot_value"].lower() in [v.lower() for v in r["ctc"]["slots"].values()]


def top_intent(r: dict) -> tuple[str, float]:
    k = max(range(len(r["probs"])), key=r["probs"].__getitem__)
    return INTENT_CLASSES[k], r["probs"][k]


def cls_accepted(r: dict, threshold: float) -> bool:
    name, conf = top_intent(r)
    return name not in EXTRA_INTENT_CLASSES and conf >= threshold


def cls_correct(r: dict, threshold: float | None, *, with_slot: bool) -> bool:
    """`threshold=None`: plain argmax, no rejection (a pure classification accuracy)."""
    name, _ = top_intent(r)
    if name != r["label"] or (threshold is not None and not cls_accepted(r, threshold)):
        return False
    if not with_slot or not r["slot_value"]:
        return True
    # manifest slot values are capitalised ("Red", "Drink water"); the grammar's are lowercase
    return str(r["slot_pred"].get(name)).lower() == r["slot_value"].lower()


def rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


def ctc_metrics(rows: list[dict], threshold: float) -> dict:
    targets = [r for r in rows if is_target(r)]
    babble = [r for r in rows if is_babble(r)]
    silence = [r for r in rows if is_noise_silence(r)]
    return {
        "n_targets": len(targets),
        "intent_ok": sum(ctc_correct(r, threshold, with_slot=False) for r in targets),
        "exact_ok": sum(ctc_correct(r, threshold, with_slot=True) for r in targets),
        "n_babble": len(babble),
        "babble_fa": sum(ctc_accepted(r, threshold) for r in babble),
        "n_silence": len(silence),
        "silence_fa": sum(ctc_accepted(r, threshold) for r in silence),
    }


def cls_metrics(rows: list[dict], threshold: float) -> dict:
    targets = [r for r in rows if is_target(r)]
    babble = [r for r in rows if is_babble(r)]
    silence = [r for r in rows if is_noise_silence(r)]
    return {
        "n_targets": len(targets),
        "threshold": threshold,
        "intent_ok": sum(cls_correct(r, threshold, with_slot=False) for r in targets),
        "exact_ok": sum(cls_correct(r, threshold, with_slot=True) for r in targets),
        "argmax_intent_ok": sum(cls_correct(r, None, with_slot=False) for r in targets),
        "argmax_exact_ok": sum(cls_correct(r, None, with_slot=True) for r in targets),
        "n_babble": len(babble),
        "babble_fa": sum(cls_accepted(r, threshold) for r in babble),
        "n_silence": len(silence),
        "silence_fa": sum(cls_accepted(r, threshold) for r in silence),
    }


def choose_cls_threshold(val_rows: list[dict], budget_rate: float) -> float:
    """Smallest max-softmax threshold whose clean-val false-accept rate (babble + noise-silence rows) is at
    most `budget_rate` (the CTC path's own clean-val rate), i.e. the most permissive one within the budget."""
    non_targets = [r for r in val_rows if not is_target(r)]
    confidences = sorted(
        (top_intent(r)[1] for r in non_targets if top_intent(r)[0] not in EXTRA_INTENT_CLASSES), reverse=True
    )
    allowed = math.floor(budget_rate * len(non_targets) + 1e-9)
    if allowed >= len(confidences):
        return 0.0
    return math.nextafter(confidences[allowed], 1.0)


def topk_coverage(rows: list[dict], k: int) -> int:
    count = 0
    for r in rows:
        if is_target(r) and r["label"] in INTENT_CLASSES:
            order = sorted(range(len(r["probs"])), key=lambda j: -r["probs"][j])[:k]
            count += INTENT_CLASSES.index(r["label"]) in order
    return count


def confusion(rows: list[dict], predict) -> dict[str, dict[str, int]]:
    """For the three confusable pairs: true intent -> {pair member: n, 'other': n, 'reject': n}.
    `predict(row)` returns the accepted intent name, or None for a rejection."""
    out: dict[str, dict[str, int]] = {}
    for pair in CONFUSABLE_PAIRS:
        for true in pair:
            counts = {pair[0]: 0, pair[1]: 0, "other": 0, "reject": 0}
            for r in rows:
                if is_target(r) and r["label"] == true:
                    p = predict(r)
                    counts["reject" if p is None else (p if p in pair else "other")] += 1
            out[true] = counts
    return out


def ctc_predict(threshold: float):
    return lambda r: r["ctc"]["intent"] if ctc_accepted(r, threshold) else None


def cls_predict(threshold: float):
    return lambda r: top_intent(r)[0] if cls_accepted(r, threshold) else None


def agreement(rows: list[dict], ctc_threshold: float) -> dict:
    """Classifier argmax vs the CTC decode, over target rows."""
    targets = [r for r in rows if is_target(r)]
    both = [r for r in targets if ctc_accepted(r, ctc_threshold)]
    ctc_right = [r for r in targets if ctc_correct(r, ctc_threshold, with_slot=False)]
    return {
        "n_targets": len(targets),
        "n_ctc_accepted": len(both),
        "agree_when_ctc_accepted": sum(top_intent(r)[0] == r["ctc"]["intent"] for r in both),
        "n_ctc_correct": len(ctc_right),
        "cls_top1_on_ctc_correct": sum(top_intent(r)[0] == r["label"] for r in ctc_right),
        "cls_top3_on_ctc_correct": topk_coverage(ctc_right, 3) if ctc_right else 0,
    }


# --------------------------------------------------------------------------- report

def read_rows(run_dir: Path, split: str, cond: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(run_dir.glob(f"{split}_{cond}.shard*.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line]
    return sorted(rows, key=lambda r: r["i"])


def gate_a(base: dict, joint: dict, n_babble_scale: float) -> dict:
    """Gate A (00-RECAP), re-expressed on ai231 counts: the joint model's CTC path must not hurt the baseline's.
    exact within 0.5 pp (clean and noisy); noisy babble false accepts <= the old '3/255' scaled to this babble set
    (rounded up), and never worse than the baseline's own."""
    out = {}
    for cond in ("clean", "noisy"):
        b, j = base[cond], joint[cond]
        floor = b["exact_ok"] - math.ceil(0.005 * b["n_targets"])
        out[f"{cond}_exact"] = {"baseline": b["exact_ok"], "joint": j["exact_ok"], "floor": floor, "pass": j["exact_ok"] >= floor}
    allowed = max(math.ceil(3 / 255 * n_babble_scale), base["noisy"]["babble_fa"])
    out["noisy_babble_fa"] = {
        "baseline": base["noisy"]["babble_fa"], "joint": joint["noisy"]["babble_fa"], "allowed": allowed,
        "pass": joint["noisy"]["babble_fa"] <= allowed,
    }
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def by_source(rows: list[dict], ctc_threshold: float, cls_threshold: float | None) -> dict:
    """Target rows grouped by `source_dataset`: n, CTC intent/exact, classifier intent/exact."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if is_target(r):
            groups.setdefault(r["source_dataset"], []).append(r)
    out = {}
    for src, rs in sorted(groups.items()):
        entry = {"n": len(rs)}
        if rs[0]["ctc"] is not None:
            entry["ctc_intent"] = sum(ctc_correct(r, ctc_threshold, with_slot=False) for r in rs)
            entry["ctc_exact"] = sum(ctc_correct(r, ctc_threshold, with_slot=True) for r in rs)
        if cls_threshold is not None:
            entry["cls_intent"] = sum(cls_correct(r, cls_threshold, with_slot=False) for r in rs)
            entry["cls_exact"] = sum(cls_correct(r, cls_threshold, with_slot=True) for r in rs)
        out[src] = entry
    return out


def build_report(
    runs: dict[str, Path],
    ctc_threshold: float,
    budget_run: str | None = None,
    fixed_cls_thresholds: dict[str, float] | None = None,
) -> dict:
    """`runs`: name -> dir holding `{val,test}_{clean,noisy}.shard*.jsonl`. `budget_run`: the run whose CTC
    clean-val false-accept rate sets the classifier threshold budget (run A). With `fixed_cls_thresholds`
    (name -> threshold) nothing is chosen from val at all: a pure transfer test at operating points set elsewhere."""
    data = {name: {(s, c): read_rows(d, s, c) for s in ("val", "test") for c in ("clean", "noisy")} for name, d in runs.items()}
    report: dict = {"ctc_threshold": ctc_threshold, "ctc": {}, "cls": {}, "agreement": {}, "confusion": {}, "topk": {}}
    budget = None
    if fixed_cls_thresholds is None:
        val = data[budget_run][("val", "clean")]
        val_non_target = [r for r in val if not is_target(r)]
        fa = sum(ctc_accepted(r, ctc_threshold) for r in val_non_target)
        budget = rate(fa, len(val_non_target))
        report["budget"] = {"run": budget_run, "val_non_target": len(val_non_target), "ctc_false_accepts": fa, "rate": budget}
    else:
        report["budget"] = None
        report["fixed_cls_thresholds"] = fixed_cls_thresholds
    report["by_source"] = {}
    report["n_slot_unparseable"] = sum(
        bool(r.get("slot_unparseable")) for d in data.values() for r in d[("test", "clean")]
    ) // max(len(data), 1)
    for name, d in data.items():
        test = {c: d[("test", c)] for c in ("clean", "noisy")}
        if test["clean"] and test["clean"][0]["ctc"] is not None:
            report["ctc"][name] = {c: ctc_metrics(test[c], ctc_threshold) for c in test}
        if test["clean"] and test["clean"][0]["probs"] is not None:
            t = fixed_cls_thresholds[name] if fixed_cls_thresholds is not None else choose_cls_threshold(d[("val", "clean")], budget)
            report["cls"][name] = {c: cls_metrics(test[c], t) for c in test}
            report["cls"][name]["threshold"] = t
            report["confusion"][name] = {
                "classifier": {c: confusion(test[c], cls_predict(t)) for c in test},
                **({"ctc": {c: confusion(test[c], ctc_predict(ctc_threshold)) for c in test}} if name in report["ctc"] else {}),
            }
            report["topk"][name] = {
                c: {"n_targets": sum(is_target(r) for r in test[c]), "top1": topk_coverage(test[c], 1), "top3": topk_coverage(test[c], 3)}
                for c in test
            }
            if name in report["ctc"]:
                report["agreement"][name] = {c: agreement(test[c], ctc_threshold) for c in test}
        if test["clean"]:
            ct = report["cls"].get(name, {}).get("threshold")
            report["by_source"][name] = {c: by_source(test[c], ctc_threshold, ct) for c in test}
    return report


def _pct(k: int, n: int) -> str:
    return f"{100 * rate(k, n):.1f}% ({k}/{n})"


def _section(report: dict, c: str) -> list[str]:
    L: list[str] = []
    L.append("| model | path | intent | intent+slot | babble FA | ESC-50 silence FA |\n|---|---|---|---|---|---|")
    for name, m in report["ctc"].items():
        x = m[c]
        L.append(f"| {name} | CTC | {_pct(x['intent_ok'], x['n_targets'])} | {_pct(x['exact_ok'], x['n_targets'])} | "
                 f"{x['babble_fa']}/{x['n_babble']} | {x['silence_fa']}/{x['n_silence']} |")
    for name, m in report["cls"].items():
        x = m[c]
        L.append(f"| {name} | classifier (thr {x['threshold']:.4f}) | {_pct(x['intent_ok'], x['n_targets'])} | {_pct(x['exact_ok'], x['n_targets'])} | "
                 f"{x['babble_fa']}/{x['n_babble']} | {x['silence_fa']}/{x['n_silence']} |")
    L.append("\nClassifier without rejection (plain argmax):\n")
    L.append("| model | intent | intent+slot |\n|---|---|---|")
    for name, m in report["cls"].items():
        x = m[c]
        L.append(f"| {name} | {_pct(x['argmax_intent_ok'], x['n_targets'])} | {_pct(x['argmax_exact_ok'], x['n_targets'])} |")
    L.append("\nTop-k coverage of the true intent, and agreement with the CTC decode:\n")
    L.append("| model | top-1 | top-3 | classifier top-3 on CTC-correct clips | classifier == CTC (when CTC accepted) |\n|---|---|---|---|---|")
    for name, m in report["topk"].items():
        t = m[c]
        a = report["agreement"].get(name, {}).get(c)
        L.append(f"| {name} | {_pct(t['top1'], t['n_targets'])} | {_pct(t['top3'], t['n_targets'])} | "
                 + (f"{_pct(a['cls_top3_on_ctc_correct'], a['n_ctc_correct'])} | {_pct(a['agree_when_ctc_accepted'], a['n_ctc_accepted'])} |" if a else "n/a | n/a |"))
    L.append("\nConfusable pairs (true intent -> where it went):\n")
    for name, m in report["confusion"].items():
        for who in ("classifier", "ctc"):
            if who in m:
                L.append(f"- **{name} / {who}**: " + "; ".join(
                    f"{true}: " + ", ".join(f"{k} {v}" for k, v in cnt.items() if v) for true, cnt in m[who][c].items()))
    return L


def _by_source_table(report: dict, c: str) -> list[str]:
    L = ["| model | path | " + " | ".join(sorted({s for m in report["by_source"].values() for s in m[c]})) + " |"]
    sources = sorted({s for m in report["by_source"].values() for s in m[c]})
    L.append("|---|---|" + "---|" * len(sources))
    for name, m in report["by_source"].items():
        for path, key in (("CTC", "ctc_exact"), ("classifier", "cls_exact")):
            if any(key in v for v in m[c].values()):
                L.append(f"| {name} | {path} | " + " | ".join(
                    _pct(m[c][s][key], m[c][s]["n"]) if s in m[c] and key in m[c][s] else "-" for s in sources) + " |")
    return L


def render_markdown(report: dict, baseline: str, joint: str, gates: dict | None) -> str:
    L: list[str] = []
    b = report["budget"]
    if b is not None:
        L.append(f"CTC path: threshold {report['ctc_threshold']}, whole clip. Classifier: max-softmax threshold chosen on clean val to match the "
                 f"CTC clean-val false-accept rate of `{b['run']}` ({b['ctc_false_accepts']}/{b['val_non_target']} = {100 * b['rate']:.2f}%); "
                 "it is never re-swept on perturbed audio. Test split, `exact` target rows; babble FA is on the 47 babble rows, "
                 "'ESC-50 silence' is the noise-pool rows (an extra probe, not part of the ai231 test).\n")
    else:
        L.append(f"CTC path: threshold {report['ctc_threshold']}, whole clip. Classifier thresholds are FIXED, not chosen here: "
                 f"{report['fixed_cls_thresholds']}. Transfer test, nothing tuned on this data. "
                 f"Rows whose slot truth could not be parsed: {report['n_slot_unparseable']}.\n")
    L.append("## Headline: unperturbed (clean) audio\n")
    L += _section(report, "clean")
    L.append("\n### By source (unperturbed, target rows; intent+slot)\n")
    L += _by_source_table(report, "clean")
    L.append("\n## Robustness: perturbed audio (fixed-seed RIR + ESC-50 noise, seed 0)\n")
    L += _section(report, "noisy")
    L.append("\n### By source (perturbed, target rows; intent+slot)\n")
    L += _by_source_table(report, "noisy")
    if gates:
        L.append(f"\n## Gate A ({joint} CTC path vs {baseline})\n")
        for k, v in gates.items():
            if k != "pass":
                L.append(f"- {k}: baseline {v['baseline']}, joint {v['joint']}, {'floor' if 'floor' in v else 'allowed'} {v.get('floor', v.get('allowed'))} -> {'PASS' if v['pass'] else 'FAIL'}")
        L.append(f"- **Gate A: {'PASS' if gates['pass'] else 'FAIL'}**")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- CLI

def _main_score(a: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    model, _ = load_checkpoint(a.checkpoint, device="cpu", weights_only=True)
    dataset = VCMDataset(a.manifest, split=a.split, augmenter=None)
    ex = LogMelFeatureExtractor()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = "clean" if a.noisy_seed is None else "noisy"
    path = out_dir / f"{a.split}_{cond}.shard{a.shard}of{a.n_shards}.jsonl"
    with path.open("w") as f:
        for n, rec in enumerate(
            score_rows(model, ex, dataset, noisy_seed=a.noisy_seed, margin=a.margin, beam_width=a.beam,
                       shard=(a.shard, a.n_shards), do_ctc=not a.no_ctc, limit=a.limit, rir_pool_size=a.rir_pool_size)
        ):
            f.write(json.dumps(rec) + "\n")
            if n % 200 == 0:
                print(f"{path.name}: {n} rows", flush=True)
    print("wrote", path)


def _main_report(a: argparse.Namespace) -> None:
    runs = {name: Path(p) for name, p in (kv.split("=", 1) for kv in a.run)}
    fixed = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.cls_threshold)} or None
    if fixed is None and a.budget_run is None:
        raise SystemExit("report needs --budget-run, or fixed --cls-threshold values")
    report = build_report(runs, a.ctc_threshold, a.budget_run, fixed)
    gates = None
    if a.baseline in report["ctc"] and a.joint in report["ctc"]:
        gates = gate_a(report["ctc"][a.baseline], report["ctc"][a.joint], report["ctc"][a.baseline]["noisy"]["n_babble"])
        report["gate_a"] = gates
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    md = render_markdown(report, a.baseline, a.joint, gates)
    out.with_suffix(".md").write_text(md)
    print(md)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-manifest")
    b.add_argument("--ai231", type=Path, required=True)
    b.add_argument("--noise", type=Path, default=None, help="optional extra noise manifest; default: dataset-only")
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--split", choices=("val", "test", "holdout"), required=True)
    s.add_argument("--noisy-seed", type=int, default=None, help="omit for the clean pass")
    s.add_argument("--rir-pool-size", type=int, default=None)
    s.add_argument("--margin", type=float, default=4.0)
    s.add_argument("--beam", type=int, default=50)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--n-shards", type=int, default=1)
    s.add_argument("--no-ctc", action="store_true", help="skip the CTC decode (classifier-only checkpoints)")
    s.add_argument("--limit", type=int, default=None, help="smoke runs: only the first N rows")
    r = sub.add_parser("report")
    r.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    r.add_argument("--ctc-threshold", type=float, default=-0.1)
    r.add_argument("--budget-run", default=None, help="needed unless --cls-threshold is given")
    r.add_argument("--cls-threshold", action="append", default=[], metavar="NAME=VALUE",
                   help="fixed classifier threshold (repeatable); skips choosing from val (transfer test)")
    r.add_argument("--baseline", required=True)
    r.add_argument("--joint", required=True)
    r.add_argument("--out", required=True, help="path without extension; writes .json and .md")
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_arg_parser().parse_args(argv)
    if a.cmd == "build-manifest":
        print(build_eval_manifest(a.ai231, a.noise, a.out))
    elif a.cmd == "score":
        _main_score(a)
    else:
        _main_report(a)


if __name__ == "__main__":
    main()
