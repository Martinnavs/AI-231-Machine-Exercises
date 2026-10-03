"""Headline metrics of the hybrid in the format of the sister repo's README (airimonda/vcm-me2): variation balanced accuracy, command and slot
accuracy, human vs synthetic voices, out-of-scope false accept, in-scope false reject, synthetic-negative misfire, per split.

Reads the per-clip rows written by `scripts/hybrid_score.py score` (hybrid, INT8 ONNX) and by `vcm.semantic_eval score` (the wide CTC alone, whole clip, margin 4.0).

    uv run python scripts/hybrid_metrics.py --out results.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from me2_voicegen.vcm.semantic_eval import ctc_accepted, read_rows

OUT = Path("/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out")
V2 = OUT / "conversions/v2/ai231-v2"
SPLITS = {  # name -> (manifest, split, hybrid rows dir, wide-CTC cached rows dir)
    "val": (V2 / "manifest.eval-exact.csv", "val", OUT / "vcm/hybrid-eval/onnx_int8/val_clean", OUT / "vcm/ai231fil50-compare/H-wide/val_clean", "val"),
    "test": (V2 / "manifest.eval-exact.csv", "test", OUT / "vcm/hybrid-eval/onnx_int8/orig_clean", OUT / "vcm/ai231fil50-compare/H-wide/orig_clean", "test"),
    "holdout": (V2 / "manifest.csv", "holdout", OUT / "vcm/hybrid-eval/onnx_int8/holdout", OUT / "vcm/ai231fil50-compare/H-wide/holdout", "holdout"),
}


def is_synthetic_voice(row: dict) -> bool:
    return "group_synthetic" in row["source_relpath"].split("/")[-1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_hybrid(d: Path, split: str) -> dict[int, tuple[str | None, list[str]]]:
    rows = {}
    for f in sorted(d.glob(f"{split}_clean.shard*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            dec = r["fallback"]
            rows[r["i"]] = (dec["intent"], [str(v).lower() for v in dec["slots"].values()])
    return rows


def load_ctc(d: Path, split: str) -> dict[int, tuple[str | None, list[str]]]:
    out = {}
    for r in read_rows(d, split, "clean"):
        c = r["ctc"]
        out[r["i"]] = (c["intent"], [str(v).lower() for v in c["slots"].values()]) if ctc_accepted(r, -0.1) else (None, [])
    return out


def metrics(manifest_rows: list[dict], pred: dict[int, tuple[str | None, list[str]]]) -> dict:
    inscope, oos, synneg = [], [], []
    for i, r in enumerate(manifest_rows):
        if i not in pred:
            continue
        if r["bucket"] == "target_commands":
            inscope.append((r, pred[i]))
        elif r["bucket"] == "babble" and r["source_dataset"] == "optionb":
            oos.append((r, pred[i]))
        else:
            synneg.append((r, pred[i]))
    ok = lambda r, p: p[0] == r["label"] and (not r["slot_value"] or r["slot_value"].lower() in p[1])  # noqa: E731
    by_var = defaultdict(list)
    for r, p in inscope:
        by_var[r["variation"]].append(ok(r, p))
    recalls = [sum(v) / len(v) for v in by_var.values()]
    oos_rej = [p[0] is None for _, p in oos]
    if oos_rej:
        recalls.append(sum(oos_rej) / len(oos_rej))
    cmd_ok = [p[0] == r["label"] for r, p in inscope]
    slotted = [(r, p) for r, p in inscope if r["slot_value"] and p[0] == r["label"]]
    human = [(r, p) for r, p in inscope if not is_synthetic_voice(r)]
    synth = [(r, p) for r, p in inscope if is_synthetic_voice(r)]
    acc = lambda xs: (sum(ok(r, p) for r, p in xs), len(xs))  # noqa: E731
    return {
        "n_inscope": len(inscope), "n_oos": len(oos), "n_synneg": len(synneg), "n_variations": len(by_var),
        "var_bal_acc": sum(recalls) / len(recalls), "accuracy": acc(inscope), "command_acc": (sum(cmd_ok), len(cmd_ok)),
        "slot_acc": (sum(r["slot_value"].lower() in p[1] for r, p in slotted), len(slotted)),
        "human": acc(human), "synthetic": acc(synth), "oos_fa": (sum(not x for x in oos_rej), len(oos_rej)),
        "false_reject": (sum(p[0] is None for _, p in inscope), len(inscope)),
        "synneg_misfire": (sum(p[0] is not None for _, p in synneg), len(synneg)),
    }


def pct(kn: tuple[int, int], ci: bool = False) -> str:
    k, n = kn
    if n == 0:
        return "n/a"
    s = f"{100 * k / n:.1f}% ({k:,}/{n:,})"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.1f}-{100 * hi:.1f}]"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for name, (man, split, hyb, ctc, ctc_split) in SPLITS.items():
        rows = [r for r in csv.DictReader(man.open(newline="", encoding="utf-8")) if r["split"] == split]
        res[name] = {"hybrid": metrics(rows, load_hybrid(hyb, split)), "ctc": metrics(rows, load_ctc(ctc, ctc_split))}
    L = ["## Headline (test split, exact-variation rows)", "", "| Metric | Hybrid (INT8 ONNX) | Wide CTC alone (fp32) |", "|---|---:|---:|"]
    t = res["test"]
    f = lambda k, fmt: (fmt(t["hybrid"][k]), fmt(t["ctc"][k]))  # noqa: E731
    rows = [
        (f"Variation balanced accuracy ({t['hybrid']['n_variations']} variations + OOS)", f("var_bal_acc", lambda v: f"{v:.4f}")),
        ("... on human voices only", f("human", lambda v: pct(v))),
        ("... on synthetic voices only", f("synthetic", lambda v: pct(v))),
        ("Command + slot accuracy (in scope)", f("accuracy", lambda v: pct(v))),
        ("Command accuracy", f("command_acc", lambda v: pct(v))),
        ("Slot accuracy (command right, slotted)", f("slot_acc", lambda v: pct(v))),
        ("Out-of-scope false accept", f("oos_fa", lambda v: pct(v))),
        ("In-scope false reject", f("false_reject", lambda v: pct(v))),
        ("Synthetic-negative misfire", f("synneg_misfire", lambda v: pct(v))),
    ]
    L += [f"| {m} | {h} | {c} |" for m, (h, c) in rows]
    L += ["", "## By split (hybrid, INT8 ONNX; accuracy = command and slot both right)", "",
          "| Split | In-scope clips | Accuracy | Command acc. | Human voices | Synthetic voices | OOS accepted | In scope rejected | Synthetic negatives accepted |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("val", "test", "holdout"):
        m = res[name]["hybrid"]
        L.append(f"| {name} | {m['n_inscope']:,} | {pct(m['accuracy'])} | {pct(m['command_acc'])} | {pct(m['human'])} | {pct(m['synthetic'])} | {pct(m['oos_fa'])} | {pct(m['false_reject'])} | {pct(m['synneg_misfire'])} |")
    L += ["", "## By split (wide CTC alone, fp32)", "", "| Split | Accuracy | Human voices | Synthetic voices | OOS accepted | In scope rejected |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("val", "test", "holdout"):
        m = res[name]["ctc"]
        L.append(f"| {name} | {pct(m['accuracy'])} | {pct(m['human'])} | {pct(m['synthetic'])} | {pct(m['oos_fa'])} | {pct(m['false_reject'])} |")
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
