"""Apply the pre-registered rules of docs/AI231-FIL50.md to scored outputs and print the report tables.

    uv run python scripts/ai231_fil50_rules.py --data-root <checkout with out/conversions> --root out/vcm/ai231fil50-compare --new P-plain H-heads \
        --refs old-production v2s1-baseline v2s1-heads-A [--holdout] --out out/vcm/ai231fil50-compare/report.md

`--root/<name>/<tag>/{val,test,holdout}_{clean,noisy}.shard*.jsonl` come from the scoring loop in the doc. Margin 4.0 and
threshold -0.1 unless the tag says `nogate`. Nothing here tunes anything on test: the classifier threshold is fit on clean
val only.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from me2_voicegen.vcm.semantic_eval import (
    choose_cls_threshold, cls_accepted, cls_correct, ctc_accepted, ctc_correct, is_babble, is_target, read_rows,
)

CTC_T = -0.1
SILENCE_SOURCES = ("background_noise", "negative_near_silence")
FLAGGED = ("Reminder Study", "Reminder Exercise", "Pause song")
MANIFESTS: dict[str, Path] = {}


def set_data_root(root: Path) -> None:
    c = root / "out/conversions/v2"
    MANIFESTS.update({"orig": c / "ai231-v2/manifest.eval-exact.csv", "persona": c / "ai231-fil50-eval/manifest.persona.csv",
                      "old": c / "optionb-v3-vcmx-fil50/manifest.csv", "holdout": c / "ai231-v2/manifest.csv"})


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}% ({k}/{n})" if n else "n/a"


def acc(rows, path="ctc", thr=None):
    t = [r for r in rows if is_target(r)]
    k = sum(ctc_correct(r, CTC_T, with_slot=True) if path == "ctc" else cls_correct(r, thr, with_slot=True) for r in t)
    return k, len(t)


def manifest_rows(key: str, split: str) -> list[dict]:
    return [r for r in csv.DictReader(open(MANIFESTS[key])) if r["split"] == split]


EXCLUDE: set[tuple[str, str]] = set()  # (ai231 split, path) of clips also present in the internal training data


class Run:
    def __init__(self, root: Path, name: str):
        self.name, self.dir = name, root / name

    def rows(self, tag, split, cond="clean"):
        d = self.dir / tag
        rows = read_rows(d, split, cond) if d.exists() and any(d.glob(f"{split}_{cond}.shard*.jsonl")) else None
        if rows is not None and EXCLUDE and (tag.startswith("orig_") or tag in ("val_clean", "holdout")):
            man = manifest_rows("orig" if tag != "holdout" else "holdout", split)
            rows = [r for r in rows if (split, man[r["i"]]["path"]) not in EXCLUDE]
        return rows


def rule_metrics(run: Run) -> dict:
    m = {}
    o = run.rows("orig_clean", "test")
    if o is None:
        return m
    orig = [r for r in o if r["source_dataset"] == "optionb"]
    m["A"] = acc(orig)
    ng = run.rows("orig_nogate", "test")
    m["A_nogate"] = acc([r for r in ng if r["source_dataset"] == "optionb"]) if ng else None
    nz = run.rows("orig_noisy_ds", "test", "noisy")
    m["B"] = acc([r for r in nz if r["source_dataset"] == "optionb"]) if nz else None
    ol = run.rows("old_clean", "test")
    if ol:
        t = [r for r in ol if is_target(r)]
        m["C"] = acc([r for r in t if r["source_dataset"] != "fil50_persona"])
        m["C_all"], m["C_persona"] = acc(t), acc([r for r in t if r["source_dataset"] == "fil50_persona"])
        m["C_real"] = acc([r for r in t if r["source_dataset"] == "vcm_balanced"])
    babble = [r for r in o if is_babble(r) and r["source_dataset"] == "optionb"]
    sil = [r for r in o if r["source_dataset"] in SILENCE_SOURCES]
    m["D_babble"] = (sum(ctc_accepted(r, CTC_T) for r in babble), len(babble))
    m["D_silence"] = (sum(ctc_accepted(r, CTC_T) for r in sil), len(sil))
    kinds = defaultdict(lambda: [0, 0])
    for r in o:
        if r["source_dataset"].startswith("negative_"):
            kinds[r["source_dataset"]][1] += 1
            kinds[r["source_dataset"]][0] += ctc_accepted(r, CTC_T)
    m["negatives"] = dict(kinds)
    pr = run.rows("persona_clean", "test")
    m["persona"] = acc([r for r in pr if r["source_dataset"] == "fil50_persona"]) if pr else None
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."), help="checkout whose out/conversions/v2 holds the manifests")
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--exclude-overlap", type=Path, default=None, help="overlap_with_ai231.csv: drop those ai231 rows (orig_*, val, holdout)")
    ap.add_argument("--holdout", action="store_true", help="include the holdout table (score it once, at the end)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    set_data_root(a.data_root)
    if a.exclude_overlap:
        EXCLUDE.update((r["ai231_split"], r["ai231_path"]) for r in csv.DictReader(a.exclude_overlap.open()))
    names = a.new + a.refs
    runs = {n: Run(a.root, n) for n in names}
    M = {n: rule_metrics(r) for n, r in runs.items()}
    L = ["# ai231-fil50 results (generated by scripts/ai231_fil50_rules.py)\n"]
    if EXCLUDE:
        L.append(f"**Subset: {len(EXCLUDE)} ai231 test/holdout rows that also occur in the internal training data are excluded (orig_*, val, holdout tables).**\n")
    cell = lambda x: pct(*x) if x else "n/a"

    L.append("## Pre-registered rules (CTC path, threshold -0.1, margin 4.0, beam 50)\n")
    L.append("| model | A clean >= 96.5 | B perturbed >= 90 | C old test, non-persona >= 93 | D babble FA (0/76), silence FA <= old prod | verdict |\n|---|---|---|---|---|---|")
    ref_sil = M.get("old-production", {}).get("D_silence", (0, 0))[0]
    for n in names:
        m = M[n]
        if not m:
            continue
        ok = {"A": 100 * m["A"][0] / m["A"][1] >= 96.5,
              "B": bool(m["B"]) and 100 * m["B"][0] / m["B"][1] >= 90,
              "C": bool(m.get("C")) and 100 * m["C"][0] / m["C"][1] >= 93,
              "D": m["D_babble"][0] == 0 and m["D_silence"][0] <= ref_sil}
        has_c = bool(m.get("C"))  # a model without the old-internal-test pass (e.g. trained on that data) gets no verdict
        mark = lambda k: "PASS" if ok[k] else ("FAIL" if k != "C" or has_c else "n/a")
        L.append(f"| {n} | {cell(m['A'])} {mark('A')} | {cell(m['B'])} {mark('B')} | {cell(m.get('C'))} {mark('C')} | "
                 f"{m['D_babble'][0]}/{m['D_babble'][1]}, {m['D_silence'][0]}/{m['D_silence'][1]} {mark('D')} | {('PASS' if all(ok.values()) else 'FAIL') if has_c else 'n/a (C not scored)'} |")
    L.append("\nVerdict applies to the new models only; references are shown for scale. One seed.\n")

    L.append("## Other numbers, CTC path\n")
    L.append("| model | A, margin gate off | persona-only test (2,946) | old test all rows | old test persona (voice-leaked) | old test real recordings (129) |\n|---|---|---|---|---|---|")
    for n in names:
        m = M[n]
        if m:
            L.append(f"| {n} | {cell(m['A_nogate'])} | {cell(m['persona'])} | {cell(m.get('C_all'))} | {cell(m.get('C_persona'))} | {cell(m.get('C_real'))} |")
    L.append("\n## Rejection by synthetic-negative kind (CTC false accepts, clean, 50 clips each)\n")
    L.append("| model | " + " | ".join(("negative_babble", "negative_reversed", "negative_truncated")) + " |\n|---|---|---|---|")
    for n in names:
        neg = M[n].get("negatives") if M[n] else None
        if neg:
            L.append(f"| {n} | " + " | ".join(f"{neg.get(k, [0, 0])[0]}/{neg.get(k, [0, 0])[1]}" for k in ("negative_babble", "negative_reversed", "negative_truncated")) + " |")

    # classifier of the heads runs (threshold fit on clean val, budget = the run's own CTC clean-val false-accept rate)
    L.append("\n## Classifier path (heads runs; threshold fit on clean val)\n")
    L.append("| model | threshold | A clean | B perturbed | persona-only | old test non-persona | babble FA (76) | silence FA (100) | A clean, plain argmax (no rejection) | babble FA at argmax |\n|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        r = runs[n]
        val, o = r.rows("val_clean", "val"), r.rows("orig_clean", "test")
        if not val or not o or o[0].get("probs") is None:
            continue
        nt = [x for x in val if not is_target(x)]
        budget = sum(ctc_accepted(x, CTC_T) for x in nt) / len(nt)
        thr = choose_cls_threshold(val, budget)
        g = lambda tag, split, cond, f=lambda x: True: [x for x in (r.rows(tag, split, cond) or []) if f(x)]
        row = [f"{thr:.4f} (val CTC FA {100 * budget:.1f}%)",
               cell(acc(g("orig_clean", "test", "clean", lambda x: x["source_dataset"] == "optionb"), "cls", thr)),
               cell(acc(g("orig_noisy_ds", "test", "noisy", lambda x: x["source_dataset"] == "optionb"), "cls", thr)),
               cell(acc(g("persona_clean", "test", "clean", lambda x: x["source_dataset"] == "fil50_persona"), "cls", thr)),
               cell(acc(g("old_clean", "test", "clean", lambda x: x["source_dataset"] != "fil50_persona"), "cls", thr))]
        babble = [x for x in o if is_babble(x) and x["source_dataset"] == "optionb"]
        sil = [x for x in o if x["source_dataset"] in SILENCE_SOURCES]
        row += [f"{sum(cls_accepted(x, thr) for x in babble)}/{len(babble)}", f"{sum(cls_accepted(x, thr) for x in sil)}/{len(sil)}"]
        origt = [x for x in o if x["source_dataset"] == "optionb"]
        row += [cell(acc(origt, "cls", None)), f"{sum(cls_accepted(x, 0.0) for x in babble)}/{len(babble)}"]
        L.append(f"| {n} | " + " | ".join(row) + " |")

    # per-variation and persona-vs-real, CTC path on the clean test
    L.append("\n## Per-variation accuracy (CTC, clean test; flagged thin variations marked *)\n")
    man = manifest_rows("orig", "test")
    for n in a.new:
        o = runs[n].rows("orig_clean", "test")
        if not o:
            continue
        per = defaultdict(lambda: [0, 0])
        for r in o:
            mr = man[r["i"]]  # row index into the test rows of the eval manifest (rows may be filtered by --exclude-overlap)
            if is_target(r) and r["source_dataset"] == "optionb":
                per[mr["variation"]][1] += 1
                per[mr["variation"]][0] += ctc_correct(r, CTC_T, with_slot=True)
        worst = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:8]
        L.append(f"**{n}** worst 8 of {len(per)} variations: " + "; ".join(f"{v}{'*' if v in FLAGGED else ''} {pct(*c)}" for v, c in worst))
        L.append("flagged: " + "; ".join(f"{v}* {pct(*per[v])}" for v in FLAGGED if v in per) + "\n")
    pm = manifest_rows("persona", "test")
    L.append("## Persona vs real (CTC, clean test)\n")
    for n in a.new:
        pr, o = runs[n].rows("persona_clean", "test"), runs[n].rows("orig_clean", "test")
        if pr and o:
            ps = acc([r for r in pr if r["source_dataset"] == "fil50_persona"]); rs = acc([r for r in o if r["source_dataset"] == "optionb"])
            gap = 100 * ps[0] / ps[1] - 100 * rs[0] / rs[1]
            L.append(f"- {n}: persona {pct(*ps)} vs original ai231 {pct(*rs)}; gap {gap:+.1f} points" + ("  **>10 points: stop and report**" if gap > 10 else ""))

    if a.holdout:
        L.append("\n## Holdout (186 real commands + 16 out of scope; scored once, not tuned on)\n")
        L.append("| model | CTC intent+slot | classifier at the val-fit threshold |\n|---|---|---|")
        for n in names:
            h = runs[n].rows("holdout", "holdout")
            if not h:
                continue
            row = cell(acc(h))
            val = runs[n].rows("val_clean", "val")
            ccell = "n/a"
            if val and h[0].get("probs") is not None:
                nt = [x for x in val if not is_target(x)]
                thr = choose_cls_threshold(val, sum(ctc_accepted(x, CTC_T) for x in nt) / len(nt))
                ccell = cell(acc(h, "cls", thr))
            L.append(f"| {n} | {row} | {ccell} |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
