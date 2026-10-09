"""Scaffold a held-out evaluation set from the internal test split that no training source has seen.

    uv run python scripts/build_internal_heldout.py --internal <dir> --ai231 <ai231-v2 dir> --out <new dir> [--stats-only]

A clip is eligible when ALL of these hold (checked on the unfiltered internal manifest, so it is safe for models trained on
either the full or the ai231-filtered internal data, and for every ai231-trained model):
  1. internal split == test, source is optionb / vcm_balanced (commands) or filipino_speech_corpus (speech, rejection probe);
     persona (all 17 voices are in every training set), ambient copies and ESC-50 noise rows are never used;
  2. audio not byte-identical to any ai231 clip (any split) or any internal train/val clip;
  3. its speaker (group_id) is absent from the internal train/val splits and from the ai231 train and val splits.
     (A speaker that appears in ai231 only as a test/holdout speaker is accepted: no model trains on it.)
Near-duplicates (re-encoded, trimmed, noisy/clean twins of another voice's clip) are not detectable here; condition 3 covers
the twin case for synthetic voices because a twin has the same voice. Noise for the perturbed gate comes from the ai231
`noise_only` test clips, appended as `background_noise` rows. Paths in the output are relative to --out.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMAND_SOURCES = ("optionb", "vcm_balanced")
PROBE_SOURCES = ("filipino_speech_corpus",)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def hashes(root: Path, rows: list[dict]) -> dict[str, str]:
    with ThreadPoolExecutor(16) as ex:
        return dict(zip((r["path"] for r in rows), ex.map(lambda r: md5(root / r["path"]), rows)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", type=Path, required=True)
    ap.add_argument("--ai231", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    internal = list(csv.DictReader((a.internal / "manifest.csv").open(newline="", encoding="utf-8")))
    ai = list(csv.DictReader((a.ai231 / "manifest.csv").open(newline="", encoding="utf-8")))
    base = [r for r in internal if not r["source_dataset"].endswith("_ambient")]
    cand = [r for r in base if r["split"] == "test" and r["source_dataset"] in COMMAND_SOURCES + PROBE_SOURCES]
    seen_int = [r for r in base if r["split"] in ("train", "val") and r["source_dataset"] not in ("fil50_persona", "background_noise")]
    ai_hash = set(hashes(a.ai231, ai).values())
    int_hash = set(hashes(a.internal, seen_int).values())
    ch = hashes(a.internal, cand)
    int_spk = {r["group_id"] for r in base if r["split"] in ("train", "val") and r["group_id"]}
    ai_trainval_spk = {r["group_id"] for r in ai if r["split"] in ("train", "val") and r["group_id"]}
    ai_spk_splits = defaultdict(set)
    for r in ai:
        ai_spk_splits[r["group_id"]].add(r["split"])
    why, keep = Counter(), []
    for r in cand:
        h, g = ch[r["path"]], r["group_id"]
        if h in ai_hash:
            why[(r["source_dataset"], "identical to an ai231 clip")] += 1
        elif h in int_hash:
            why[(r["source_dataset"], "identical to an internal train/val clip")] += 1
        elif g and g in int_spk:
            why[(r["source_dataset"], "speaker in internal train/val")] += 1
        elif g and g in ai_trainval_spk:
            why[(r["source_dataset"], "speaker in ai231 train/val")] += 1
        else:
            keep.append(r)
            why[(r["source_dataset"], "KEPT")] += 1
    print(f"candidates {len(cand)} -> kept {len(keep)}")
    for k, v in sorted(why.items()):
        print(f"  {k[0]:24s} {k[1]:42s} {v}")
    cmd = [r for r in keep if r["source_dataset"] in COMMAND_SOURCES and r["bucket"] == "target_commands"]
    print("kept command clips:", len(cmd), "| speakers:", len({r["group_id"] for r in cmd}), "| by source:", dict(Counter(r["source_dataset"] for r in cmd)))
    print("kept speakers' ai231 splits:", dict(Counter("+".join(sorted(ai_spk_splits.get(g, {"(not in ai231)"}))) for g in {r["group_id"] for r in keep})))
    print("intents covered:", len({r["label"] for r in cmd}), "of 19; per intent min/median/max:",
          sorted(Counter(r["label"] for r in cmd).values())[:1], sorted(Counter(r["label"] for r in cmd).values())[len(Counter(r["label"] for r in cmd)) // 2:][:1], sorted(Counter(r["label"] for r in cmd).values())[-1:])
    if a.stats_only:
        return
    noise = [r for r in ai if r["split"] == "test" and r["source_dataset"] == "background_noise"] or [
        r for r in ai if r["split"] == "test" and r.get("label") == "silence" and r["filename"].startswith("")][:0]
    out_rows = []
    for r in keep:
        out_rows.append({**r, "split": "test", "path": os.path.relpath(os.path.realpath(a.internal / r["path"]), os.path.realpath(a.out))})
    for r in noise:
        out_rows.append({k: r.get(k, "") for k in internal[0].keys()} | {"split": "test", "source_dataset": "background_noise",
                         "path": os.path.relpath(os.path.realpath(a.ai231 / r["path"]), os.path.realpath(a.out))})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(internal[0].keys()), restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows ({len(noise)} noise-pool rows from ai231 test) to {a.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
