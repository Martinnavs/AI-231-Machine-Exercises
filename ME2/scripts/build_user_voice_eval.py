"""Evaluation manifests from the user-voice recordings (feature user-voice-weak-phrases); nothing here is trained on.

    uv run python scripts/build_user_voice_eval.py --root <checkout holding out/conversions/v2> --ai231 <ai231-v2 dir> --out <dir>

Writes `<out>/raw/manifest.csv` (the 20 raw recordings of the user's own voice + the ai231 `noise_only` test clips as the noise
pool for a perturbed pass) and `<out>/converted/manifest.csv` (the QA-passed persona renderings of those recordings).
All rows are `split=test`; paths are relative to the manifest folder. The audio is the user's own voice: keep it local.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

FIELDS = ["filename", "path", "bucket", "label", "duration", "sample_rate", "resampled", "source_dataset", "source_relpath",
          "group_id", "split", "transcript", "variation", "slot_value", "variation_match"]


def rel(p: Path, base: Path) -> str:
    return os.path.relpath(os.path.realpath(p), os.path.realpath(base))


def row(clip_id, path, label, text, dur, group, slot, out):
    return {"filename": f"{clip_id}.wav", "path": rel(path, out), "bucket": "target_commands", "label": label, "duration": dur,
            "sample_rate": 16000, "resampled": False, "source_dataset": "optionb", "source_relpath": str(path), "group_id": group,
            "split": "test", "transcript": text, "variation": "", "slot_value": slot, "variation_match": "exact"}


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ai231", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    uv = a.root / "out/conversions/v2/user_voice"
    clips = {c["clip_id"]: c for c in csv.DictReader((uv / "clips.csv").open(newline="", encoding="utf-8"))}
    slot_of = lambda c: next(iter(json.loads(c["slots"]).values()), "") if c["slots"] else ""
    raw = [row(i, a.root / c["path"], c["label"], c["text"], c["duration"], "user", slot_of(c), a.out / "raw") for i, c in clips.items()]
    ai = list(csv.DictReader((a.ai231 / "manifest.csv").open(newline="", encoding="utf-8")))
    noise = [{"filename": r["filename"], "path": rel(a.ai231 / r["path"], a.out / "raw"), "bucket": "silence", "label": "silence",
              "source_dataset": "background_noise", "split": "test", "sample_rate": 16000, "duration": r["duration"]}
             for r in ai if r["split"] == "test" and r["source_dataset"] == "background_noise"]
    write(a.out / "raw" / "manifest.csv", raw + noise)
    passed = {r["job_id"] for r in csv.DictReader((uv / "qa/qa_pass.csv").open(newline="", encoding="utf-8")) if r["passed"] == "True"}
    conv = []
    for g in csv.DictReader((uv / "gen_manifest.csv").open(newline="", encoding="utf-8")):
        if g["status"] == "ok" and g["job_id"] in passed:
            c = clips[g["source_row_ref"]]
            conv.append(row(g["job_id"], a.root / g["path"], g["label"], g["text"], g["duration"], g["voice_id"], slot_of(c), a.out / "converted"))
    write(a.out / "converted" / "manifest.csv", conv + noise)
    print(f"raw {len(raw)} recordings (+{len(noise)} noise-pool rows); converted {len(conv)} QA-passed clips from {len({r['group_id'] for r in conv})} voices")


if __name__ == "__main__":
    main()
