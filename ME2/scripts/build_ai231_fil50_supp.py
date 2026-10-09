"""ai231-fil50 manifest plus the ai231 `supplemental_synth` clips of train voices, written to a new folder.

    uv run python scripts/build_ai231_fil50_supp.py --base out/conversions/v2/ai231-fil50/manifest.csv \
        --supplemental raw_datasets/ai231-me2-voice-commands-v2/supplemental_synth --out out/conversions/v2/ai231-fil50-supp

The base rows are copied with their paths made relative to the new folder (the base folder is not touched). Only
supplemental rows whose `voice_split` is `train` are used (the others belong to test or holdout voices); they become
train rows, or val rows for the train speakers that `import_ai231.is_val_speaker` sends to val, exactly as in the base.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from me2_voicegen.vcm.optionb.import_ai231 import convert_row, is_val_speaker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--supplemental", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with a.base.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for r in rows:
        r["path"] = os.path.relpath(os.path.realpath(a.base.parent / r["path"]), os.path.realpath(a.out))
    added, skipped = Counter(), Counter()
    for shard in sorted(a.supplemental.glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
            for r in batch.to_pylist():
                if r["voice_split"] != "train":
                    skipped[f"voice_split={r['voice_split']}"] += 1
                    continue
                out = convert_row(r, "train")
                if out is None:
                    skipped["off-schema"] += 1
                    continue
                split = "val" if is_val_speaker(r["speaker_id"]) else "train"
                name = f"supp_{sum(added.values()):05d}_{out['label']}.wav"
                dest = a.out / "audio" / split / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r["audio"]["bytes"])
                rows.append({"filename": name, "path": f"audio/{split}/{name}", "duration": f"{r['duration_s']:.6f}",
                             "sample_rate": 16000, "resampled": False, "source_dataset": "optionb",
                             "source_relpath": f"supplemental/{r['file']}", "group_id": r["speaker_id"], "split": split, **out})
                added[split] += 1
    with (a.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"base rows {len(rows) - sum(added.values())}; supplemental added {dict(added)}; skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
