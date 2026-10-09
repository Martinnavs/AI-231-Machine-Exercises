"""Convert the `airimonda/ai231-me2-voice-commands` parquet shards into our VCM manifest layout.

    uv run python -m me2_voicegen.vcm.optionb.import_ai231 \
        --src raw_datasets/ai231-me2-voice-commands --out out/conversions/v2/ai231-me2-voice-commands

Writes `<out>/audio/<split>/*.wav` and `<out>/manifest.csv` (the usual VCM columns plus `variation`,
`slot_value`, `variation_match`). Splits: train, val (10% of train speakers, by hash), test, holdout.
The numerals set is skipped, and so are command clips whose slot value is outside the schema.
Out-of-scope clips become `unknown` (speech) or `silence` (empty transcript) rows.

`synthetic_negatives/` (generated out-of-scope clips, train + test) is imported too, with empty transcripts:
`noise_only` -> `silence` rows with `source_dataset=background_noise` (the additive-noise pool, also the noisy-eval
pool), `babble` -> `unknown` rows (`negative_babble`, the babble-overlay pool), and `near_silence`, `reversed`,
`truncated` -> silence / unknown rows (`negative_*`) kept out of both pools. The file's `neg_kind` is the row's
`source_relpath` suffix. Train negatives are split into val by filename hash (one in ten).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

FIELDS = ["filename", "path", "bucket", "label", "duration", "sample_rate", "resampled", "source_dataset",
          "source_relpath", "group_id", "split", "transcript", "variation", "slot_value", "variation_match"]
VAL_FRACTION = 10  # one train speaker in 10 goes to val


def is_val_speaker(speaker: str) -> bool:
    return int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % VAL_FRACTION == 0


# neg_kind -> (bucket, label, source_dataset). Only `background_noise` and `negative_babble` feed augmentation pools.
NEGATIVE_KINDS = {
    "noise_only": ("silence", "silence", "background_noise"),
    "near_silence": ("silence", "silence", "negative_near_silence"),
    "babble": ("babble", "unknown", "negative_babble"),
    "reversed": ("babble", "unknown", "negative_reversed"),
    "truncated": ("babble", "unknown", "negative_truncated"),
}


def convert_negative(r: dict) -> dict:
    """Manifest row (without path/filename) for one `synthetic_negatives` row; the transcript tag is dropped."""
    bucket, label, source = NEGATIVE_KINDS[r["neg_kind"]]
    return {"bucket": bucket, "label": label, "transcript": "", "variation": "", "slot_value": "",
            "variation_match": "", "source_dataset": source}


def convert_row(r: dict, split: str) -> dict | None:
    """Manifest row (without path/filename) for one parquet row, or None if it is skipped."""
    if r["command"] == "OUT_OF_SCOPE":
        text = re.sub(r"\d+", " ", r["transcript"] or "").strip()  # grammar-free speech; keep digits out of CTC targets
        bucket, label = ("babble", "unknown") if text else ("silence", "silence")
        return {"bucket": bucket, "label": label, "transcript": text, "variation": "", "slot_value": "",
                "variation_match": ""}
    if not r["variation"]:  # slot value outside the schema
        return None
    slot = re.sub(r":00\b", "", r["slot_value"] or "")
    return {"bucket": "target_commands", "label": r["command"], "transcript": re.sub(r":00\b", "", r["transcript"]),
            "variation": r["variation"], "slot_value": slot, "variation_match": r["variation_match"] or ""}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rows, skipped = [], Counter()
    shards = [(sp, f"data/{sp}-*.parquet", False) for sp in ("train", "test", "holdout")]
    shards += [(sp, f"synthetic_negatives/{sp}-*.parquet", True) for sp in ("train", "test")]
    for split, pattern, negatives in shards:
        for shard in sorted(a.src.glob(pattern)):
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
                for r in batch.to_pylist():
                    out = convert_negative(r) if negatives else convert_row(r, split)
                    if out is None:
                        skipped[split] += 1
                        continue
                    key = r["file"] if negatives else r["speaker_id"]
                    s = "val" if split == "train" and is_val_speaker(key) else split
                    name = f"{len(rows):06d}_{out['label']}.wav"
                    dest = a.out / "audio" / s / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(r["audio"]["bytes"])
                    rows.append({"filename": name, "path": f"audio/{s}/{name}", "duration": f"{r['duration_s']:.6f}",
                                 "sample_rate": 16000, "resampled": False,
                                 "source_relpath": f"{'negatives/' if negatives else ''}{split}/{r['file']}",
                                 "group_id": r["speaker_id"], "split": s, **{"source_dataset": "optionb", **out}})
    with (a.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(Counter((r["split"], r["bucket"]) for r in rows), "skipped:", dict(skipped))


if __name__ == "__main__":
    main()
