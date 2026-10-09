#!/usr/bin/env python3
"""Cut a tiny, fast manifest from a real one for pipeline smoke runs.

Keeps a few rows per (split, bucket) and a handful of `background_noise` rows
per split (the training augmenter's noise pool is built from them). `path` is rewritten
absolute so the smoke manifest can live anywhere (e.g. read-only inputs on
Kaggle) -- `VCMDataset` joins `audio_root / path`, and an absolute path wins.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-bucket", type=int, default=24, help="rows kept per (split, bucket)")
    ap.add_argument("--noise-rows", type=int, default=12, help="background_noise rows kept per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    groups: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["bucket"], r["source_dataset"] == "background_noise")].append(r)
    rng = random.Random(args.seed)
    keep: list[dict] = []
    for (_split, _bucket, is_noise), g in sorted(groups.items()):
        keep.extend(rng.sample(g, min(args.noise_rows if is_noise else args.per_bucket, len(g))))

    base = args.manifest.parent
    for r in keep:
        r["path"] = os.path.abspath(os.path.normpath(base / r["path"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    print(f"smoke manifest: {len(keep)}/{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
