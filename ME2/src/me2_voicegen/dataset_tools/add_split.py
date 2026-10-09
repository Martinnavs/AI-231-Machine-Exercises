"""Add a train/val/test split column to test_set/manifest.csv.

Assigns each row a `split` in {train, val, test} at a 70/20/10 ratio,
stratified by `bucket` (target_commands/babble/silence) so each split
preserves the overall 65/25/10 target/babble/silence composition. Within
each bucket, assignment is a seeded random shuffle — no group-awareness
(a given group_id, e.g. one speaker or video, can land in more than one
split). Per-bucket split counts are computed by largest-remainder
allocation (ties broken train > val > test), the same allocation style
already used by build_test_set.py, so counts are exact rather than
approximate.

Rerunning overwrites the `split` column in place with a fresh assignment
for the given --seed; it does not reorder or duplicate manifest rows.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def allocate_counts(n: int) -> dict[str, int]:
    """Largest-remainder allocation of n rows across RATIOS, ties train>val>test."""
    order = ["train", "val", "test"]
    raw = {k: n * RATIOS[k] for k in order}
    base = {k: int(raw[k]) for k in order}
    remainder = n - sum(base.values())
    fractions = sorted(order, key=lambda k: (-(raw[k] - base[k]), order.index(k)))
    for k in fractions[:remainder]:
        base[k] += 1
    assert sum(base.values()) == n
    return base


def assign_splits(rows: list[dict], seed: int) -> None:
    """Mutate rows in place, adding/overwriting a 'split' key on each."""
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_bucket[row["bucket"]].append(idx)

    rng = random.Random(seed)
    for bucket, indices in by_bucket.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        counts = allocate_counts(len(shuffled))
        cursor = 0
        for split_name in ["train", "val", "test"]:
            n = counts[split_name]
            for idx in shuffled[cursor : cursor + n]:
                rows[idx]["split"] = split_name
            cursor += n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("out/conversions/v2/test_set/manifest.csv"),
        help="Path to manifest.csv (default: out/conversions/v2/test_set/manifest.csv, "
        "resolved relative to cwd — run from ME2/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.manifest.exists():
        print(f"error: manifest not found at {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print(f"error: manifest at {args.manifest} has no rows", file=sys.stderr)
        return 1

    assign_splits(rows, args.seed)

    if "split" not in fieldnames:
        fieldnames = fieldnames + ["split"]

    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_split = Counter(row["split"] for row in rows)
    per_bucket_split = Counter((row["bucket"], row["split"]) for row in rows)
    print(f"seed={args.seed} total={len(rows)}")
    print(f"overall: {dict(per_split)}")
    for bucket in sorted({row["bucket"] for row in rows}):
        counts = {
            split_name: per_bucket_split[(bucket, split_name)]
            for split_name in ["train", "val", "test"]
        }
        print(f"  {bucket}: {counts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
