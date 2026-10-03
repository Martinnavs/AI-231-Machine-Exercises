"""Copy an internal manifest without any clip that is byte-identical to a clip of the ai231 test or holdout split.

    uv run python scripts/filter_internal_overlap.py --internal <dir with manifest.csv> --ai231 <ai231-v2 dir> --out <new dir>

A removed base clip also removes the noisy copies mixed from it (`*_ambient` rows, matched on the base file stem that
sits between the `__` markers of their filename). Hashing covers every non-generated internal row (not persona or noise
rows); all internal splits are filtered, so val (early stopping) is clean too. Paths in the new manifest are relative to
the new folder; the internal folder is not modified. Near-duplicates (re-encoded or trimmed copies) are not caught.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

SKIP_HASH = ("fil50_persona", "background_noise")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", type=Path, required=True)
    ap.add_argument("--ai231", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overlap-only", action="store_true", help="only write <out>/overlap_with_ai231.csv, not the manifest")
    a = ap.parse_args()
    ai = [r for r in csv.DictReader((a.ai231 / "manifest.csv").open(newline="", encoding="utf-8")) if r["split"] in ("test", "holdout")]
    by_hash: dict[str, list[dict]] = {}
    for r in ai:
        by_hash.setdefault(md5(a.ai231 / r["path"]), []).append(r)
    eval_hashes = set(by_hash)
    matched: set[str] = set()
    rows = list(csv.DictReader((a.internal / "manifest.csv").open(newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    removed_stems, removed = set(), Counter()
    for r in rows:
        sd = r["source_dataset"]
        if sd.endswith("_ambient") or sd in SKIP_HASH:
            continue
        h = md5(a.internal / r["path"])
        if h in eval_hashes:
            matched.add(h)
            removed_stems.add(Path(r["filename"]).stem)
            removed[(r["split"], sd)] += 1
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "overlap_with_ai231.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ai231_split", "ai231_path"])
        w.writerows([r["split"], r["path"]] for h in sorted(matched) for r in by_hash[h])
    if a.overlap_only:
        print(f"ai231 eval clips {len(ai)}; overlapping the internal data: {sum(len(by_hash[h]) for h in matched)}")
        return
    keep = []
    for r in rows:
        sd = r["source_dataset"]
        if sd.endswith("_ambient"):
            parts = Path(r["filename"]).stem.split("__")
            if len(parts) > 1 and parts[1] in removed_stems:
                removed[(r["split"], sd)] += 1
                continue
        elif Path(r["filename"]).stem in removed_stems and sd not in SKIP_HASH:
            continue
        keep.append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    for r in keep:
        r["path"] = os.path.relpath(os.path.realpath(a.internal / r["path"]), os.path.realpath(a.out))
    with (a.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    print(f"ai231 eval clips hashed {len(eval_hashes)}; internal rows {len(rows)} -> {len(keep)} kept")
    print("removed (split, source):", dict(sorted(removed.items())))
    print("train rows by source after:", dict(Counter(r["source_dataset"] for r in keep if r["split"] == "train").most_common(9)))


if __name__ == "__main__":
    main()
