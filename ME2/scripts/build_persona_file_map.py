"""Map the persona pool's original file names to the published ones, by audio bytes (SHA-256).

The persona dataset on Hugging Face renamed its clips (`fil50_NNNNN_<voice>_<COMMAND>.wav`), but `build_ai231_fil50.py` selects by the original
`filename`. This needs the original audio, so it runs once where that audio lives and its output (`persona-pool/published_file_map.csv`) is committed.
`published_file` is empty for pool clips that were not published (old wordings such as "Place a call"; the four spelled-out-number clips the
manifest uses are in the `numeral_wordings` config, the rest of that kind are not published).

    PYTHONPATH=src uv run python scripts/build_persona_file_map.py --pool-manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
        --published raw_datasets/fil50-persona-export/fil50_persona <numeral_wordings dir> --out recipes/ai231-fil50-supp/persona-pool/published_file_map.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-manifest", type=Path, required=True)
    ap.add_argument("--published", type=Path, nargs="+", required=True, help="directories of the published train-*.parquet shards (default split, numeral_wordings)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    published: dict[str, str] = {}
    for shard in sorted(s for d in a.published for s in d.glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["audio", "file"]):
            for r in batch.to_pylist():
                sha = hashlib.sha256(r["audio"]["bytes"]).hexdigest()
                assert sha not in published, f"two published clips share audio: {r['file']}"
                published[sha] = Path(r["file"]).name
    rows = []
    with a.pool_manifest.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_dataset"] != "fil50_persona" or "_noisy" in r["filename"]:
                continue
            sha = hashlib.sha256((a.pool_manifest.parent / r["path"]).read_bytes()).hexdigest()
            rows.append({"filename": r["filename"], "published_file": published.get(sha, ""), "sha256": sha})
    assert len({r["sha256"] for r in rows}) == len(rows), "two pool clips share audio"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["filename", "published_file", "sha256"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["filename"]))
    n = sum(bool(r["published_file"]) for r in rows)
    print(f"{len(rows)} pool clips, {n} published, {len(rows) - n} not; {len(published)} published clips in all, {len(published) - n} not in the pool")


if __name__ == "__main__":
    main()
