#!/usr/bin/env python3
"""Pack the audio the training/eval manifests reference into one Kaggle-uploadable tar.

Manifest `path` columns are relative to the manifest's own directory (e.g.
`../optionb-v3/audio/...`, `../../../../raw_datasets/VCM_BALANCED/...`), so the
tar keeps every file at its repo-relative location (`out/conversions/v2/...`,
`raw_datasets/...`). Unpacked anywhere, the manifests resolve unchanged.

Only files a manifest actually references are packed (plus the manifests), not
whole directories -- `out/` holds ~100 GB of things training never reads.

    python scripts/kaggle/pack_data.py --out dist/kaggle/me2-data --owner <kaggle-user>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
from pathlib import Path

DEFAULT_MANIFESTS = [
    "out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv",  # train
    "out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv",  # eval gate
]


def referenced_files(manifest: Path) -> list[Path]:
    base = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as f:
        return [Path(os.path.normpath(base / r["path"])) for r in csv.DictReader(f)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", help="repeatable; default: the train + eval-gate manifests")
    ap.add_argument("--out", type=Path, default=Path("dist/kaggle/me2-data"))
    ap.add_argument("--owner", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--slug", default="me2-data")
    ap.add_argument("--dry-run", action="store_true", help="count files/bytes, write nothing")
    args = ap.parse_args()

    root = Path.cwd()
    manifests = [Path(m) for m in (args.manifest or DEFAULT_MANIFESTS)]
    files: set[Path] = set(manifests)
    for m in manifests:
        if not m.is_file():
            print(f"missing manifest: {m}", file=sys.stderr)
            return 1
        files.update(referenced_files(m))

    missing = sorted(p for p in files if not p.is_file())
    if missing:
        print(f"{len(missing)} referenced files missing, e.g. {missing[:3]}", file=sys.stderr)
        return 1
    outside = [p for p in files if p.is_absolute() or str(p).startswith("..")]
    if outside:
        print(f"files outside the repo root can't be packed repo-relative: {outside[:3]}", file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB from {len(manifests)} manifest(s)")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    tar_path = args.out / "me2-data.tar"
    # Uncompressed on purpose: WAV barely compresses, and extraction on Kaggle is then I/O-bound.
    with tarfile.open(tar_path, "w") as tar:
        for p in sorted(files):
            tar.add(root / p, arcname=str(p), recursive=False)
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "ME2 VCM training data",
                "id": f"{args.owner}/{args.slug}",
                "licenses": [{"name": "CC-BY-NC-SA-4.0"}],  # ESC-50 noise is in the mix (docs/VCM-CONTRACT.md s9)
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
