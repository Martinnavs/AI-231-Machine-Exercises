"""Scrub `out/conversions/v2/common_voice_negative` for any "computer" token
(never done on this corpus before -- it was previously only scrubbed
against the 20 VCM command phrases, which don't include "computer"), then
seeded-random sample the remainder down to `--target-count`, copy+resample
each selected clip to this feature's 16kHz/mono/16-bit invariant, and write
a self-contained subset:
`out/conversions/v2/wakeword/common_voice_negative_sample/{audio/,manifest.csv,summary.md}`.

This exists to broaden the `_unknown_` class beyond `adversaries`' 320
phonetic near-miss rows with real, licensed-CC0 general speech -- a
DS-CNN keyword spotter trained only on near-miss phrases never learns to
reject ordinary speech, which is a real false-accept risk in deployment,
not a cosmetic gap.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path

from me2_voicegen.wakeword.fetch_positives import (
    MANIFEST_FIELDS,
    ManifestValidationError,
    PathTraversalError,
    REQUIRED_CHANNELS,
    REQUIRED_SAMPWIDTH,
    REQUIRED_SR,
    probe_wav,
    resample_to_16k_in_place,
    resolve_under,
    sanitize_component,
)

LABEL_UNKNOWN = "_unknown_"
SOURCE_DATASET = "common_voice_negative"

DEFAULT_COMMON_VOICE_ROOT = Path("out/conversions/v2/common_voice_negative")
DEFAULT_OUT_ROOT = Path("out/conversions/v2/wakeword/common_voice_negative_sample")

# Word-boundary, case-insensitive; catches "computer", "computers",
# "computer's" (ticket 04's own stated requirement -- a bare-word match
# without the possessive/plural forms would silently under-scrub).
COMPUTER_TOKEN_RE = re.compile(r"\bcomputer(?:'s|s)?\b", re.IGNORECASE)


def contains_computer_token(row: dict) -> bool:
    text = f"{row.get('transcript', '')} {row.get('sentence', '')}"
    return bool(COMPUTER_TOKEN_RE.search(text))


def load_common_voice_manifest(root: Path) -> list[dict]:
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise ManifestValidationError(f"common_voice_negative manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestValidationError(f"common_voice_negative manifest is empty: {manifest_path}")
    required = {"filename", "transcript", "sentence", "source_file"}
    missing = required - set(rows[0])
    if missing:
        raise ManifestValidationError(f"common_voice_negative manifest missing required columns {missing}")
    return rows


def scrub_and_sample(rows: list[dict], target_count: int, seed: int) -> tuple[list[dict], int]:
    """Returns (sampled rows, dropped-by-scrub count). Sampling is seeded,
    without replacement, over rows sorted by filename first (determinism
    independent of on-disk manifest row order). Raises if the scrubbed pool
    can't supply `target_count` -- never silently clips."""
    scrubbed = sorted(
        (r for r in rows if not contains_computer_token(r)), key=lambda r: r["filename"]
    )
    dropped = len(rows) - len(scrubbed)

    if target_count > len(scrubbed):
        raise ManifestValidationError(
            f"requested target_count={target_count} exceeds the scrubbed pool "
            f"({len(scrubbed)} of {len(rows)} rows survive the 'computer'-token scrub)"
        )

    rng = random.Random(seed)
    sampled = rng.sample(scrubbed, target_count)
    return sampled, dropped


def build_manifest_row(row: dict, dest: Path, staging_root: Path, *, resampled: bool, probe) -> dict:
    return {
        "filename": dest.name,
        "path": str(dest.relative_to(staging_root)),
        "label": LABEL_UNKNOWN,
        "duration": f"{probe.duration:.6f}",
        "sample_rate": str(probe.sample_rate),
        "resampled": "True" if resampled else "False",
        "source_dataset": SOURCE_DATASET,
        "source_relpath": f"audio/{row['filename']}",
        "group_id": sanitize_component(Path(row["source_file"]).stem, "source_file stem/group_id"),
        "split": "",
    }


def copy_and_measure(sampled: list[dict], common_voice_root: Path, staging_root: Path) -> list[dict]:
    manifest_rows: list[dict] = []
    for row in sampled:
        filename = sanitize_component(row["filename"], "filename")
        src = resolve_under(common_voice_root, common_voice_root / "audio" / filename)
        if src.is_symlink() or not src.is_file():
            raise ManifestValidationError(f"{src}: refusing non-regular-file/symlinked source")

        dest = resolve_under(staging_root, staging_root / "audio" / filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

        probe = probe_wav(dest)
        if (probe.channels, probe.sampwidth) != (REQUIRED_CHANNELS, REQUIRED_SAMPWIDTH):
            raise RuntimeError(
                f"{dest}: {probe.channels}ch/{probe.sampwidth * 8}bit != required "
                f"{REQUIRED_CHANNELS}ch/{REQUIRED_SAMPWIDTH * 8}bit"
            )
        resampled = False
        if probe.sample_rate != REQUIRED_SR:
            resample_to_16k_in_place(dest)
            probe = probe_wav(dest)
            resampled = True

        manifest_rows.append(build_manifest_row(row, dest, staging_root, resampled=resampled, probe=probe))
    return manifest_rows


def write_manifest(out_root: Path, rows: list[dict]) -> Path:
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_summary(out_root: Path, *, seed: int, target_count: int, total_pool: int, dropped: int) -> None:
    group_counts: dict[str, int] = {}
    lines = [
        "# `common_voice_negative_sample` -- scrubbed/sampled negative-speech summary",
        "",
        f"seed={seed}. {total_pool} rows in `common_voice_negative`, {dropped} dropped by the "
        "'computer'-token scrub (case-insensitive, catches 'computer'/'computers'/\"computer's\", "
        "never applied to this corpus before -- it was previously only scrubbed against the 20 "
        "VCM command phrases), {remaining} survive, {target_count} sampled without replacement.".format(
            remaining=total_pool - dropped, target_count=target_count
        ),
        "",
        "## License",
        "",
        "`common_voice_negative` is CC0-1.0 (\"Creative Commons 0, No Rights Reserved\") -- verified "
        "against both Kaggle's dataset metadata and the archive's own `LICENSE.txt` "
        "(`out/conversions/v2/README.md` section 4). No encumbrance introduced by this subset.",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--common-voice-root", type=Path, default=DEFAULT_COMMON_VOICE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common_voice_root = args.common_voice_root.resolve()

    try:
        rows = load_common_voice_manifest(common_voice_root)
        sampled, dropped = scrub_and_sample(rows, args.target_count, args.seed)
    except ManifestValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"dry-run: {len(rows)} rows, {dropped} dropped by scrub, "
            f"{len(rows) - dropped} survive, {len(sampled)} would be sampled -> {args.out_root}"
        )
        return 0

    out_root: Path = args.out_root
    staging_root = out_root.parent / f".{out_root.name}.staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    staging_root = staging_root.resolve()

    try:
        manifest_rows = copy_and_measure(sampled, common_voice_root, staging_root)
        write_manifest(staging_root, manifest_rows)
        write_summary(
            staging_root,
            seed=args.seed,
            target_count=len(sampled),
            total_pool=len(rows),
            dropped=dropped,
        )
    except (ManifestValidationError, PathTraversalError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    staging_root.rename(out_root)

    print(f"{len(manifest_rows)} rows -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
