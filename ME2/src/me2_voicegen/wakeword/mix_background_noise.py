"""Probabilistically mix `out/conversions/v2/background_noise` (ESC-50
ambient/domestic noise) into a copy of a sample of `adversaries/`
(`_unknown_`) and/or `positives_converted/` (`_wakeword_`) clips, additively
-- originals are never modified, a noise-mixed copy is written alongside
them in a new sibling subset (`<source>_noisy/`), padding the dataset.

This is a deliberate, documented exception to this feature's own prior
Non-Goal ("noise/RIR augmentation is train-time only, via
`common/augment.py`, not a dataset-build concern") and to
`docs/WAKEWORD-DATASET-CONTRACT.md` section 3's original exclusion of
`background_noise` from this dataset. `background_noise` is ESC-50,
licensed CC-BY-NC-SA-4.0 -- baking it into stored clips here means the
wakeword dataset as a whole is no longer cleanly Apache-2.0/public-domain
once these rows are included. See `docs/WAKEWORD-DATASET-CONTRACT.md`
section 7 for the accepted-encumbrance decision this module implements.

Mixing itself reuses `common.augment.apply_noise` (SNR-based additive
mixing, already used for VCM train-time augmentation) rather than
reimplementing it. Every written wav is format-verified the same way every
other wakeword script does (`fetch_positives.probe_wav`/
`resample_to_16k_in_place`).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from me2_voicegen.common.augment import SNR_MAX_DB, SNR_MIN_DB, apply_noise
from me2_voicegen.wakeword.fetch_positives import (
    MANIFEST_FIELDS as BASE_MANIFEST_FIELDS,
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

DEFAULT_NOISE_PROB = 0.35
DEFAULT_NOISE_ROOT = Path("out/conversions/v2/background_noise")
DEFAULT_WAKEWORD_ROOT = Path("out/conversions/v2/wakeword")

# One entry per source subset this module knows how to augment. `_silence_`
# (synthetic-only, contract section 3) and `positives_real` are
# deliberately excluded -- not requested, and augmenting real positives was
# never asked for either.
SOURCE_SUBSETS = ("adversaries", "positives_converted")

MANIFEST_FIELDS = BASE_MANIFEST_FIELDS + ["ref_voice", "noise_source_file", "snr_db"]

_UNSAFE_SLUG_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: str) -> str:
    """Turn an arbitrary manifest value (e.g. `adversaries`' natural-
    language phrase `group_id`, which legitimately contains spaces -- see
    `generate_adversaries.py`'s `ADVERSARY_PHRASES`) into a filesystem-safe
    filename component. Not a security boundary on its own -- the fully
    assembled filename is still run through `sanitize_component` below,
    which is."""
    slug = _UNSAFE_SLUG_CHARS.sub("-", value).strip("-")
    if not slug:
        raise ManifestValidationError(f"cannot derive a filename-safe slug from {value!r}")
    return slug


def load_source_manifest(subset_root: Path) -> list[dict]:
    manifest_path = subset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise ManifestValidationError(f"source manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestValidationError(f"source manifest is empty: {manifest_path}")
    required = {"filename", "path", "label", "source_dataset", "group_id"}
    missing = required - set(rows[0])
    if missing:
        raise ManifestValidationError(f"source manifest missing required columns {missing}: {manifest_path}")
    return sorted(rows, key=lambda r: (r["group_id"], r["filename"]))


def load_noise_manifest(noise_root: Path) -> list[dict]:
    manifest_path = noise_root / "manifest.csv"
    if not manifest_path.is_file():
        raise ManifestValidationError(f"background_noise manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestValidationError(f"background_noise manifest is empty: {manifest_path}")
    return sorted(rows, key=lambda r: r["filename"])


def plan_jobs(
    source_rows: list[dict],
    noise_rows: list[dict],
    *,
    noise_prob: float,
    snr_min_db: float,
    snr_max_db: float,
    seed: int,
) -> list[dict]:
    """Deterministic (given `seed`) selection of which source rows get a
    noisy copy, plus which noise clip/SNR each selected row draws. Visits
    `source_rows` in the order already sorted by `load_source_manifest`
    (group_id, filename) so re-running with the same seed and the same
    input manifest reproduces byte-identical output. Does not touch the
    filesystem."""
    rng = random.Random(seed)
    jobs: list[dict] = []
    for row in source_rows:
        if rng.random() >= noise_prob:
            continue
        noise_row = noise_rows[rng.randrange(len(noise_rows))]
        snr_db = rng.uniform(snr_min_db, snr_max_db)
        jobs.append({"source_row": row, "noise_row": noise_row, "snr_db": snr_db})
    return jobs


def dest_filename(source_row: dict, noise_row: dict) -> str:
    """Derive a filename that stays unique within one `_noisy` subset's own
    `audio/` dir. Bare `filename` alone is NOT a reliable unique key across
    all wakeword subsets (`positives_converted`'s own filenames repeat once
    per `group_id`, by design -- see docs/WAKEWORD-DATASET-CONTRACT.md
    section 5); `(group_id, filename)` is the pair this feature's own
    manifests already guarantee unique, so that pair is what this filename
    is built from."""
    group_slug = _slug(source_row["group_id"])
    source_slug = _slug(Path(source_row["filename"]).stem)
    noise_slug = _slug(Path(noise_row["filename"]).stem)
    return sanitize_component(f"{group_slug}__{source_slug}__noise-{noise_slug}.wav", "dest filename")


def mix_one(
    *,
    source_row: dict,
    noise_row: dict,
    snr_db: float,
    subset_root: Path,
    noise_root: Path,
    staging_root: Path,
) -> dict:
    """Load the source clip + noise clip, mix at `snr_db`, write the result
    under `staging_root/audio/`, format-verify it, and return the new
    manifest row. Raises on any path escaping its root or on unexpected
    input format (source/noise clips are contractually already 16kHz/mono/
    16-bit -- probed, not assumed)."""
    source_path = resolve_under(subset_root, subset_root / source_row["path"])
    if not source_path.is_file():
        raise ManifestValidationError(f"source clip missing on disk: {source_path}")
    noise_path = resolve_under(noise_root, noise_root / "audio" / noise_row["filename"])
    if not noise_path.is_file():
        raise ManifestValidationError(f"noise clip missing on disk: {noise_path}")

    source_probe = probe_wav(source_path)
    if (source_probe.sample_rate, source_probe.channels, source_probe.sampwidth) != (
        REQUIRED_SR,
        REQUIRED_CHANNELS,
        REQUIRED_SAMPWIDTH,
    ):
        raise ManifestValidationError(
            f"{source_path}: {source_probe.sample_rate}Hz/{source_probe.channels}ch/"
            f"{source_probe.sampwidth * 8}bit != required {REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/"
            f"{REQUIRED_SAMPWIDTH * 8}bit -- refusing to mix a non-conformant source clip"
        )

    source_wave, _sr = torchaudio.load(str(source_path))
    noise_wave, _sr = torchaudio.load(str(noise_path))
    mixed = apply_noise(source_wave[0], noise_wave[0], snr_db)
    mixed = torch.clamp(mixed, -1.0, 1.0)

    dest = resolve_under(
        staging_root, staging_root / "audio" / dest_filename(source_row, noise_row)
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), mixed.numpy(), REQUIRED_SR, subtype="PCM_16")

    probe = probe_wav(dest)
    resampled = False
    if (probe.sample_rate, probe.channels, probe.sampwidth) != (
        REQUIRED_SR,
        REQUIRED_CHANNELS,
        REQUIRED_SAMPWIDTH,
    ):
        resample_to_16k_in_place(dest)
        probe = probe_wav(dest)
        resampled = True

    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "filename": dest.name,
            "path": str(dest.relative_to(staging_root)),
            "label": source_row["label"],
            "duration": f"{probe.duration:.6f}",
            "sample_rate": str(probe.sample_rate),
            "resampled": "True" if resampled else "False",
            "source_dataset": f"{source_row['source_dataset']}_noisy",
            "source_relpath": source_row["source_relpath"],
            "group_id": source_row["group_id"],
            "split": "",
            "ref_voice": source_row.get("ref_voice", ""),
            "noise_source_file": noise_row["filename"],
            "snr_db": f"{snr_db:.2f}",
        }
    )
    return row


def write_manifest(out_root: Path, rows: list[dict]) -> Path:
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_summary(
    out_root: Path,
    *,
    source_subset: str,
    seed: int,
    noise_prob: float,
    snr_min_db: float,
    snr_max_db: float,
    n_candidates: int,
    n_planned: int,
    n_written: int,
    noise_category_counts: dict[str, int],
) -> None:
    lines = [
        f"# `{source_subset}_noisy` -- probabilistic background-noise augmentation summary",
        "",
        f"Source subset: `{source_subset}`. seed={seed}, noise_prob={noise_prob}, "
        f"snr_db in [{snr_min_db}, {snr_max_db}].",
        f"{n_planned} of {n_candidates} source clips selected by noise_prob "
        f"({n_planned / n_candidates:.1%} realized, vs {noise_prob:.1%} configured).",
    ]
    if n_written != n_planned:
        lines.append(
            f"This invocation actually wrote {n_written} of those {n_planned} planned "
            "rows (--max-rows cap; re-run without it, or with a higher cap, for the rest)."
        )
    lines += [
        "",
        "## Noise category distribution (of the noise clips actually drawn)",
        "",
    ]
    for category, count in sorted(noise_category_counts.items()):
        lines.append(f"- `{category}`: {count}")
    lines += [
        "",
        "## License -- read before using this subset",
        "",
        "This subset mixes in audio from `out/conversions/v2/background_noise` "
        "(ESC-50, CC-BY-NC-SA-4.0 -- NonCommercial + ShareAlike; see "
        "`docs/WAKEWORD-DATASET-CONTRACT.md` section 7). Every row in this subset "
        "is therefore CC-BY-NC-SA-4.0-encumbered: non-commercial use only, and "
        "redistribution of this data (or of anything trained on it) must be shared "
        "under the same license. This is a deliberate, documented decision, not an "
        "oversight -- it is why this data lives in its own `_noisy` subset rather "
        "than being mixed silently into the clean source subset.",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        choices=[*SOURCE_SUBSETS, "both"],
        default="both",
        help="which source subset(s) to augment (default: both)",
    )
    parser.add_argument(
        "--wakeword-root",
        type=Path,
        default=DEFAULT_WAKEWORD_ROOT,
        help=f"parent dir holding every subset (default: {DEFAULT_WAKEWORD_ROOT}); "
        "source subsets are read from <wakeword-root>/<subset>, output subsets are "
        "written to <wakeword-root>/<subset>_noisy",
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=DEFAULT_NOISE_ROOT,
        help=f"background_noise corpus root (default: {DEFAULT_NOISE_ROOT})",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=DEFAULT_NOISE_PROB,
        help=f"per-clip probability of getting a noisy copy (default: {DEFAULT_NOISE_PROB})",
    )
    parser.add_argument("--snr-min-db", type=float, default=SNR_MIN_DB)
    parser.add_argument("--snr-max-db", type=float, default=SNR_MAX_DB)
    parser.add_argument("--seed", type=int, required=True, help="seed for deterministic selection/noise-pick/SNR")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="cap the number of rows actually mixed per source subset this invocation "
        "(for smoke runs); --dry-run always reports the full planned selection",
    )
    parser.add_argument("--dry-run", action="store_true", help="report planned selection without writing anything")
    return parser.parse_args(argv)


def run_subset(
    source_subset: str,
    *,
    wakeword_root: Path,
    noise_root: Path,
    noise_rows: list[dict],
    args: argparse.Namespace,
) -> int:
    subset_root = (wakeword_root / source_subset).resolve()
    out_root = wakeword_root / f"{source_subset}_noisy"

    source_rows = load_source_manifest(subset_root)
    jobs = plan_jobs(
        source_rows,
        noise_rows,
        noise_prob=args.noise_prob,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        seed=args.seed,
    )

    if args.dry_run:
        print(
            f"dry-run [{source_subset}]: {len(jobs)} of {len(source_rows)} clips selected "
            f"(noise_prob={args.noise_prob}) -> {out_root}"
        )
        return 0

    active_jobs = jobs if args.max_rows is None else jobs[: args.max_rows]

    staging_root = out_root.parent / f".{out_root.name}.staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    staging_root = staging_root.resolve()

    manifest_rows: list[dict] = []
    noise_category_counts: dict[str, int] = {}
    try:
        for job in active_jobs:
            row = mix_one(
                source_row=job["source_row"],
                noise_row=job["noise_row"],
                snr_db=job["snr_db"],
                subset_root=subset_root,
                noise_root=noise_root,
                staging_root=staging_root,
            )
            manifest_rows.append(row)
            category = job["noise_row"].get("category", "unknown")
            noise_category_counts[category] = noise_category_counts.get(category, 0) + 1

        write_manifest(staging_root, manifest_rows)
        write_summary(
            staging_root,
            source_subset=source_subset,
            seed=args.seed,
            noise_prob=args.noise_prob,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
            n_candidates=len(source_rows),
            n_planned=len(jobs),
            n_written=len(active_jobs),
            noise_category_counts=noise_category_counts,
        )
    except (ManifestValidationError, PathTraversalError, RuntimeError, OSError) as exc:
        print(f"error [{source_subset}]: {exc}", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    staging_root.rename(out_root)

    print(f"[{source_subset}]: {len(manifest_rows)} noisy clips -> {out_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subsets = list(SOURCE_SUBSETS) if args.source == "both" else [args.source]

    try:
        noise_rows = load_noise_manifest(args.noise_root)
    except ManifestValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    for subset in subsets:
        try:
            exit_code |= run_subset(
                subset,
                wakeword_root=args.wakeword_root,
                noise_root=args.noise_root.resolve(),
                noise_rows=noise_rows,
                args=args,
            )
        except (ManifestValidationError, PathTraversalError) as exc:
            print(f"error [{subset}]: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
