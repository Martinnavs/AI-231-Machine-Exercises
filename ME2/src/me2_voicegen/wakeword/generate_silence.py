"""Generate seeded synthetic colored-noise chunks for this feature's
`_silence_` class (see `docs/WAKEWORD-DATASET-CONTRACT.md`).

Per this repo's own `docs/raw_requirements/voice_generation_approach.md`,
`_silence_` is bulked out with Gaussian (white), pink, and brown noise at
varying volume levels rather than relying solely on the small real
YouTube-ambient pool (~258 clips, see ticket 04) -- that doc also warns
relying SOLELY on synthetic noise "will eventually cause real-world false
positives", which is why the real-ambient pool stays in the mix via ticket
04 rather than being replaced outright.

Colored noise is synthesized by shaping a white-noise spectrum in the
frequency domain (`docs/raw_requirements/sources.md`'s own linked
reference, voytekresearch's colored-noise tutorial, uses this same
technique): pink noise divides the white spectrum by `sqrt(f)` (power
spectral density ~ 1/f, i.e. -3dB/octave), brown divides by `f` (power
spectral density ~ 1/f^2, i.e. -6dB/octave). This is verified by spectral-
slope assertions in the test suite, not just assumed correct from this
generation code (a sign error or wrong exponent here would otherwise
silently produce "pink" noise that measures as white).

Every chunk is generated directly at 16kHz mono 16-bit PCM (no resampling
step needed, unlike `fetch_positives.py`'s upstream-sourced audio), then
read back and format-verified via the `wave` module before its manifest
row is written -- same discipline as `fetch_positives.py`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2

LABEL_SILENCE = "_silence_"
SOURCE_SYNTHETIC = "synthetic_noise"

NOISE_COLORS = ("white", "pink", "brown")

# Volume-level bands, in dBFS RMS (relative to int16 full scale). Three
# bands per color gives 3 colors x 3 bands = 9 groups total (see
# `group_id` reasoning below) -- enough spread to cover "varying volume
# levels" per the design note without fragmenting into groups so small
# ticket 04's proportional split allocation would have nothing to work
# with.
RMS_BANDS: dict[str, tuple[float, float]] = {
    "quiet": (-50.0, -40.0),
    "moderate": (-35.0, -25.0),
    "loud": (-20.0, -12.0),
}

# Real YouTube-ambient `_silence_` clips (ticket 04's pool) range
# 1.5-2.0s, mean ~1.71s (measured from
# out/conversions/v2/youtube_institutional/manifest.csv's `ambient`-bucket
# rows this session -- see ticket's Execution Log). Synthetic chunk
# duration is drawn from the same range so the combined `_silence_` class
# ticket 04 assembles doesn't have two visibly different duration
# populations.
MIN_DURATION_S = 1.5
MAX_DURATION_S = 2.0

# ~7x the real ambient pool (258), per the ticket's own framing of the
# gap this ticket exists to fill. Configurable via --total-count; this is
# a starting default, not a hard requirement.
DEFAULT_TOTAL_COUNT = 1806

# Keep headroom below full scale so per-sample fluctuations around the
# target RMS can't clip and distort the intended spectral shape.
PEAK_CEILING = 0.98

MANIFEST_FIELDS = [
    "filename",
    "path",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
]


class ManifestValidationError(RuntimeError):
    pass


def group_id_for(color: str, rms_band: str) -> str:
    """`<noise_color>_<rms_band>`, e.g. `pink_moderate`.

    Granularity reasoning (see ticket's Acceptance Criteria): one group
    per (color, rms_band) combination -- coarse enough that a single
    color/RMS combination is never itself split across train/val/test by
    ticket 04's group-disjoint splitter fragmenting it into slivers (each
    group here is 1/9th of the whole synthetic pool, hundreds of clips at
    the realistic scale this ticket targets, not a handful), yet fine
    enough that ticket 04's proportional allocator still has 9
    independent buckets to draw a proportional train/val/test split from
    per split, rather than being forced to put e.g. all of "pink" (at any
    volume) into one split wholesale.
    """
    return f"{color}_{rms_band}"


def colored_noise(n_samples: int, color: str, rng: np.random.Generator) -> np.ndarray:
    """Unit-std colored noise of `color` ("white"/"pink"/"brown"), shaped
    by dividing a white-noise spectrum by `f**0` / `sqrt(f)` / `f`
    respectively (see module docstring)."""
    if color not in NOISE_COLORS:
        raise ValueError(f"unknown noise color: {color!r}")

    white = rng.standard_normal(n_samples)
    if color == "white":
        noise = white
    else:
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs = freqs.copy()
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid divide-by-zero at DC
        if color == "pink":
            spectrum = spectrum / np.sqrt(freqs)
        else:  # brown
            spectrum = spectrum / freqs
        noise = np.fft.irfft(spectrum, n=n_samples)

    std = float(np.std(noise))
    if std == 0.0:
        raise RuntimeError(f"generated {color} noise has zero variance -- cannot normalize")
    return noise / std


def rms_dbfs_to_linear(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def render_chunk(
    color: str, target_rms_dbfs: float, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a float32 array in [-1, 1], `duration_s` seconds at
    REQUIRED_SR, of `color` noise scaled to `target_rms_dbfs` RMS, with
    peak-ceiling headroom applied if needed."""
    n_samples = int(round(duration_s * REQUIRED_SR))
    noise = colored_noise(n_samples, color, rng)

    target_rms = rms_dbfs_to_linear(target_rms_dbfs)
    current_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    scaled = noise * (target_rms / current_rms)

    peak = float(np.max(np.abs(scaled))) if n_samples else 0.0
    if peak > PEAK_CEILING:
        scaled = scaled * (PEAK_CEILING / peak)

    return scaled.astype(np.float32)


@dataclass(frozen=True)
class WavProbe:
    sample_rate: int
    channels: int
    sampwidth: int
    duration: float


def probe_wav(path: Path) -> WavProbe:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, nframes = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    return WavProbe(sample_rate=sr, channels=ch, sampwidth=sw, duration=nframes / float(sr))


def largest_remainder_allocation(total: int, weights: list[float]) -> list[int]:
    """Split `total` into `len(weights)` integer parts proportional to
    `weights`, using largest-remainder rounding so the parts sum exactly
    to `total` (mirrors this repo's existing allocator convention, e.g.
    `.scratch/kws-test-set/tickets/01-build-test-set.md`)."""
    if not weights:
        return []
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive number")

    raw = [total * w / weight_sum for w in weights]
    base = [int(x) for x in raw]
    remainder = total - sum(base)

    order = sorted(range(len(weights)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base


def plan_chunks(total_count: int, seed: int) -> list[dict]:
    """Deterministically plan every chunk's (color, rms_band, index,
    duration_s, target_rms_dbfs) without touching the filesystem or numpy's
    global RNG state, so the same `(total_count, seed)` always yields the
    same plan."""
    groups = [(color, band) for color in NOISE_COLORS for band in RMS_BANDS]
    counts = largest_remainder_allocation(total_count, [1.0] * len(groups))

    plan_rng = np.random.default_rng(seed)
    plan: list[dict] = []
    for (color, band), count in zip(groups, counts):
        lo, hi = RMS_BANDS[band]
        for idx in range(count):
            duration_s = float(plan_rng.uniform(MIN_DURATION_S, MAX_DURATION_S))
            target_rms_dbfs = float(plan_rng.uniform(lo, hi))
            plan.append(
                {
                    "color": color,
                    "rms_band": band,
                    "index": idx,
                    "duration_s": duration_s,
                    "target_rms_dbfs": target_rms_dbfs,
                }
            )
    return plan


def build_manifest_rows(plan: list[dict], out_root: Path) -> list[dict]:
    rows: list[dict] = []
    seen_dest: dict[str, dict] = {}
    for entry in plan:
        color, band, idx = entry["color"], entry["rms_band"], entry["index"]
        group_id = group_id_for(color, band)
        filename = f"{group_id}_{idx:05d}.wav"
        rel_path = f"audio/{SOURCE_SYNTHETIC}/{filename}"

        if rel_path in seen_dest:
            raise ManifestValidationError(f"duplicate destination {rel_path}")
        seen_dest[rel_path] = entry

        rows.append(
            {
                "filename": filename,
                "path": rel_path,
                "label": LABEL_SILENCE,
                "duration": None,
                "sample_rate": None,
                "resampled": "False",
                "source_dataset": SOURCE_SYNTHETIC,
                "source_relpath": (
                    f"synthetic:color={color};rms_band={band};"
                    f"target_rms_dbfs={entry['target_rms_dbfs']:.4f};index={idx}"
                ),
                "group_id": group_id,
                "split": "",
            }
        )
    return rows


def render_and_write(
    plan: list[dict], rows: list[dict], out_root: Path, seed: int
) -> None:
    """Render every chunk, write it under `out_root`, and fill in each
    row's `duration`/`sample_rate` measured from the *written* file (never
    assumed from the render parameters), format-verifying every file by
    reading it back via `wave`.

    Each chunk uses an independent RNG seeded from `(seed, chunk_index)`
    so rendering order never affects any individual chunk's samples, and
    the whole run is reproducible from `seed` alone."""
    audio_dir = out_root / "audio" / SOURCE_SYNTHETIC
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i, (entry, row) in enumerate(zip(plan, rows)):
        chunk_rng = np.random.default_rng((seed, i))
        samples = render_chunk(
            entry["color"], entry["target_rms_dbfs"], entry["duration_s"], chunk_rng
        )

        dest = out_root / row["path"]
        sf.write(str(dest), samples, REQUIRED_SR, subtype="PCM_16")

        probe = probe_wav(dest)
        if (probe.sample_rate, probe.channels, probe.sampwidth) != (
            REQUIRED_SR,
            REQUIRED_CHANNELS,
            REQUIRED_SAMPWIDTH,
        ):
            raise RuntimeError(
                f"{dest}: written format {probe.sample_rate}Hz/{probe.channels}ch/"
                f"{probe.sampwidth * 8}bit != required {REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/"
                f"{REQUIRED_SAMPWIDTH * 8}bit"
            )

        row["duration"] = f"{probe.duration:.6f}"
        row["sample_rate"] = str(probe.sample_rate)


def verify_manifest_files_exist(rows: list[dict], out_root: Path) -> None:
    for row in rows:
        resolved = (out_root / row["path"]).resolve()
        if not resolved.is_file():
            raise ManifestValidationError(f"manifest row path does not exist on disk: {row['path']} ({resolved})")


def write_manifest(out_root: Path, rows: list[dict]) -> Path:
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_summary(out_root: Path, *, rows: list[dict], seed: int, total_count: int) -> None:
    color_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    durations = []
    for row in rows:
        color = row["source_relpath"].split("color=", 1)[1].split(";", 1)[0]
        color_counts[color] = color_counts.get(color, 0) + 1
        group_counts[row["group_id"]] = group_counts.get(row["group_id"], 0) + 1
        durations.append(float(row["duration"]))

    rms_values = [
        float(row["source_relpath"].split("target_rms_dbfs=", 1)[1].split(";", 1)[0]) for row in rows
    ]

    lines = [
        "# Wakeword synthetic silence/noise generation summary",
        "",
        f"Seed: `{seed}`. Requested total: {total_count}. Realized total: {len(rows)}.",
        "",
        "## Noise-color counts",
        "",
        *(f"- `{color}`: {count}" for color, count in sorted(color_counts.items())),
        "",
        "## group_id (`<noise_color>_<rms_band>`) counts",
        "",
        *(f"- `{group}`: {count}" for group, count in sorted(group_counts.items())),
        "",
        "## RMS/volume-level distribution (dBFS)",
        "",
        f"- min: {min(rms_values):.2f}, max: {max(rms_values):.2f}, "
        f"mean: {sum(rms_values) / len(rms_values):.2f}" if rms_values else "- (no rows)",
        "",
        "## Duration distribution (seconds)",
        "",
        f"- min: {min(durations):.3f}, max: {max(durations):.3f}, "
        f"mean: {sum(durations) / len(durations):.3f}"
        if durations
        else "- (no rows)",
        f"- Drawn from [{MIN_DURATION_S}, {MAX_DURATION_S}]s to match the real "
        "YouTube-ambient `_silence_` pool's measured 1.5-2.0s range (ticket 04).",
        "",
        "## Format",
        "",
        f"Every file is {REQUIRED_SR}Hz mono {REQUIRED_SAMPWIDTH * 8}-bit PCM WAV, generated "
        "directly at that format (no resampling needed) and format-verified by reading the "
        "written file back.",
        "",
        "## License",
        "",
        "Pure local synthesis -- no external dependency, no license implication.",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/conversions/v2/wakeword/silence_synthetic"),
        help="output dir (default: out/conversions/v2/wakeword/silence_synthetic)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-count", type=int, default=DEFAULT_TOTAL_COUNT)
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing anything")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_root

    if args.total_count <= 0:
        print(f"error: --total-count must be positive, got {args.total_count}", file=sys.stderr)
        return 1

    plan = plan_chunks(args.total_count, args.seed)
    staging_root = out_root.parent / f".{out_root.name}.staging"
    rows = build_manifest_rows(plan, staging_root)

    if args.dry_run:
        by_color: dict[str, int] = {}
        for entry in plan:
            by_color[entry["color"]] = by_color.get(entry["color"], 0) + 1
        print(
            f"dry-run: would write {len(rows)} rows {dict(sorted(by_color.items()))}, "
            f"seed={args.seed}, to {out_root}"
        )
        return 0

    try:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True)

        render_and_write(plan, rows, staging_root, args.seed)
        verify_manifest_files_exist(rows, staging_root)
        write_manifest(staging_root, rows)
        write_summary(staging_root, rows=rows, seed=args.seed, total_count=args.total_count)
    except (ManifestValidationError, RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    staging_root.rename(out_root)

    print(f"{len(rows)} rows -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
