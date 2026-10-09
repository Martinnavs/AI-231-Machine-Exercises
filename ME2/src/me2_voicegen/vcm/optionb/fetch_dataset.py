"""Clone the upstream AI231 `MEX2/OptionB` dataset and convert it into
`out/conversions/v2/optionb/`'s VCM manifest schema (see the owning ticket,
`.scratch/optionb-dataset/tickets/02-fetch-convert.md`, for the full design
rationale -- summarized here only where it isn't obvious from the code).

The upstream `manifest.csv` is **untrusted input**: it comes from a
third-party GitHub repo and its `path` column is used to build filesystem
read/write destinations. Every path derived from it is resolved and
verified to stay under the relevant root (the clone dir for reads, this
module's `out_root` for writes) before any I/O happens -- see
`resolve_under()`/`PathTraversalError`. Nothing in the cloned tree is ever
executed or imported; it is read only via `csv`/`wave`/`shutil`.

`label` in the emitted manifest is upstream's `intent` column, not its own
`label` column -- `evaluate_grammar` keys exact-match accuracy off this
field against a 19-way intent predictor, so the wrong source column would
make accuracy read as 0 (see docs in the owning ticket, decision D4).

Every copied file's sample rate is measured from disk (D13) and, if it
isn't already 16 kHz, resampled to 16 kHz mono 16-bit in place before the
manifest row is written -- `sample_rate`/`resampled` then describe the
final on-disk format. `VCMDataset._load_waveform`'s own
`sample_rate != int(row["sample_rate"])` check is a safety net for a
manifest/file mismatch it wasn't told about; it is not a substitute for
writing the correct rate here, since a converter that always records a
file's own (possibly wrong) measured rate makes that check compare a
value against itself and it can never fire.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

REPO_URL = "https://github.com/markandrian30/AI231"
SPARSE_PATH = "MEX2/OptionB"
FLAGGED_SUBDIR = "FLAGGED"

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2

UPSTREAM_MANIFEST_COLUMNS = {
    "path",
    "label",
    "intent",
    "speaker",
    "split",
    "phrase_id",
    "variant_id",
    "transcript",
    "slot",
    "slot_value",
    "duration_sec",
}

# Sanity bounds only -- not trusted values. Real corpus (verified this
# session against a real clone): 17,656 rows, ~985 MB of active WAV content.
# These exist so a wildly wrong/adversarial manifest fails loudly instead of
# silently producing a tiny or unbounded dataset.
MIN_EXPECTED_ROWS = 15_000
MAX_EXPECTED_ROWS = 20_000
MAX_TOTAL_BYTES = 5 * 1024**3
MAX_SINGLE_FILE_BYTES = 200 * 1024**2

PROBE_BUCKETS = ("babble", "silence")

VALID_SPLITS = frozenset({"train", "val", "test"})

MANIFEST_FIELDS = [
    "filename",
    "path",
    "bucket",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
    "transcript",
]

# Upstream labels/intents/speakers are restricted to this charset; anything
# else is a sign the manifest is either corrupt or hostile, not a legitimate
# new category, so it's rejected rather than path-sanitized-and-allowed.
_SAFE_COMPONENT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


class ManifestValidationError(RuntimeError):
    pass


class PathTraversalError(RuntimeError):
    pass


def resolve_under(root: Path, candidate: Path) -> Path:
    """Resolve `candidate` and assert it stays under `root`. Raises
    PathTraversalError otherwise. Used for every filesystem path derived
    from the untrusted upstream manifest, on both the read side (clone dir)
    and the write side (out_root)."""
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        raise PathTraversalError(
            f"{candidate} resolves to {candidate_resolved}, which escapes {root_resolved}"
        ) from None
    return candidate_resolved


def sanitize_component(value: str, what: str) -> str:
    """Validate a single upstream-derived string destined to become a path
    *component* (not a whole path): non-empty, no separators, no `..`, and
    drawn only from `_SAFE_COMPONENT`."""
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise PathTraversalError(f"unsafe {what} from upstream manifest: {value!r}")
    if not set(value) <= _SAFE_COMPONENT:
        raise PathTraversalError(f"unsafe characters in {what} from upstream manifest: {value!r}")
    return value


def git_commit_sha(repo_dir: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def clone_optionb(clone_dir: Path, repo_url: str = REPO_URL, force: bool = False) -> tuple[Path, str]:
    """Blob-filtered, sparse-checkout clone of `repo_url` into `clone_dir`,
    materializing only `MEX2/OptionB` minus `FLAGGED/`. Reused across runs
    (idempotent, no re-download) unless `force` is set or the existing
    directory doesn't look like a valid prior clone.

    Returns (optionb_dir, commit_sha). Hooks are disabled
    (`core.hooksPath=/dev/null`) and cloned content is never executed."""
    optionb_dir = clone_dir / "MEX2" / "OptionB"
    git_dir = clone_dir / ".git"

    if not force and git_dir.is_dir() and (optionb_dir / "manifest.csv").is_file():
        return optionb_dir, git_commit_sha(clone_dir)

    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    clone_dir.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--no-local",
            repo_url,
            str(clone_dir),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "sparse-checkout", "init", "--no-cone"],
        check=True,
    )
    sparse_checkout_file = git_dir / "info" / "sparse-checkout"
    sparse_checkout_file.write_text(
        f"{SPARSE_PATH}/*\n!{SPARSE_PATH}/{FLAGGED_SUBDIR}/\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "sparse-checkout", "reapply"],
        check=True,
    )

    if not (optionb_dir / "manifest.csv").is_file():
        raise FileNotFoundError(
            f"sparse checkout did not materialize {optionb_dir / 'manifest.csv'}"
        )
    return optionb_dir, git_commit_sha(clone_dir)


def load_upstream_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not UPSTREAM_MANIFEST_COLUMNS <= set(reader.fieldnames):
            missing = UPSTREAM_MANIFEST_COLUMNS - set(reader.fieldnames or [])
            raise ManifestValidationError(
                f"{manifest_path}: missing required column(s) {sorted(missing)}"
            )
        rows = list(reader)

    if not (MIN_EXPECTED_ROWS <= len(rows) <= MAX_EXPECTED_ROWS):
        raise ManifestValidationError(
            f"{manifest_path}: {len(rows)} rows is outside the expected sanity range "
            f"[{MIN_EXPECTED_ROWS}, {MAX_EXPECTED_ROWS}] -- refusing to proceed silently"
        )
    return rows


def count_active_wavs(optionb_dir: Path) -> int:
    return sum(
        1
        for p in optionb_dir.rglob("*.wav")
        if FLAGGED_SUBDIR not in p.relative_to(optionb_dir).parts
    )


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


def resample_to_16k_in_place(path: Path) -> None:
    """Resample `path` (already copied, mono/16-bit verified) to
    REQUIRED_SR mono 16-bit PCM, overwriting it in place. Mirrors
    `vcm/slot_eval_set.py`'s `resample_mono_16k`/`write_eval_clip` pattern:
    resample via librosa, then re-probe the written file to confirm the
    format actually landed rather than trusting the resample call's intent."""
    import librosa

    data, orig_sr = sf.read(str(path), dtype="float32")
    resampled = librosa.resample(data, orig_sr=orig_sr, target_sr=REQUIRED_SR)
    sf.write(str(path), resampled, REQUIRED_SR, subtype="PCM_16")

    probe = probe_wav(path)
    if (probe.sample_rate, probe.channels, probe.sampwidth) != (
        REQUIRED_SR,
        REQUIRED_CHANNELS,
        REQUIRED_SAMPWIDTH,
    ):
        raise RuntimeError(
            f"{path}: resampled format {probe.sample_rate}Hz/{probe.channels}ch/"
            f"{probe.sampwidth * 8}bit != required {REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/"
            f"{REQUIRED_SAMPWIDTH * 8}bit"
        )


def build_target_manifest_rows(
    upstream_rows: list[dict], optionb_dir: Path, out_root: Path
) -> tuple[list[dict], list[tuple[Path, Path]]]:
    """Validate + map every upstream row into a 12-column manifest row plus
    its (verified src, verified dest) copy job. Raises on any path that
    would escape `optionb_dir` (reads) or `out_root` (writes), or on an
    unsafe filename/label component. Does not touch the filesystem beyond
    the read-only `manifest.csv` already loaded by the caller."""
    manifest_rows: list[dict] = []
    copy_jobs: list[tuple[Path, Path]] = []
    seen_dest: dict[Path, str] = {}

    for row in upstream_rows:
        upstream_relpath = row["path"]
        filename = sanitize_component(Path(upstream_relpath).name, "filename")
        label = sanitize_component(row["intent"], "intent/label")
        group_id = sanitize_component(row["speaker"], "speaker/group_id")
        split = row["split"]
        if split not in VALID_SPLITS:
            raise ManifestValidationError(
                f"unsafe split {split!r} from upstream manifest for path={upstream_relpath!r} "
                f"-- expected one of {sorted(VALID_SPLITS)}"
            )

        src = resolve_under(optionb_dir, optionb_dir / upstream_relpath)
        dest = resolve_under(out_root, out_root / "audio" / label / filename)

        if dest in seen_dest:
            raise ManifestValidationError(
                f"duplicate destination {dest} for rows with path={upstream_relpath!r} "
                f"and {seen_dest[dest]!r}"
            )
        seen_dest[dest] = upstream_relpath

        manifest_rows.append(
            {
                "filename": filename,
                "path": str(dest.relative_to(out_root.resolve())),
                "bucket": "target_commands",
                "label": label,
                "duration": None,  # filled in after copy, from the written file
                "sample_rate": None,
                "resampled": "False",
                "source_dataset": "optionb",
                "source_relpath": upstream_relpath,
                "group_id": group_id,
                "split": split,
                "transcript": row["transcript"],
            }
        )
        copy_jobs.append((src, dest))

    return manifest_rows, copy_jobs


def build_probe_manifest_rows(test_set_manifest_path: Path) -> list[dict]:
    """The 786 existing babble/silence rows from `test_set/manifest.csv`,
    reprojected as rows (not files) with `path` rewritten to
    `../test_set/...` -- see decision D6c in the owning ticket. Zero bytes
    are written for these; they only add reject-probe rows so the
    threshold sweep in `evaluate.py` isn't degenerate on an Option-B-only
    manifest."""
    with test_set_manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    probe_rows = []
    for row in rows:
        if row["bucket"] not in PROBE_BUCKETS:
            continue
        probe_rows.append(
            {
                "filename": row["filename"],
                "path": f"../test_set/{row['path']}",
                "bucket": row["bucket"],
                "label": row["label"],
                "duration": row["duration"],
                "sample_rate": row["sample_rate"],
                "resampled": row["resampled"],
                "source_dataset": row["source_dataset"],
                "source_relpath": row["source_relpath"],
                "group_id": row["group_id"],
                "split": row["split"],
                "transcript": "",
            }
        )
    return probe_rows


def copy_and_measure(
    copy_jobs: list[tuple[Path, Path]], manifest_rows: list[dict]
) -> int:
    """Copy every (src, dest) pair, verify the written file is mono 16-bit
    PCM (raising loudly if not), and fill in each row's `duration`/
    `sample_rate` measured from the *final* written file (D13), not copied
    from upstream's `duration_sec` or assumed to be 16000. Returns total
    bytes copied, checked against MAX_TOTAL_BYTES/MAX_SINGLE_FILE_BYTES as
    it goes.

    D13's "verified 16kHz on 2 files" claim does NOT hold across the full
    corpus: a real run found 196 VOLUME_DOWN/v2 clips at 12000 Hz. Any file
    whose copied sample rate isn't REQUIRED_SR is resampled in place to
    16 kHz (mirroring `vcm/slot_eval_set.py`'s resample-then-verify
    pattern) *before* `row["sample_rate"]`/`row["duration"]` are measured,
    so the manifest always records the rate the file is actually at and
    `resampled` reflects whether that happened -- writing the file's own
    unresampled rate into `sample_rate` would make
    `VCMDataset._load_waveform`'s `sample_rate != int(row["sample_rate"])`
    resample-on-mismatch check never fire, since the two values would be
    equal by construction."""
    total_bytes = 0
    for (src, dest), row in zip(copy_jobs, manifest_rows):
        size = src.stat().st_size
        if size > MAX_SINGLE_FILE_BYTES:
            raise ManifestValidationError(f"{src}: {size} bytes exceeds per-file cap {MAX_SINGLE_FILE_BYTES}")
        total_bytes += size
        if total_bytes > MAX_TOTAL_BYTES:
            raise ManifestValidationError(f"cumulative copy size exceeded cap {MAX_TOTAL_BYTES} bytes")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

        probe = probe_wav(dest)
        if (probe.channels, probe.sampwidth) != (REQUIRED_CHANNELS, REQUIRED_SAMPWIDTH):
            raise RuntimeError(
                f"{dest}: {probe.channels}ch/{probe.sampwidth * 8}bit != required "
                f"{REQUIRED_CHANNELS}ch/{REQUIRED_SAMPWIDTH * 8}bit (no format editing beyond "
                f"sample-rate resampling is done here)"
            )

        resampled = False
        if probe.sample_rate != REQUIRED_SR:
            resample_to_16k_in_place(dest)
            probe = probe_wav(dest)
            resampled = True

        row["duration"] = f"{probe.duration:.6f}"
        row["sample_rate"] = str(probe.sample_rate)
        row["resampled"] = "True" if resampled else "False"

    return total_bytes


def verify_manifest_files_exist(manifest_rows: list[dict], out_root: Path) -> None:
    seen: set[str] = set()
    for row in manifest_rows:
        if row["filename"] in seen and row["bucket"] == "target_commands":
            raise ManifestValidationError(f"filename {row['filename']!r} referenced by more than one target row")
        seen.add(row["filename"])
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


def write_summary(
    out_root: Path,
    *,
    commit_sha: str,
    du_bytes: int,
    active_wav_count: int,
    manifest_row_count: int,
    distinct_labels: int,
    probe_row_count: int,
    license_note: str,
    sample_rate_counts: dict[str, int],
) -> None:
    lines = [
        "# Option B dataset conversion summary",
        "",
        f"Upstream repo: `markandrian30/AI231`, path `MEX2/OptionB`, commit `{commit_sha}`.",
        "",
        "## Row-count reconciliation (D8)",
        "",
        f"- Active WAV files found in the clone (excluding `FLAGGED/`): **{active_wav_count}**",
        f"- Upstream `manifest.csv` data rows: **{manifest_row_count}**",
        "- These two numbers matched exactly in this run (17,656 in the reference clone this "
        "ticket was written against). The upstream README's own summary table also states "
        "17,656 remaining active WAV files; an earlier note citing 17,658/8,829+8,829 does not "
        "reproduce against the README's own table or the real clone and should be treated as "
        "stale.",
        "",
        "## Folder count (D9)",
        "",
        f"- Distinct `label` (intent) directories written: **{distinct_labels}** "
        "(19 intents; 13 unslotted + 6 slotted collapse to 19 once grouped by intent rather "
        "than upstream's 31-value `label` column).",
        "",
        "## Disk usage",
        "",
        f"- `du -sh` of the checkout (active content, excluding `FLAGGED/`): {du_bytes / (1024**3):.2f} GiB",
        "",
        "## Audio format (D13)",
        "",
        "- `sample_rate` is measured per file from the final written WAV, not hardcoded. "
        f"Distribution across target rows: {dict(sorted(sample_rate_counts.items()))}. The "
        "2-file spot check this ticket started from does not hold across the full corpus -- a "
        "real run found 196 `VOLUME_DOWN`/`v2` clips at 12000 Hz rather than 16000 Hz. Those "
        "196 clips are resampled to 16 kHz mono 16-bit in place before this manifest is "
        "written, with `resampled=True` and `sample_rate=16000` recorded for them (verified by "
        "reading the resampled file back); all other rows are `resampled=False` at their "
        "native 16000 Hz.",
        "",
        "## Reject-probe rows carried from test_set (D6c)",
        "",
        f"- {probe_row_count} `babble`/`silence` rows appended with `path` rewritten to "
        "`../test_set/...`; zero bytes copied, zero files duplicated.",
        "",
        "## License",
        "",
        license_note,
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "me2_optionb_clone",
        help="scratch clone location, outside the project repo tree (default: a stable temp dir, "
        "reused across runs so re-running does not re-download)",
    )
    parser.add_argument("--force-clone", action="store_true", help="ignore any existing clone and re-clone")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/conversions/v2/optionb"),
        help="output dir (default: out/conversions/v2/optionb)",
    )
    parser.add_argument(
        "--test-set-manifest",
        type=Path,
        default=Path("out/conversions/v2/test_set/manifest.csv"),
        help="existing test_set manifest to source babble/silence probe rows from (read-only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing anything")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_root

    try:
        optionb_dir, commit_sha = clone_optionb(args.clone_dir, args.repo_url, force=args.force_clone)
        upstream_rows = load_upstream_manifest(optionb_dir / "manifest.csv")
        active_wav_count = count_active_wavs(optionb_dir)
        if active_wav_count != len(upstream_rows):
            raise ManifestValidationError(
                f"active WAV count on disk ({active_wav_count}) != manifest row count "
                f"({len(upstream_rows)}) -- refusing to proceed on an unreconciled discrepancy"
            )

        du_bytes = sum(p.stat().st_size for p in optionb_dir.rglob("*") if p.is_file() and FLAGGED_SUBDIR not in p.relative_to(optionb_dir).parts)
        if du_bytes > MAX_TOTAL_BYTES:
            raise ManifestValidationError(
                f"cloned checkout is {du_bytes} bytes, exceeding cap {MAX_TOTAL_BYTES} -- "
                "refusing to proceed with conversion"
            )

        staging_root = out_root.parent / f".{out_root.name}.staging"
        manifest_rows, copy_jobs = build_target_manifest_rows(upstream_rows, optionb_dir, staging_root)
        probe_rows = build_probe_manifest_rows(args.test_set_manifest)
        all_rows = manifest_rows + probe_rows
        distinct_labels = len({r["label"] for r in manifest_rows})

        license_path = optionb_dir.parent.parent / "LICENSE"
        license_note = (
            "No LICENSE file present in the upstream repo (checked at clone root). Nothing from "
            "this dataset is redistributed -- `out/` is gitignored -- but treat it as "
            "all-rights-reserved absent an explicit license if that ever changes."
            if not license_path.is_file()
            else f"See {license_path}"
        )
    except (
        FileNotFoundError,
        ManifestValidationError,
        PathTraversalError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        total_bytes = sum(src.stat().st_size for src, _dest in copy_jobs)
        print(
            f"dry-run: would write {len(manifest_rows)} target rows + {len(probe_rows)} probe rows "
            f"= {len(all_rows)} manifest rows, {len(copy_jobs)} files, "
            f"{total_bytes / (1024**3):.2f} GiB, to {out_root}"
        )
        return 0

    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    sample_rate_counts: dict[str, int] = {}
    try:
        copy_and_measure(copy_jobs, manifest_rows)
        verify_manifest_files_exist(all_rows, staging_root)

        for row in manifest_rows:
            sample_rate_counts[row["sample_rate"]] = sample_rate_counts.get(row["sample_rate"], 0) + 1

        write_manifest(staging_root, all_rows)
        write_summary(
            staging_root,
            commit_sha=commit_sha,
            du_bytes=du_bytes,
            active_wav_count=active_wav_count,
            manifest_row_count=len(upstream_rows),
            distinct_labels=distinct_labels,
            probe_row_count=len(probe_rows),
            license_note=license_note,
            sample_rate_counts=sample_rate_counts,
        )
    except (ManifestValidationError, RuntimeError, PathTraversalError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    staging_root.rename(out_root)

    print(f"{len(all_rows)} rows ({len(manifest_rows)} target + {len(probe_rows)} probe) -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
