"""Fetch real `"computer"` positive wakeword audio from two upstream repos
and assemble `out/conversions/v2/wakeword/positives_real/` in this
feature's manifest schema (see `docs/WAKEWORD-DATASET-CONTRACT.md`).

Closely mirrors `src/me2_voicegen/vcm/optionb/fetch_dataset.py`'s clone /
path-traversal / staging-and-swap discipline (see that module's own
docstring for the rationale this one shares). Two differences worth
calling out because they aren't obvious from the code:

1. Neither upstream repo ships a `manifest.csv` of its own -- both are
   fetched via a real, live, blob-filtered sparse `git clone` and the
   audio file listing is discovered from the checked-out directory tree,
   not read from any upstream-provided index.
2. Mycroft `Precise-Community-Data` licensing isn't a single repo-root
   `LICENSE` file (there isn't one); it's per-contributor waiver files
   under `licenses/`, each enumerating the files it covers as
   **repo-root-relative paths** (e.g. `computer/en/<uuid>.wav`), not bare
   filenames. `verify_mycroft_waiver_coverage()` re-derives this bijection
   from the live clone every run rather than trusting a cached count --
   see docs/WAKEWORD-DATASET-CONTRACT.md section 7 for why a bare-filename
   match would silently under-report coverage.

Nothing in either cloned tree is ever executed or imported; it is read
only via `csv`/`wave`/`shutil`/plain-text `Path.read_text`. Every path
derived from either clone's file listing is resolved and verified to stay
under the relevant root (the clone dir for reads, this module's staging
`out_root` for writes) before any I/O happens -- see
`resolve_under()`/`PathTraversalError`.
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

PICOVOICE_REPO_URL = "https://github.com/Picovoice/wake-word-benchmark"
PICOVOICE_SPARSE_PATH = "audio/computer"

MYCROFT_REPO_URL = "https://github.com/MycroftAI/Precise-Community-Data"
MYCROFT_SPARSE_PATH = "computer/en"
MYCROFT_LICENSES_PATH = "licenses"

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2

LABEL_WAKEWORD = "_wakeword_"
SOURCE_PICOVOICE = "picovoice"
SOURCE_MYCROFT = "mycroft_precise"

# Sanity bounds -- not trusted values, verified live against both upstreams
# this session (see the owning ticket's Execution Log). A real divergence
# from these exact counts means the upstream tree changed shape since this
# ticket was written and must fail loudly rather than silently adopting
# whatever count shows up.
EXPECTED_PICOVOICE_COUNT = 411
EXPECTED_MYCROFT_COUNT = 57
EXPECTED_TOTAL_ROWS = EXPECTED_PICOVOICE_COUNT + EXPECTED_MYCROFT_COUNT

# Real corpus is ~39 MB (Picovoice) + a few MB (Mycroft); these caps are
# generous headroom over that, not tuned to it, so a wildly wrong/hostile
# clone fails loudly instead of silently proceeding.
MAX_SINGLE_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024

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

# Upstream-derived filenames are restricted to this charset; anything else
# is a sign of a corrupt/hostile clone, not a legitimate new file, so it's
# rejected rather than path-sanitized-and-allowed.
_SAFE_COMPONENT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)

# Mycroft contributors have used both header lines interchangeably across
# the repo's history; both are treated as equivalent waiver headers.
WAIVER_HEADERS = (
    "Files released into public domain:",
    "Files covered by this update:",
)


class ManifestValidationError(RuntimeError):
    pass


class PathTraversalError(RuntimeError):
    pass


def resolve_under(root: Path, candidate: Path) -> Path:
    """Resolve `candidate` and assert it stays under `root`. Raises
    PathTraversalError otherwise. Used for every filesystem path derived
    from either clone's file listing, on both the read side (clone dir)
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
    *component* (not a whole path): non-empty, no separators, no `..`,
    drawn only from `_SAFE_COMPONENT`, and not leading with a character
    (`- + = @`) that would (a) be parsed as a CLI/git option if it ever
    reached a subprocess argv, or (b) trigger spreadsheet formula
    execution if it ever reached a CSV cell in a program that opens
    `manifest.csv` in a spreadsheet rather than reading it programmatically."""
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise PathTraversalError(f"unsafe {what} from upstream clone: {value!r}")
    if not set(value) <= _SAFE_COMPONENT:
        raise PathTraversalError(f"unsafe characters in {what} from upstream clone: {value!r}")
    if value[0] in "-+=@":
        raise PathTraversalError(f"unsafe leading character in {what} from upstream clone: {value!r}")
    return value


def assert_safe_rmtree_root(path: Path, what: str) -> None:
    """Reject a path this module might later pass to `shutil.rmtree()` if
    it's empty, `.`, or resolves to a filesystem root or the user's home
    directory. `Path("")` normalizes to `Path(".")`, so an empty CLI value
    for `--out-root`/`--clone-dir` is not a no-op: it silently targets the
    current working directory -- reproduced live (security-auditor finding
    R2-1 on this ticket) destroying a repo's working tree via
    `shutil.rmtree(out_root)`'s success-path swap, and reachable through
    this module's own Makefile target (`WAKEWORD_POSITIVES_OUT_DIR=`
    defined-but-empty isn't caught by `?=`). Deliberately conservative --
    over-rejecting a legitimate-but-unusual path is safe, under-rejecting
    is the bug this guards against."""
    raw = str(path)
    if raw.strip() in ("", "."):
        raise ManifestValidationError(
            f"{what} is empty or refers to the current directory ({raw!r}) -- refusing: this path "
            "may later be passed to shutil.rmtree() and an empty/'.' value would target the CWD"
        )
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ManifestValidationError(f"{what} resolves to filesystem root {resolved} -- refusing")
    if resolved == Path.home().resolve():
        raise ManifestValidationError(f"{what} resolves to the home directory {resolved} -- refusing")


def git_commit_sha(repo_dir: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def clone_sparse(
    clone_dir: Path,
    repo_url: str,
    sparse_paths: list[str],
    marker_relpath: str,
    force: bool = False,
) -> tuple[Path, str]:
    """Blob-filtered, sparse-checkout clone of `repo_url` into `clone_dir`,
    materializing only the directories named in `sparse_paths` (each
    rendered as `f"{p}/*"` -- callers must pass exact subtree roots, never
    a pattern broad enough to catch an unrelated sibling directory).
    Reused across runs (idempotent, no re-download) unless `force` is set
    or `clone_dir / marker_relpath` doesn't already exist.

    Returns (clone_dir, commit_sha). Hooks and git-lfs smudge/process are
    disabled at clone time; cloned content is never executed."""
    if not repo_url.startswith("https://"):
        raise PathTraversalError(f"refusing non-https repo_url: {repo_url!r}")

    git_dir = clone_dir / ".git"
    marker = clone_dir / marker_relpath

    if not force and git_dir.is_dir() and marker.exists():
        return clone_dir, git_commit_sha(clone_dir)

    if clone_dir.exists():
        assert_safe_rmtree_root(clone_dir, "--clone-dir")
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
            "--",
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
        "".join(f"{p}/*\n" for p in sparse_paths), encoding="utf-8"
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "sparse-checkout", "reapply"],
        check=True,
    )

    if not marker.exists():
        raise FileNotFoundError(f"sparse checkout did not materialize {marker}")
    return clone_dir, git_commit_sha(clone_dir)


def discover_wav_files(directory: Path, clone_root: Path) -> list[Path]:
    """List `*.wav` files directly under `directory` (non-recursive --
    both upstream sources are flat), sorted for determinism, excluding any
    non-wav entry (e.g. Mycroft's `README.md` sitting alongside the
    clips). Rejects symlinked or otherwise non-regular-file entries --
    `git checkout` cannot itself produce these, but a reused/tampered
    clone dir could."""
    if not directory.is_dir():
        raise ManifestValidationError(f"{directory} does not exist in the clone")

    files: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".wav":
            continue
        if entry.is_symlink() or not entry.is_file():
            raise ManifestValidationError(
                f"{entry}: refusing non-regular-file/symlinked source"
            )
        resolve_under(clone_root, entry)
        files.append(entry)
    return files


def load_waiver_covered_paths(licenses_dir: Path) -> set[str]:
    """Parse every `licenses/license-*.txt` (not the template) under
    `licenses_dir`, returning the set of repo-root-relative paths each
    waiver covers, exactly as written under either header variant this
    repo's contributors have used (docs/WAKEWORD-DATASET-CONTRACT.md
    section 7)."""
    covered: set[str] = set()
    for path in sorted(licenses_dir.glob("license-*.txt")):
        if path.name == "license-template.txt":
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip() in WAIVER_HEADERS:
                header_idx = i
                break
        if header_idx is None:
            continue
        for line in lines[header_idx + 1 :]:
            line = line.strip()
            if line:
                covered.add(line)
    return covered


def verify_mycroft_waiver_coverage(clone_dir: Path, wav_files: list[Path]) -> int:
    """Raise ManifestValidationError unless every file in `wav_files` is
    covered by some waiver in `clone_dir/licenses/`. Returns the number of
    files verified (== len(wav_files) on success) for use in `summary.md`.

    Re-derives the bijection from the live clone every run -- a cached
    "verified N/N" claim from a previous session is not a substitute for
    this, since the upstream tree (and its waiver coverage) can change."""
    licenses_dir = resolve_under(clone_dir, clone_dir / MYCROFT_LICENSES_PATH)
    if not licenses_dir.is_dir():
        raise ManifestValidationError(f"{licenses_dir} missing -- cannot verify waiver coverage")

    covered = load_waiver_covered_paths(licenses_dir)
    uncovered = []
    for wav in wav_files:
        relpath = wav.relative_to(clone_dir).as_posix()
        if relpath not in covered:
            uncovered.append(relpath)

    if uncovered:
        raise ManifestValidationError(
            f"{len(uncovered)} of {len(wav_files)} mycroft {MYCROFT_SPARSE_PATH} files are not "
            f"covered by any waiver in {licenses_dir}: {uncovered[:5]}"
            + (" ..." if len(uncovered) > 5 else "")
        )
    return len(wav_files)


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
    REQUIRED_SR mono 16-bit PCM, overwriting it in place, then re-probe
    the written file to confirm the format actually landed rather than
    trusting the resample call's intent (mirrors
    `vcm/optionb/fetch_dataset.py`'s identically-named helper)."""
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


def build_manifest_rows(
    wav_files: list[Path], clone_dir: Path, out_root: Path, source_dataset: str
) -> tuple[list[dict], list[tuple[Path, Path]]]:
    """Validate + map every discovered wav into a manifest row plus its
    (verified src, verified dest) copy job. Raises on any path that would
    escape `clone_dir` (reads) or `out_root` (writes), or on an unsafe
    filename. Does not touch the filesystem."""
    manifest_rows: list[dict] = []
    copy_jobs: list[tuple[Path, Path]] = []
    seen_dest: dict[Path, str] = {}

    for wav in wav_files:
        filename = sanitize_component(wav.name, "filename")
        group_id = sanitize_component(wav.stem, "filename-stem/group_id")

        src = resolve_under(clone_dir, wav)
        dest = resolve_under(out_root, out_root / "audio" / source_dataset / filename)

        if dest in seen_dest:
            raise ManifestValidationError(
                f"duplicate destination {dest} for {wav} and {seen_dest[dest]!r}"
            )
        seen_dest[dest] = str(wav)

        manifest_rows.append(
            {
                "filename": filename,
                "path": str(dest.relative_to(out_root.resolve())),
                "label": LABEL_WAKEWORD,
                "duration": None,  # filled in after copy, from the written file
                "sample_rate": None,
                "resampled": "False",
                "source_dataset": source_dataset,
                "source_relpath": wav.relative_to(clone_dir).as_posix(),
                "group_id": group_id,
                "split": "",
            }
        )
        copy_jobs.append((src, dest))

    return manifest_rows, copy_jobs


def copy_and_measure(copy_jobs: list[tuple[Path, Path]], manifest_rows: list[dict]) -> int:
    """Copy every (src, dest) pair, verify the written file is mono
    16-bit PCM (raising loudly if not), resample to REQUIRED_SR in place
    if it isn't already, and fill in each row's `duration`/`sample_rate`
    measured from the *final* written file. Returns total bytes copied,
    checked against MAX_TOTAL_BYTES/MAX_SINGLE_FILE_BYTES as it goes."""
    total_bytes = 0
    for (src, dest), row in zip(copy_jobs, manifest_rows):
        if src.is_symlink() or not src.is_file():
            raise ManifestValidationError(f"{src}: refusing non-regular-file/symlinked source")

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
        key = (row["source_dataset"], row["filename"])
        if key in seen:
            raise ManifestValidationError(f"filename {row['filename']!r} referenced by more than one row for {row['source_dataset']!r}")
        seen.add(key)
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
    picovoice_sha: str,
    mycroft_sha: str,
    picovoice_count: int,
    mycroft_count: int,
    mycroft_waiver_verified: int,
    total_bytes: int,
    sample_rate_counts: dict[str, int],
) -> None:
    lines = [
        "# Wakeword positives (real audio) fetch summary",
        "",
        f"Upstream `Picovoice/wake-word-benchmark`, path `{PICOVOICE_SPARSE_PATH}`, commit `{picovoice_sha}`.",
        f"Upstream `MycroftAI/Precise-Community-Data`, path `{MYCROFT_SPARSE_PATH}`, commit `{mycroft_sha}`.",
        "",
        "## Row-count reconciliation",
        "",
        f"- Picovoice `{PICOVOICE_SPARSE_PATH}/`: **{picovoice_count}** files "
        f"(expected exactly {EXPECTED_PICOVOICE_COUNT}).",
        f"- Mycroft `{MYCROFT_SPARSE_PATH}/`: **{mycroft_count}** files "
        f"(expected exactly {EXPECTED_MYCROFT_COUNT}).",
        f"- Total: **{picovoice_count + mycroft_count}** rows.",
        "",
        "## Audio format",
        "",
        "`sample_rate` is measured per file from the final written WAV, never assumed. "
        f"Distribution across all rows: {dict(sorted(sample_rate_counts.items()))}.",
        "",
        "## Disk usage",
        "",
        f"- Total bytes copied: {total_bytes} ({total_bytes / (1024**2):.2f} MiB), "
        f"cap was {MAX_TOTAL_BYTES / (1024**2):.0f} MiB.",
        "",
        "## Scope exclusion (not oversight)",
        "",
        "Mycroft `heycomputer/en/` (10 clips) was deliberately excluded -- this ticket's "
        "`_wakeword_` class covers the single word \"computer\" only, never \"hey computer\" "
        "(see docs/WAKEWORD-DATASET-CONTRACT.md section 1). This is a scope decision, not an "
        "omission; adding \"hey computer\" later is a deliberate follow-up, not a bug fix.",
        "",
        "## License",
        "",
        "- **Picovoice `wake-word-benchmark`:** whole-repo Apache-2.0 (root `LICENSE` file).",
        "- **Mycroft `Precise-Community-Data`:** per-file public-domain waivers, verified "
        f"{mycroft_waiver_verified}/{mycroft_count}, no repo-root LICENSE (GitHub reports "
        "`license: None`). Waiver files under `licenses/license-*.txt` enumerate covered files "
        "as repo-root-relative paths (e.g. `computer/en/<uuid>.wav`); this was re-derived live "
        "against the clone this run, not assumed from a prior session's report.",
        "",
        "Nothing from either source is redistributed -- `out/` is gitignored.",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--picovoice-repo-url", default=PICOVOICE_REPO_URL)
    parser.add_argument("--mycroft-repo-url", default=MYCROFT_REPO_URL)
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "me2_wakeword_clone",
        help="scratch clone parent dir, outside the project repo tree (default: a stable temp "
        "dir, reused across runs so re-running does not re-download)",
    )
    parser.add_argument("--force-clone", action="store_true", help="ignore any existing clones and re-clone")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/conversions/v2/wakeword/positives_real"),
        help="output dir (default: out/conversions/v2/wakeword/positives_real)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing anything")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_root

    try:
        assert_safe_rmtree_root(out_root, "--out-root")
        assert_safe_rmtree_root(args.clone_dir, "--clone-dir")
    except ManifestValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    picovoice_clone_dir = args.clone_dir / "picovoice"
    mycroft_clone_dir = args.clone_dir / "mycroft"

    try:
        picovoice_clone_dir, picovoice_sha = clone_sparse(
            picovoice_clone_dir,
            args.picovoice_repo_url,
            [PICOVOICE_SPARSE_PATH],
            marker_relpath=PICOVOICE_SPARSE_PATH,
            force=args.force_clone,
        )
        mycroft_clone_dir, mycroft_sha = clone_sparse(
            mycroft_clone_dir,
            args.mycroft_repo_url,
            [MYCROFT_SPARSE_PATH, MYCROFT_LICENSES_PATH],
            marker_relpath=MYCROFT_SPARSE_PATH,
            force=args.force_clone,
        )

        picovoice_dir = resolve_under(picovoice_clone_dir, picovoice_clone_dir / PICOVOICE_SPARSE_PATH)
        mycroft_dir = resolve_under(mycroft_clone_dir, mycroft_clone_dir / MYCROFT_SPARSE_PATH)

        picovoice_wavs = discover_wav_files(picovoice_dir, picovoice_clone_dir)
        mycroft_wavs = discover_wav_files(mycroft_dir, mycroft_clone_dir)

        if len(picovoice_wavs) != EXPECTED_PICOVOICE_COUNT:
            raise ManifestValidationError(
                f"picovoice {PICOVOICE_SPARSE_PATH}/ has {len(picovoice_wavs)} files, "
                f"expected exactly {EXPECTED_PICOVOICE_COUNT}"
            )
        if len(mycroft_wavs) != EXPECTED_MYCROFT_COUNT:
            raise ManifestValidationError(
                f"mycroft {MYCROFT_SPARSE_PATH}/ has {len(mycroft_wavs)} files, "
                f"expected exactly {EXPECTED_MYCROFT_COUNT}"
            )

        mycroft_waiver_verified = verify_mycroft_waiver_coverage(mycroft_clone_dir, mycroft_wavs)

        du_bytes = sum(p.stat().st_size for p in picovoice_wavs + mycroft_wavs)
        if du_bytes > MAX_TOTAL_BYTES:
            raise ManifestValidationError(
                f"clone contents are {du_bytes} bytes, exceeding cap {MAX_TOTAL_BYTES} -- "
                "refusing to proceed with conversion"
            )

        staging_root = out_root.parent / f".{out_root.name}.staging"
        picovoice_rows, picovoice_jobs = build_manifest_rows(
            picovoice_wavs, picovoice_clone_dir, staging_root, SOURCE_PICOVOICE
        )
        mycroft_rows, mycroft_jobs = build_manifest_rows(
            mycroft_wavs, mycroft_clone_dir, staging_root, SOURCE_MYCROFT
        )
        all_rows = picovoice_rows + mycroft_rows
        all_jobs = picovoice_jobs + mycroft_jobs

        if len(all_rows) != EXPECTED_TOTAL_ROWS:
            raise ManifestValidationError(
                f"{len(all_rows)} total rows built, expected exactly {EXPECTED_TOTAL_ROWS}"
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
        total_bytes = sum(src.stat().st_size for src, _dest in all_jobs)
        print(
            f"dry-run: would write {len(all_rows)} rows "
            f"({len(picovoice_rows)} picovoice + {len(mycroft_rows)} mycroft), "
            f"{len(all_jobs)} files, {total_bytes / (1024**2):.2f} MiB, to {out_root}"
        )
        return 0

    assert_safe_rmtree_root(staging_root, "staging dir (derived from --out-root)")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    sample_rate_counts: dict[str, int] = {}
    try:
        total_bytes = copy_and_measure(all_jobs, all_rows)
        verify_manifest_files_exist(all_rows, staging_root)

        for row in all_rows:
            sample_rate_counts[row["sample_rate"]] = sample_rate_counts.get(row["sample_rate"], 0) + 1

        write_manifest(staging_root, all_rows)
        write_summary(
            staging_root,
            picovoice_sha=picovoice_sha,
            mycroft_sha=mycroft_sha,
            picovoice_count=len(picovoice_rows),
            mycroft_count=len(mycroft_rows),
            mycroft_waiver_verified=mycroft_waiver_verified,
            total_bytes=total_bytes,
            sample_rate_counts=sample_rate_counts,
        )
    except (ManifestValidationError, RuntimeError, PathTraversalError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    if out_root.exists():
        assert_safe_rmtree_root(out_root, "--out-root")
        shutil.rmtree(out_root)
    staging_root.rename(out_root)

    print(f"{len(all_rows)} rows ({len(picovoice_rows)} picovoice + {len(mycroft_rows)} mycroft) -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
