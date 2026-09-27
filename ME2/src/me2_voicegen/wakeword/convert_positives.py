"""Convert ticket 01's 468 real "computer" positives into a seeded, K=8
stratified sample of voice-converted variants, one per (source clip,
reference voice) pair, via `generate_conversions.py`'s existing
`synthesizer.convert_voice(raw, ref)` mechanism (discovered by `getattr`
the same way that module does, so a backend without it fails loudly).

Deliberately NOT a full cartesian product (468 positives x 35 reference
voices = 16,380 conversions -- measured-infeasible, see the owning
ticket's `Established` note). `convert` mode (the default) is NOT the
probabilistic zero-shot RESYNTHESIS branch `generate_conversions.py`
also offers: that branch re-types content through Whisper transcription
+ the TTS LLM, which for a single-word wakeword risks the word itself
changing, so `convert` mode uses only the content-preserving
`convert_voice` path and is byte-identical to this module's pre-sesame
behavior. `--mode resynthesize` (the wakeword-sesame ticket's T3; see
`ME2/.scratch/wakeword-sesame/tickets/00-RECAP.md`) reuses this exact
pairing plan but swaps the per-pair audio operation: it zero-shot
synthesizes the fixed `--resynth-text` (default "Sesame.") from each
reference voice, with the reference clip + its Whisper transcript as the
voice prompt. The spoken content is a constant chosen by the operator,
never re-derived from the source clip, which is what removes the
word-drift risk the probabilistic branch had. Why it exists: the
established facts in that ticket show no upstream real-recording
"sesame" corpus exists, and the raw `positives_real` clips (2.0-3.07s)
are too short to be safe zero-shot prompts. In this mode `group_id`
becomes the reference voice (the only speaker-like identity left once
the source clip's audio is discarded -- a shared ref voice must not
straddle train and val/test), the filename becomes
`audio/{ref_voice}/{source_clip_group_id}.wav` (the original real clip's
id becomes a per-job disambiguator + `source_relpath` provenance only),
and `source_dataset` is `cosyvoice_resynth`. This module adapts that
mechanism for K-sampling; it does not modify `generate_conversions.py`
itself.

Output: `out/conversions/v2/wakeword/positives_converted/{audio/,manifest.csv,summary.md}`,
schema = `docs/WAKEWORD-DATASET-CONTRACT.md` section 2's 10 columns plus one
extra `ref_voice` provenance column (a deliberate, documented deviation from
the contract's "exactly this column set" wording for per-subset manifests --
this ticket's own Action explicitly requires the column; flagged for
tech-lead/contract-doc reconciliation rather than silently either adding it
or silently dropping it).
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import shutil
import sys
import time
import wave
from collections import Counter
from pathlib import Path

from me2_voicegen.dataset_tools.transcribe import transcribe_cached
from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, build_config
from me2_voicegen.generation.generate_conversions import (
    _list_audio_files,
    _resolve_prompt_wav,
)
from me2_voicegen.synthesis.base import VoicePrompt, save_wav
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class, list_backends
from me2_voicegen.wakeword.fetch_positives import (
    MANIFEST_FIELDS as POSITIVES_REAL_FIELDS,
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
from me2_voicegen.wakeword.generate_adversaries import DEFAULT_WHISPER_MODEL

logger = logging.getLogger(__name__)

LABEL_WAKEWORD = "_wakeword_"
SOURCE_DATASET = "cosyvoice_conversion"
SOURCE_DATASET_RESYNTH = "cosyvoice_resynth"
DEFAULT_K = 8
DEFAULT_RESYNTH_TEXT = "Sesame."

# Resynthesis-mode seconds/pair, re-derived for resynthesis by this
# module's own timed pilot (the same convention PILOT_SECONDS_PER_PAIR
# follows for convert_voice). Real measurement, NOT the conversion value
# assumed to transfer (T3 of the wakeword-sesame ticket forbids that
# assumption): 20-pair timed pilot, 2026-09-26 21:25, real CosyVoice2
# backend on one A100 (GPU 7), real reference-clip trimming + format
# finalization included, 20 ok / 0 failed, wall-clock 145.891s / 20 pairs
# = 7.295s/pair (log: ME2/.scratch/wakeword-sesame/resynth_pilot.log). The
# pilot also Whisper-transcribed its fresh refs (now sidecar-cached), so
# the full 3,744-pair run should land at or slightly under this rate.
RESYNTH_PILOT_SECONDS_PER_PAIR: float = 7.295

MANIFEST_FIELDS = POSITIVES_REAL_FIELDS + ["ref_voice"]

# Real pilot measurement (this ticket's own execution log: 20/20 real
# conversions against the real CosyVoice2 backend on a single A100, real
# ~3.08s source clips, real reference-clip trimming included) --
# 101.585s / 20 pairs = 5.079s/pair. Close to, not identical to, the
# unrelated 5.3s/file `generate_conversions.py` baseline (measured on
# different/longer clips) -- this constant is independently re-derived,
# not that baseline reused blindly, and happens to land nearby.
PILOT_SECONDS_PER_PAIR = 5.079

_FAMILY_RE = re.compile(r"^([a-z]+)")


def discover_reference_families(refs_dir: Path) -> dict[str, list[Path]]:
    """Group every audio file directly under `refs_dir` by its leading
    lowercase-alpha filename prefix (e.g. `tagalog14.mp3` -> `tagalog`).
    Raises if any file's name doesn't start with a recognizable alpha
    prefix -- this repo's reference-voice pool is 100% `<family><n>.<ext>`
    named, so a non-conforming name signals something unexpected was
    added, not a family to invent silently."""
    families: dict[str, list[Path]] = {}
    for path in _list_audio_files(refs_dir):
        match = _FAMILY_RE.match(path.stem.lower())
        if not match:
            raise ValueError(f"cannot derive a language family from reference filename: {path.name!r}")
        families.setdefault(match.group(1), []).append(path)
    for family in families:
        families[family].sort()
    return families


def stratified_sample_refs(rng: random.Random, families: dict[str, list[Path]], k: int) -> list[Path]:
    """Deterministic stratified sample of `k` reference voices out of
    `families` (family name -> candidate paths), given `rng`.

    Algorithm (fixed, since it is part of this module's seed contract):
      1. Guaranteed draw: one voice sampled from every family that has
         >=1 candidate (families visited in name-sorted order).
      2. Any remaining `k - n_families` draws are allocated across
         families proportionally to each family's ORIGINAL pool size
         (Hamilton/largest-remainder apportionment: floor each family's
         ideal share, then hand out leftover slots one at a time in
         descending-fractional-remainder order, family name ascending as
         the tiebreak), capped by that family's pool remaining after
         step 1; a family that can't absorb its share because its pool is
         exhausted yields its slot to the next family in that same order
         that still has room.
      3. Each family's total allocation is then drawn via `rng.randrange`
         without replacement, families visited in name-sorted order.
    """
    total_available = sum(len(paths) for paths in families.values())
    k = min(k, total_available)
    sorted_families = sorted(families)

    pools = {family: list(paths) for family, paths in families.items()}
    picks: dict[str, list[Path]] = {family: [] for family in sorted_families}

    remaining = k
    for family in sorted_families:
        if remaining <= 0:
            break
        pool = pools[family]
        if not pool:
            continue
        picks[family].append(pool.pop(rng.randrange(len(pool))))
        remaining -= 1

    if remaining > 0:
        original_sizes = {family: len(families[family]) for family in sorted_families}
        total_size = sum(original_sizes.values())
        shares = {family: remaining * original_sizes[family] / total_size for family in sorted_families}
        alloc = {family: int(shares[family]) for family in sorted_families}

        order = sorted(sorted_families, key=lambda f: (-(shares[f] - alloc[f]), f))
        leftover = remaining - sum(alloc.values())
        i = 0
        max_iterations = 10 * max(len(order), 1)
        while leftover > 0 and i < max_iterations:
            family = order[i % len(order)]
            if alloc[family] < len(pools[family]):
                alloc[family] += 1
                leftover -= 1
            i += 1

        for family in sorted_families:
            take = min(alloc[family], len(pools[family]))
            for _ in range(take):
                picks[family].append(pools[family].pop(rng.randrange(len(pools[family]))))

    result: list[Path] = []
    for family in sorted_families:
        result.extend(picks[family])
    return result


def load_positives_manifest(manifest_path: Path) -> list[dict]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ManifestValidationError(f"positives manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ManifestValidationError(f"positives manifest is empty: {manifest_path}")
    required = {"filename", "path", "group_id", "label"}
    missing = required - set(rows[0])
    if missing:
        raise ManifestValidationError(f"positives manifest missing required columns {missing}: {manifest_path}")
    return rows


def build_conversion_jobs(
    positive_rows: list[dict],
    positives_root: Path,
    families: dict[str, list[Path]],
    refs_dir: Path,
    k: int,
    seed: int,
) -> list[dict]:
    """Deterministic (given `seed`) list of per-pair job dicts: one per
    (source positive clip, sampled reference voice). Does not touch the
    filesystem beyond resolving/validating paths."""
    rng = random.Random(seed)
    jobs: list[dict] = []

    for row in positive_rows:
        group_id = sanitize_component(row["group_id"], "group_id")
        source_relpath = row["path"]
        source_path = resolve_under(positives_root, positives_root / source_relpath)
        if not source_path.is_file():
            raise ManifestValidationError(f"positive clip missing on disk: {source_path}")

        sampled_refs = stratified_sample_refs(rng, families, k)
        for ref_path in sampled_refs:
            ref_path = resolve_under(refs_dir, ref_path)
            ref_voice = sanitize_component(ref_path.stem, "ref_voice")
            jobs.append(
                {
                    "group_id": group_id,
                    "source_path": source_path,
                    "source_relpath": source_relpath,
                    "ref_path": ref_path,
                    "ref_voice": ref_voice,
                }
            )

    return jobs


def _finalize_wav_format(path: Path) -> tuple[float, int, bool]:
    """Read back the written wav; force it to REQUIRED_SR/mono/16-bit PCM
    if it isn't already (CosyVoice2 always emits 24kHz, so this branch is
    always taken for this backend's real output -- but written to detect
    format drift generically, not just handle the one known case)."""
    try:
        probe = probe_wav(path)
        matches = (probe.sample_rate, probe.channels, probe.sampwidth) == (
            REQUIRED_SR,
            REQUIRED_CHANNELS,
            REQUIRED_SAMPWIDTH,
        )
    except wave.Error:
        matches = False

    resampled = False
    if not matches:
        resample_to_16k_in_place(path)
        probe = probe_wav(path)
        resampled = True

    return probe.duration, probe.sample_rate, resampled


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ticket 01's real positive 'computer' clips into a seeded, "
            "K=8 stratified sample of voice-converted variants per clip."
        )
    )
    parser.add_argument(
        "--positives-manifest",
        type=Path,
        default=Path("out/conversions/v2/wakeword/positives_real/manifest.csv"),
        help="ticket 01's real positives manifest.csv",
    )
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=Path.home() / "cosy-voice-data" / "References",
        help="reference-voice pool directory (same as generate_conversions.py's --refs-dir)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/conversions/v2/wakeword/positives_converted"),
        help="output dir (default: out/conversions/v2/wakeword/positives_converted)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"reference voices sampled per source clip (default: {DEFAULT_K})")
    parser.add_argument("--seed", type=int, required=True, help="seed for deterministic (source, ref) pairing and ordering")
    parser.add_argument(
        "--mode",
        default="convert",
        choices=["convert", "resynthesize"],
        help=(
            "convert (default): content-preserving convert_voice timbre transfer of the "
            "source clip -- byte-identical to this module's pre-sesame behavior. "
            "resynthesize: zero-shot synthesize --resynth-text from each reference voice "
            "(the wakeword-sesame ticket's T3); group_id becomes the ref voice, filename "
            "audio/{ref_voice}/{source_clip_group_id}.wav, source_dataset cosyvoice_resynth"
        ),
    )
    parser.add_argument(
        "--resynth-text",
        default=DEFAULT_RESYNTH_TEXT,
        help=f"text synthesized per pair in resynthesize mode (default: {DEFAULT_RESYNTH_TEXT!r})",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=(
            "Whisper model for reference-clip transcription, resynthesize mode only "
            f"(default: {DEFAULT_WHISPER_MODEL!r}, matching generate_adversaries.DEFAULT_WHISPER_MODEL)"
        ),
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=list_backends())
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="backend-specific constructor option, repeatable",
    )
    parser.add_argument(
        "--max-conversions",
        type=int,
        default=None,
        help="cap the number of pairs actually converted this invocation (for piloting/chunked "
        "runs); --dry-run always reports the FULL planned pair count regardless of this cap",
    )
    parser.add_argument("--dry-run", action="store_true", help="report planned pairs + projected wall-clock, write nothing, load no model")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


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
    seed: int,
    k: int,
    n_source_clips: int,
    n_planned_pairs: int,
    n_ok: int,
    n_failed: int,
    ref_voice_counts: Counter,
    elapsed_seconds: float,
    mode: str = "convert",
    resynth_text: str | None = None,
) -> None:
    if mode == "resynthesize":
        header = "# Wakeword resynthesized positives (zero-shot TTS, K-sampled) summary"
        mode_line = (
            f"mode=resynthesize: each pair zero-shot synthesizes {resynth_text!r} from its "
            "reference voice (voice prompt = the reference clip + its Whisper transcript); "
            "the source clip's audio is not used -- provenance only. `group_id` is the "
            "reference voice (the only speaker-like identity left once the source audio is "
            "discarded; a shared ref voice must not straddle train and val/test), the "
            "filename is `audio/{ref_voice}/{source_clip_group_id}.wav`, and "
            "`source_dataset` is `cosyvoice_resynth`."
        )
    else:
        header = "# Wakeword converted positives (voice-conversion, K-sampled) summary"
        mode_line = None

    lines = [
        header,
        "",
        f"seed={seed}, k={k}, source clips={n_source_clips}, planned pairs={n_planned_pairs}.",
    ]
    if mode_line is not None:
        lines.append(mode_line)
    lines += [
        f"This run: {n_ok} ok, {n_failed} failed, wall-clock {elapsed_seconds:.1f}s "
        f"({(elapsed_seconds / n_ok) if n_ok else float('nan'):.3f}s/pair realized).",
        "",
        "## Realized per-reference-voice counts (this run)",
        "",
    ]
    for ref_voice, count in sorted(ref_voice_counts.items()):
        lines.append(f"- `{ref_voice}`: {count}")
    lines += [
        "",
        "## Reference-clip consent/licensing",
        "",
        "Acknowledged open provenance gap, not resolved by this ticket: the 35 reference "
        "clips' own consent/licensing basis is not documented anywhere in this repo (see the "
        "project README's own 'Reference-clip consent/licensing' section). Recorded here "
        "honestly, not papered over.",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        positive_rows = load_positives_manifest(args.positives_manifest)
        positives_root = args.positives_manifest.resolve().parent
        families = discover_reference_families(args.refs_dir)
        if not families:
            raise ValueError(f"no reference audio files found under {args.refs_dir}")

        jobs = build_conversion_jobs(
            positive_rows, positives_root, families, args.refs_dir.resolve(), args.k, args.seed
        )
    except (ManifestValidationError, PathTraversalError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ref_voice_planned = Counter(job["ref_voice"] for job in jobs)

    if args.dry_run:
        if args.mode == "convert":
            projected_seconds = len(jobs) * PILOT_SECONDS_PER_PAIR
            print(
                f"dry-run: {len(positive_rows)} source clips x k={args.k} -> {len(jobs)} planned pairs "
                f"to {args.out_root}"
            )
            print(
                f"dry-run: projected wall-clock ~{projected_seconds:.0f}s "
                f"(~{projected_seconds / 3600:.2f}h) at {PILOT_SECONDS_PER_PAIR:.3f}s/pair "
                "(real pilot measurement, see module docstring/summary.md -- not the "
                "generate_conversions.py baseline)"
            )
        else:
            print(
                f"dry-run: {len(positive_rows)} source clips x k={args.k} -> {len(jobs)} planned pairs "
                f"to {args.out_root} (mode=resynthesize, text={args.resynth_text!r})"
            )
            if RESYNTH_PILOT_SECONDS_PER_PAIR is None:
                print(
                    "dry-run: resynthesis wall-clock not projected -- this module has no "
                    "resynthesis seconds/pair measurement yet (the conversion "
                    "PILOT_SECONDS_PER_PAIR is a convert_voice measurement and does NOT "
                    "transfer to synthesize); re-derive it with a small timed "
                    "--max-conversions pilot, per this module's pilot-measurement convention"
                )
            else:
                projected_seconds = len(jobs) * RESYNTH_PILOT_SECONDS_PER_PAIR
                print(
                    f"dry-run: projected wall-clock ~{projected_seconds:.0f}s "
                    f"(~{projected_seconds / 3600:.2f}h) at {RESYNTH_PILOT_SECONDS_PER_PAIR:.3f}s/pair "
                    "(real resynthesis pilot measurement, see module docstring/summary.md)"
                )
        print(f"dry-run: planned per-reference-voice counts: {dict(sorted(ref_voice_planned.items()))}")
        return 0

    backend_cls = get_backend_class(args.backend)
    config = build_config(backend_cls, args.backend, args.device, args.opt)
    synthesizer = create_synthesizer(args.backend, **config)

    if args.mode == "convert":
        convert_voice = getattr(synthesizer, "convert_voice", None)
        if convert_voice is None:
            logger.error("backend %r does not support voice conversion (no convert_voice method)", args.backend)
            return 1
    else:
        synthesize = getattr(synthesizer, "synthesize", None)
        if synthesize is None:
            logger.error("backend %r does not support zero-shot synthesis (no synthesize method)", args.backend)
            return 1

    staging_root = args.out_root.parent / f".{args.out_root.name}.staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    staging_root = staging_root.resolve()

    trimmed_cache_dir = staging_root / "_trimmed_refs"
    effective_ref_paths: dict[Path, Path] = {}

    n_ok = n_failed = 0
    ref_voice_ok = Counter()
    manifest_rows: list[dict] = []

    active_jobs = jobs if args.max_conversions is None else jobs[: args.max_conversions]
    if args.max_conversions is not None:
        logger.info("--max-conversions=%d: converting %d of %d planned pairs", args.max_conversions, len(active_jobs), len(jobs))

    if args.mode == "resynthesize":
        planned_dests = [f"audio/{j['ref_voice']}/{j['group_id']}.wav" for j in active_jobs]
        if len(set(planned_dests)) != len(planned_dests):
            dupes = sorted({d for d in set(planned_dests) if planned_dests.count(d) > 1})
            print(
                f"error: resynthesize filename-scheme collision on {len(dupes)} planned job(s), "
                f"e.g. {dupes[0]!r} -- refusing to run (a collision would silently overwrite a pair)",
                file=sys.stderr,
            )
            shutil.rmtree(staging_root, ignore_errors=True)
            return 1

    start = time.perf_counter()
    ref_transcripts: dict[Path, str] = {}
    for i, job in enumerate(active_jobs):
        pair_label = f"[{i + 1}/{len(active_jobs)}] {job['group_id']!r} x {job['ref_voice']!r}"
        try:
            ref_path = job["ref_path"]
            if ref_path not in effective_ref_paths:
                effective_ref_paths[ref_path] = _resolve_prompt_wav(ref_path, trimmed_cache_dir)
            effective_ref_path = effective_ref_paths[ref_path]

            if args.mode == "convert":
                dest = resolve_under(
                    staging_root,
                    staging_root / "audio" / job["group_id"] / f"{job['ref_voice']}.wav",
                )
                result = convert_voice(str(job["source_path"]), str(effective_ref_path))
                row_group_id = job["group_id"]
                row_source_dataset = SOURCE_DATASET
            else:
                if effective_ref_path not in ref_transcripts:
                    ref_transcripts[effective_ref_path] = transcribe_cached(effective_ref_path, args.whisper_model)
                dest = resolve_under(
                    staging_root,
                    staging_root / "audio" / job["ref_voice"] / f"{job['group_id']}.wav",
                )
                result = synthesize(
                    args.resynth_text,
                    prompt=VoicePrompt(wav_path=effective_ref_path, text=ref_transcripts[effective_ref_path]),
                )
                row_group_id = job["ref_voice"]
                row_source_dataset = SOURCE_DATASET_RESYNTH

            save_wav(result, dest)
            duration, sample_rate, resampled = _finalize_wav_format(dest)

            manifest_rows.append(
                {
                    "filename": dest.name,
                    "path": str(dest.relative_to(staging_root)),
                    "label": LABEL_WAKEWORD,
                    "duration": f"{duration:.6f}",
                    "sample_rate": str(sample_rate),
                    "resampled": "True" if resampled else "False",
                    "source_dataset": row_source_dataset,
                    "source_relpath": job["source_relpath"],
                    "group_id": row_group_id,
                    "split": "",
                    "ref_voice": job["ref_voice"],
                }
            )
            n_ok += 1
            ref_voice_ok[job["ref_voice"]] += 1
            logger.info(
                "%s: %s -> %s",
                pair_label,
                "converted" if args.mode == "convert" else "resynthesized",
                dest.name,
            )
        except Exception:
            n_failed += 1
            logger.exception(
                "%s: %s failed",
                pair_label,
                "voice conversion" if args.mode == "convert" else "resynthesis",
            )
    elapsed = time.perf_counter() - start

    try:
        write_manifest(staging_root, manifest_rows)
        write_summary(
            staging_root,
            seed=args.seed,
            k=args.k,
            n_source_clips=len(positive_rows),
            n_planned_pairs=len(jobs),
            n_ok=n_ok,
            n_failed=n_failed,
            ref_voice_counts=ref_voice_ok,
            elapsed_seconds=elapsed,
            mode=args.mode,
            resynth_text=args.resynth_text,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    staging_root.rename(args.out_root)

    print(
        f"{'conversions' if args.mode == 'convert' else 'resyntheses'}: {n_ok} ok, "
        f"{n_failed} failed (of {len(active_jobs)} attempted, {len(jobs)} planned)"
    )
    print(f"output dir: {args.out_root}")
    print(f"wall-clock: {elapsed:.3f}s")

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
