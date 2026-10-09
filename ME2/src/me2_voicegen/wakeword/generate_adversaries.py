"""Generate phonetic near-miss TTS adversaries of "computer" for the
`_unknown_` class (docs/WAKEWORD-DATASET-CONTRACT.md sections 2-3), across
this project's existing reference-voice pool (`$HOME/cosy-voice-data/References`
by default -- same pool `generate_conversions.py` already drives).

Dispatch is direct through `synthesis.factory` rather than
`generation/generate_personas.py`'s manifest-driven persona flow:
`generate_personas.py` requires a hand-authored `personas.json` naming each
reference clip's transcript up front, which doesn't fit "drive this from a
raw directory of reference clips" without bending that script to a shape it
wasn't built for. `generate_conversions.py` already solved the same problem
(discover reference clips straight from a directory, transcribe any clip
lacking a `<stem>.txt` sidecar via Whisper, drive `synthesizer.synthesize`
directly) -- this module follows that precedent instead.

No backend-specific identifier may appear anywhere in this file except
DEFAULT_BACKEND's value (see cli_common.py and
tests/test_generate_sample_cli.py's source-scan test for the convention
this follows) -- dispatch goes entirely through synthesis.factory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from me2_voicegen.dataset_tools.transcribe import transcribe_cached
from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, build_config
from me2_voicegen.synthesis.base import VoicePrompt, save_wav
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class, list_backends
from me2_voicegen.wakeword.fetch_positives import (
    MANIFEST_FIELDS,
    REQUIRED_CHANNELS,
    REQUIRED_SAMPWIDTH,
    REQUIRED_SR,
    ManifestValidationError,
    PathTraversalError,
    probe_wav,
    resample_to_16k_in_place,
    resolve_under,
    sanitize_component,
)

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

LABEL_UNKNOWN = "_unknown_"
SOURCE_DATASET = "adversaries_tts"

DEFAULT_REFS_DIR = Path.home() / "cosy-voice-data" / "References"
DEFAULT_WHISPER_MODEL = "base"

# The vendored TTS backend's speech-tokenizer path hard-asserts prompt
# audio <= 30s -- confirmed live this session (a real reference clip in
# this pool exceeds 30s and fails outright without this). Mirrors
# generate_conversions.py's identically-valued MAX_PROMPT_SECONDS/
# _resolve_prompt_wav.
MAX_PROMPT_SECONDS = 25.0

# Mycroft's stated phonemes for "computer" (see this feature's Established
# facts): K AH M P Y UW T ER. Every phrase below is chosen to overlap that
# sequence in onset, stress pattern, or syllable count/coda -- none of them
# may ever be "hey computer" or contain a "hey" token (see
# docs/WAKEWORD-DATASET-CONTRACT.md section 1; guarded by a unit test).
ADVERSARY_PHRASES: tuple[tuple[str, str], ...] = (
    (
        "commuter",
        "K AH M Y UW T ER -- drops the medial P only; a near-homophone off "
        "a single missing stop consonant.",
    ),
    (
        "computing",
        "K AH M P Y UW T IH NG -- shares the onset K AH M P Y UW verbatim "
        "with \"computer\", diverging only in the final syllable.",
    ),
    (
        "compute her",
        "K AH M P Y UW T ER -- the identical phoneme sequence as "
        "\"computer\", split across a word boundary; tests whether the "
        "model keys on phonemes rather than a single-word boundary.",
    ),
    (
        "come here",
        "K AH M . HH IY R -- shares the onset K AH M and the two-syllable, "
        "stress-initial cadence of \"computer\"; a common everyday "
        "confusable despite the differing coda.",
    ),
    (
        "confuser",
        "K AH N F Y UW Z ER -- matches \"computer\"'s syllable count, "
        "stress placement (on the Y UW), and ER onset+coda envelope.",
    ),
    (
        "put her",
        "P UH T ER -- rhymes with the T ER tail of \"computer\"; isolates "
        "whether the model overweights the word-final syllable alone.",
    ),
    (
        "computer science",
        "K AH M P Y UW T ER . S AY AH N S -- embeds the full \"computer\" "
        "phoneme sequence as a strict prefix, testing whether trailing "
        "speech after the wakeword still triggers a false positive.",
    ),
)


def _phrase_slug(phrase: str) -> str:
    return phrase.lower().replace(" ", "_")


def resolve_prompt_wav(path: Path, cache_dir: Path) -> Path:
    """Returns path unchanged if it's within MAX_PROMPT_SECONDS, else a
    cached, trimmed-to-MAX_PROMPT_SECONDS copy under cache_dir (reused
    across runs)."""
    info = sf.info(str(path))
    if info.duration <= MAX_PROMPT_SECONDS:
        return path

    cache_path = cache_dir / f"{path.stem}.trimmed.wav"
    if not cache_path.is_file():
        data, sr = sf.read(str(path), dtype="float32")
        max_frames = int(MAX_PROMPT_SECONDS * sr)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(cache_path), data[:max_frames], sr)
        logger.info(
            "trimmed %s (%.1fs) to %.1fs -> %s",
            path.name,
            info.duration,
            MAX_PROMPT_SECONDS,
            cache_path.name,
        )
    return cache_path


def list_reference_voices(refs_dir: Path) -> list[Path]:
    if not refs_dir.is_dir():
        raise ValueError(f"not a directory: {refs_dir}")
    files = sorted(
        p for p in refs_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no reference-voice audio files found in {refs_dir}")
    return files


def select_reference_voices(
    ref_files: list[Path], max_voices_per_phrase: int | None, seed: int
) -> list[Path]:
    """Full pool (sorted, so the default run is deterministic with no seed
    needed) unless `max_voices_per_phrase` bounds it below the pool size, in
    which case a seeded `random.Random(seed).sample` makes the reduced
    selection itself reproducible run-to-run."""
    if max_voices_per_phrase is None or max_voices_per_phrase >= len(ref_files):
        return ref_files
    if max_voices_per_phrase < 1:
        raise ValueError(f"max_voices_per_phrase must be >= 1, got {max_voices_per_phrase}")
    rng = random.Random(seed)
    return sorted(rng.sample(ref_files, max_voices_per_phrase), key=lambda p: p.name)


@dataclass(frozen=True)
class GenerationJob:
    phrase: str
    ref_path: Path
    filename: str
    group_id: str
    source_relpath: str


def build_jobs(
    phrases: tuple[tuple[str, str], ...], ref_files: list[Path], refs_dir: Path
) -> list[GenerationJob]:
    """Planning only -- no filesystem writes, no model load, no
    transcription. Safe to call from --dry-run."""
    jobs: list[GenerationJob] = []
    seen_filenames: set[str] = set()

    for phrase, _justification in phrases:
        slug = sanitize_component(_phrase_slug(phrase), "adversary phrase slug")
        for ref_path in ref_files:
            resolve_under(refs_dir, ref_path)
            ref_slug = sanitize_component(ref_path.stem, "reference-voice filename stem")
            filename = f"{slug}__{ref_slug}.wav"
            if filename in seen_filenames:
                raise ManifestValidationError(f"duplicate generated filename {filename!r}")
            seen_filenames.add(filename)

            jobs.append(
                GenerationJob(
                    phrase=phrase,
                    ref_path=ref_path,
                    filename=filename,
                    group_id=phrase,
                    source_relpath=ref_path.relative_to(refs_dir).as_posix(),
                )
            )
    return jobs


def run_job(
    job: GenerationJob,
    synthesizer,
    effective_ref_path: Path,
    ref_transcript: str,
    staging_root: Path,
) -> dict:
    """Synthesize one job, write it under staging_root, verify its final
    on-disk format by reading the file back (resampling in place if the
    backend's native sample rate isn't already REQUIRED_SR), and return the
    manifest row. Raises on any format-verification failure -- a written
    file is never assumed correct because synthesize() didn't raise.

    `effective_ref_path` is `job.ref_path` unless it needed trimming
    (resolve_prompt_wav) -- `job.ref_path`/`job.source_relpath` still name
    the real reference clip for provenance regardless."""
    prompt = VoicePrompt(wav_path=effective_ref_path, text=ref_transcript)
    result = synthesizer.synthesize(job.phrase, prompt=prompt)

    dest = resolve_under(
        staging_root, staging_root / "audio" / SOURCE_DATASET / job.filename
    )
    save_wav(result, dest)

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

    return {
        "filename": job.filename,
        "path": str(dest.relative_to(staging_root.resolve())),
        "label": LABEL_UNKNOWN,
        "duration": f"{probe.duration:.6f}",
        "sample_rate": str(probe.sample_rate),
        "resampled": "True" if resampled else "False",
        "source_dataset": SOURCE_DATASET,
        "source_relpath": job.source_relpath,
        "group_id": job.group_id,
        "split": "",
    }


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
    refs_dir: Path,
    backend: str,
    whisper_model: str,
    seed: int,
    max_voices_per_phrase: int | None,
    ok: int,
    failed: int,
    sample_rate_counts: dict[str, int],
) -> None:
    lines = [
        "# Wakeword phonetic adversaries (`_unknown_`) generation summary",
        "",
        f"Reference-voice pool: `{refs_dir}`.",
        f"Backend: `{backend}`. Whisper model (reference-clip transcription): `{whisper_model}`.",
        f"Seed: {seed}. Max voices per phrase: "
        f"{max_voices_per_phrase if max_voices_per_phrase is not None else 'all (full pool)'}.",
        "",
        "## Phrase list (phonetic justification against \"computer\" = K AH M P Y UW T ER)",
        "",
    ]
    for phrase, justification in ADVERSARY_PHRASES:
        lines.append(f"- **{phrase!r}**: {justification}")
    lines += [
        "",
        "## Results",
        "",
        f"- Synthesized OK: **{ok}**",
        f"- Failed (logged, run continued): **{failed}**",
        "",
        "## Audio format",
        "",
        "`sample_rate` is measured per file from the final written WAV, never "
        f"assumed. Distribution across all rows: {dict(sorted(sample_rate_counts.items()))}.",
        "",
        "## License",
        "",
        "In-repo TTS output (no upstream redistribution) -- same basis as this "
        "repo's other synthetic/converted audio; no new dependency or license "
        "restriction introduced. `out/` is gitignored.",
        "",
    ]
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate phonetic near-miss TTS adversaries of \"computer\" for "
            "the _unknown_ class, across a reference-voice pool."
        )
    )
    parser.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/conversions/v2/wakeword/adversaries"),
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=list_backends())
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="backend-specific constructor option, repeatable (see generate_sample.py --opt)",
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument(
        "--max-voices-per-phrase",
        type=int,
        default=None,
        help="cap the reference-voice pool per phrase (default: use the full pool)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for reproducible voice selection when --max-voices-per-phrase is set",
    )
    parser.add_argument("--dry-run", action="store_true", help="report planned count without loading the model")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug-level logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    out_root: Path = args.out_root
    refs_dir: Path = args.refs_dir

    try:
        ref_files = list_reference_voices(refs_dir)
        selected = select_reference_voices(ref_files, args.max_voices_per_phrase, args.seed)
        jobs = build_jobs(ADVERSARY_PHRASES, selected, refs_dir)
    except (ValueError, ManifestValidationError, PathTraversalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"dry-run: would synthesize {len(jobs)} clips "
            f"({len(ADVERSARY_PHRASES)} phrases x {len(selected)} reference voices) "
            f"to {out_root} (model not loaded)"
        )
        return 0

    backend_cls = get_backend_class(args.backend)
    config = build_config(backend_cls, args.backend, args.device, args.opt)
    synthesizer = create_synthesizer(args.backend, **config)

    staging_root = out_root.parent / f".{out_root.name}.staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    trimmed_cache_dir = out_root.parent / "_trimmed_refs"
    effective_ref_paths: dict[Path, Path] = {}
    ref_transcripts: dict[Path, str] = {}
    rows: list[dict] = []
    ok = failed = 0

    for job in jobs:
        try:
            if job.ref_path not in effective_ref_paths:
                effective_ref_paths[job.ref_path] = resolve_prompt_wav(job.ref_path, trimmed_cache_dir)
            effective_ref_path = effective_ref_paths[job.ref_path]
            if effective_ref_path not in ref_transcripts:
                ref_transcripts[effective_ref_path] = transcribe_cached(effective_ref_path, args.whisper_model)
            row = run_job(
                job, synthesizer, effective_ref_path, ref_transcripts[effective_ref_path], staging_root
            )
            rows.append(row)
            ok += 1
            logger.info("[%d/%d] %r x %r -> %s", ok + failed, len(jobs), job.phrase, job.ref_path.name, row["filename"])
        except Exception:
            failed += 1
            logger.exception("[%d/%d] %r x %r: generation failed", ok + failed, len(jobs), job.phrase, job.ref_path.name)

    if ok == 0:
        print("error: every generation job failed, refusing to write an empty subset", file=sys.stderr)
        shutil.rmtree(staging_root, ignore_errors=True)
        return 1

    sample_rate_counts: dict[str, int] = {}
    for row in rows:
        sample_rate_counts[row["sample_rate"]] = sample_rate_counts.get(row["sample_rate"], 0) + 1

    write_manifest(staging_root, rows)
    write_summary(
        staging_root,
        refs_dir=refs_dir,
        backend=args.backend,
        whisper_model=args.whisper_model,
        seed=args.seed,
        max_voices_per_phrase=args.max_voices_per_phrase,
        ok=ok,
        failed=failed,
        sample_rate_counts=sample_rate_counts,
    )

    if out_root.exists():
        shutil.rmtree(out_root)
    staging_root.rename(out_root)

    print(f"{ok} rows ok, {failed} failed (of {len(jobs)} planned) -> {out_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
