#!/usr/bin/env python3
"""Cartesian batch: every Raw_Audio clip re-rendered in every Reference clip's
voice via CosyVoice2 voice conversion (content preserved exactly, no
transcript needed), plus - with configurable probability per pair - an
additional zero-shot resynthesis of the same pair using Whisper-transcribed
text (content re-typed by the TTS LLM rather than converted verbatim).

No backend-specific identifier may appear anywhere in this file except the
DEFAULT_BACKEND constant's value (see cli_common.py and
tests/test_generate_sample_cli.py's source-scan test for the convention this
follows) - dispatch goes through synthesis.factory, and the voice-conversion
capability is discovered via getattr() the same way a backend's own
DEFAULT_PROMPT_WAV is, so a backend without convert_voice fails loudly rather
than silently skipping conversion.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, build_config
from me2_voicegen.generation.cosyvoice_env import OUT_DIR
from me2_voicegen.synthesis.base import VoicePrompt, save_wav
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class, list_backends
from me2_voicegen.dataset_tools.transcribe import transcribe_cached

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
DEFAULT_RESYNTH_PROB = 0.3
DEFAULT_WHISPER_MODEL = "base"

# CosyVoice2's speech-tokenizer path hard-asserts prompt audio <= 30s
# (vendor/CosyVoice/cosyvoice/cli/frontend.py's _extract_speech_token). Trim a
# cached copy of any longer reference clip rather than let every pair using it
# fail outright.
MAX_PROMPT_SECONDS = 25.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cartesian voice-conversion batch: every file in --raw-dir "
            "converted into every voice in --refs-dir, plus probabilistic "
            "zero-shot resynthesis of the same pairs."
        )
    )
    parser.add_argument("--raw-dir", type=Path, required=True, help="directory of source clips to convert")
    parser.add_argument("--refs-dir", type=Path, required=True, help="directory of reference-voice clips")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory to write output wavs to (default: ME2/out/conversions)",
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=list_backends())
    parser.add_argument(
        "--resynth-prob",
        type=float,
        default=DEFAULT_RESYNTH_PROB,
        help=(
            "probability, per (raw, reference) pair, of ALSO producing a "
            f"zero-shot resynthesis alongside the voice conversion (default: {DEFAULT_RESYNTH_PROB})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the resynthesis-probability RNG (default: nondeterministic)",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=(
            "Whisper model size used to transcribe raw/reference clips for the "
            f"resynthesis branch (default: {DEFAULT_WHISPER_MODEL}); cached as a "
            "<clip>.txt sidecar next to each audio file, so repeat runs don't "
            "re-transcribe"
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="backend-specific constructor option, repeatable (see generate_sample.py --opt)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug-level logging")
    return parser.parse_args(argv)


def _resolve_prompt_wav(path: Path, cache_dir: Path) -> Path:
    """Returns path unchanged if it's within MAX_PROMPT_SECONDS, else a cached,
    trimmed-to-MAX_PROMPT_SECONDS copy under cache_dir (reused across runs)."""
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


def _list_audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no audio files ({sorted(AUDIO_EXTENSIONS)}) found in {directory}")
    return files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        raw_files = _list_audio_files(args.raw_dir)
        ref_files = _list_audio_files(args.refs_dir)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    out_dir = args.out_dir or (OUT_DIR / "conversions")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_cls = get_backend_class(args.backend)
    config = build_config(backend_cls, args.backend, args.device, args.opt)
    synthesizer = create_synthesizer(args.backend, **config)

    convert_voice = getattr(synthesizer, "convert_voice", None)
    if convert_voice is None:
        logger.error("backend %r does not support voice conversion (no convert_voice method)", args.backend)
        return 1

    rng = random.Random(args.seed)
    trimmed_cache_dir = out_dir / "_trimmed_refs"
    effective_ref_paths = {
        ref_path: _resolve_prompt_wav(ref_path, trimmed_cache_dir) for ref_path in ref_files
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    n_pairs = len(raw_files) * len(ref_files)
    logger.info(
        "converting %d raw clip(s) x %d reference(s) = %d pairs (resynth_prob=%.2f)",
        len(raw_files),
        len(ref_files),
        n_pairs,
        args.resynth_prob,
    )

    conversions_ok = conversions_failed = 0
    resynths_attempted = resynths_ok = resynths_failed = 0

    for i, raw_path in enumerate(raw_files):
        for j, ref_path in enumerate(ref_files):
            pair_label = f"[{i * len(ref_files) + j + 1}/{n_pairs}] {raw_path.name!r} x {ref_path.name!r}"
            effective_ref_path = effective_ref_paths[ref_path]

            try:
                result = convert_voice(str(raw_path), str(effective_ref_path))
                out_path = out_dir / f"{raw_path.stem} - {ref_path.stem}.wav"
                save_wav(result, out_path)
                conversions_ok += 1
                logger.info("%s: converted -> %s", pair_label, out_path.name)
            except Exception:
                conversions_failed += 1
                logger.exception("%s: voice conversion failed", pair_label)

            if rng.random() >= args.resynth_prob:
                continue

            resynths_attempted += 1
            try:
                text = transcribe_cached(raw_path, args.whisper_model)
                prompt_text = transcribe_cached(effective_ref_path, args.whisper_model)
                result = synthesizer.synthesize(
                    text, prompt=VoicePrompt(wav_path=effective_ref_path, text=prompt_text)
                )
                out_path = out_dir / f"{raw_path.stem} - {ref_path.stem} (synthesized).wav"
                save_wav(result, out_path)
                resynths_ok += 1
                logger.info("%s: resynthesized -> %s", pair_label, out_path.name)
            except Exception:
                resynths_failed += 1
                logger.exception("%s: zero-shot resynthesis failed", pair_label)

    elapsed = time.perf_counter() - start
    peak_mem_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    print(f"conversions: {conversions_ok} ok, {conversions_failed} failed (of {n_pairs} pairs)")
    print(
        f"resynthesis: {resynths_ok} ok, {resynths_failed} failed "
        f"(attempted {resynths_attempted} of {n_pairs} pairs at p={args.resynth_prob:.2f})"
    )
    print(f"output dir: {out_dir}")
    print(f"total wall-clock time: {elapsed:.3f}s")
    print(
        "peak GPU memory: "
        + (f"{peak_mem_bytes} bytes" if peak_mem_bytes is not None else "n/a (no CUDA device)")
    )

    return 1 if (conversions_failed or resynths_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
