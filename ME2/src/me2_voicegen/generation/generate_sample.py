#!/usr/bin/env python3
"""Thin, backend-agnostic CLI for synthesizing a sample wav.

No backend-specific identifier may appear anywhere in this file except the
DEFAULT_BACKEND constant's value - dispatch goes entirely through
synthesis.factory, proven end-to-end in tests/test_generate_sample_cli.py via a
FakeSynthesizer with a deliberately unrelated constructor signature.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, DEFAULT_TEXT
from me2_voicegen.generation.cli_common import build_config as _cli_build_config
from me2_voicegen.generation.cli_common import coerce_opt_value as _coerce_opt_value
from me2_voicegen.generation.cli_common import accepted_param_names as _accepted_param_names
from me2_voicegen.generation.cli_common import parse_opts as _parse_opts
from me2_voicegen.generation.cosyvoice_env import OUT_DIR
from me2_voicegen.synthesis.base import VoicePrompt, save_wav
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class, list_backends

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize a speech sample via a swappable TTS backend."
    )
    parser.add_argument(
        "--text",
        default=None,
        help="text to synthesize (default: a natural carrier sentence)",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=list_backends(),
        help=f"synthesis backend to use (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--prompt-wav",
        type=Path,
        default=None,
        help=(
            "path to a reference clip for voice-cloning backends (default: the "
            "vendored sample asset). Silently ignored if the selected --backend "
            "doesn't use a prompt."
        ),
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        help=(
            "transcript of --prompt-wav (default: the vendored sample's "
            "transcript, only used when --prompt-wav is also left at its "
            "default). Silently ignored if the selected --backend doesn't use it."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory to write the synthesized wav to (default: ME2/out)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help=(
            "device hint forwarded to the backend's constructor if it accepts "
            "one (default: auto). Silently ignored otherwise."
        ),
    )
    parser.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "backend-specific constructor option, repeatable (e.g. "
            "--opt model_dir=/path/to/model --opt fp16=true). Passed straight "
            "through to the backend's constructor - an unknown key raises "
            "immediately, unlike the common flags above. Values are coerced: "
            "true/false -> bool, none -> None, else int/float/str."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug-level logging"
    )
    return parser.parse_args(argv)


def _build_config(backend_cls: type, args: argparse.Namespace) -> dict[str, object]:
    return _cli_build_config(backend_cls, args.backend, args.device, args.opt)


def _build_prompt(backend_cls: type, args: argparse.Namespace) -> VoicePrompt | None:
    """A backend may declare its own DEFAULT_PROMPT_WAV/DEFAULT_PROMPT_TEXT class
    attributes (mirroring the reference's optional model_label convention) - this
    keeps any backend-specific default asset out of this file entirely. A backend
    that declares neither, with no --prompt-wav given either, gets prompt=None;
    if it actually requires a prompt, its own synthesize() raises a clear error."""
    default_wav = getattr(backend_cls, "DEFAULT_PROMPT_WAV", None)
    default_text = getattr(backend_cls, "DEFAULT_PROMPT_TEXT", None)

    wav_path = args.prompt_wav if args.prompt_wav is not None else default_wav
    if args.prompt_text is not None:
        text = args.prompt_text
    elif args.prompt_wav is None:
        text = default_text
    else:
        # a custom --prompt-wav with no --prompt-text: don't guess a transcript
        text = None

    if wav_path is None:
        return None
    return VoicePrompt(wav_path=wav_path, text=text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    text = args.text if args.text else DEFAULT_TEXT
    out_dir = args.out_dir or OUT_DIR

    try:
        backend_cls = get_backend_class(args.backend)
        prompt = _build_prompt(backend_cls, args)
        config = _build_config(backend_cls, args)
        synthesizer = create_synthesizer(args.backend, **config)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        result = synthesizer.synthesize(text, prompt=prompt)
        elapsed = time.perf_counter() - start

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sample_{int(time.time() * 1000)}.wav"
        save_wav(result, out_path)

        num_samples = result.audio.shape[-1]
        duration = num_samples / result.sample_rate if result.sample_rate else 0.0
        rtf = elapsed / duration if duration > 0 else float("inf")
        peak_mem_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )

        print(f"output path: {out_path}")
        print(f"duration: {duration:.3f}s")
        print(f"sample rate: {result.sample_rate}")
        print(f"wall-clock synthesis time: {elapsed:.3f}s")
        print(f"RTF: {rtf:.3f}")
        print(
            "peak GPU memory: "
            + (f"{peak_mem_bytes} bytes" if peak_mem_bytes is not None else "n/a (no CUDA device)")
        )
    except Exception:
        logger.exception("synthesis failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
