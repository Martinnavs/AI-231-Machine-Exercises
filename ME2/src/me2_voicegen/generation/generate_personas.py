#!/usr/bin/env python3
"""Backend-agnostic CLI for synthesizing one sample per persona in a manifest,
sharing a single synthesizer construction across the whole batch. No
backend-specific identifier may appear anywhere in this file except the
DEFAULT_BACKEND constant's value - dispatch goes entirely through
synthesis.factory, same contract as generate_sample.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, DEFAULT_TEXT, build_config
from me2_voicegen.generation.cosyvoice_env import OUT_DIR
from me2_voicegen.generation.personas import load_personas
from me2_voicegen.synthesis.base import save_wav
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class, list_backends

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize one speech sample per persona via a swappable TTS backend."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to a persona manifest JSON file (see personas.example.json)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="text to synthesize for every persona (default: a natural carrier sentence)",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=list_backends(),
        help=f"synthesis backend to use (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory to write the synthesized wavs to (default: ME2/out)",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    text = args.text if args.text else DEFAULT_TEXT
    out_dir = args.out_dir or OUT_DIR

    try:
        personas = load_personas(args.manifest)
    except (ValueError, OSError) as exc:
        logger.error("manifest validation failed: %s", exc)
        return 1

    try:
        backend_cls = get_backend_class(args.backend)
        config = build_config(backend_cls, args.backend, args.device, args.opt)
        synthesizer = create_synthesizer(args.backend, **config)
    except Exception:
        logger.exception("failed to construct backend %r", args.backend)
        return 1

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = int(time.time() * 1000)

    batch_start = time.perf_counter()
    outcomes: list[tuple[str, bool]] = []

    for persona in personas:
        try:
            start = time.perf_counter()
            result = synthesizer.synthesize(text, prompt=persona.prompt)
            elapsed = time.perf_counter() - start

            out_path = out_dir / f"sample_{persona.name}_{run_ts}.wav"
            save_wav(result, out_path)

            num_samples = result.audio.shape[-1]
            duration = num_samples / result.sample_rate if result.sample_rate else 0.0
            rtf = elapsed / duration if duration > 0 else float("inf")

            print(f"[{persona.name}] output path: {out_path}")
            print(f"[{persona.name}] duration: {duration:.3f}s")
            print(f"[{persona.name}] sample rate: {result.sample_rate}")
            print(f"[{persona.name}] wall-clock synthesis time: {elapsed:.3f}s")
            print(f"[{persona.name}] RTF: {rtf:.3f}")
            outcomes.append((persona.name, True))
        except Exception:
            logger.exception("synthesis failed for persona %r", persona.name)
            outcomes.append((persona.name, False))

    batch_elapsed = time.perf_counter() - batch_start
    peak_mem_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    )

    print("--- summary ---")
    for name, ok in outcomes:
        print(f"[{name}] {'ok' if ok else 'failed'}")
    print(f"total wall-clock time: {batch_elapsed:.3f}s")
    print(
        "peak GPU memory: "
        + (f"{peak_mem_bytes} bytes" if peak_mem_bytes is not None else "n/a (no CUDA device)")
    )

    return 1 if any(not ok for _, ok in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
