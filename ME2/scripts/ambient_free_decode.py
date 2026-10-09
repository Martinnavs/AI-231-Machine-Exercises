#!/usr/bin/env python3
"""Batch free-decode for the ambient-noise-overlay `_unknown_` gate
(feature-engineering/ambient-noise-overlay/SPEC.md, stage 2).

Single process over every wav in --in-dir (the transcriber's model is
loaded once, not per file -- the transcriber repo's own `transcribe.py` is
single-file and would pay the model-load cost per clip), writing
`filename,decoded_text` rows: the decisions csv `mix_ambient_noise
--stage finalize --negatives-decisions` consumes.

Deliberately standalone: it must run under ~/simple-audio-transcriber's
own venv (scripts/ambient_noise.sh cds into that repo before invoking it,
the accent_balance_fil50.sh convention), so it imports nothing from
me2_voicegen.

Usage (from the transcriber repo, its venv):
    python <me2>/scripts/ambient_free_decode.py --in-dir DIR --out-csv CSV \
        [--backend faster-whisper] [--model small] [--device cuda] [--device-index 0]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from audio_transcript_parser.transcription.factory import create_transcriber

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, required=True, help="dir of wavs to free-decode")
    parser.add_argument("--out-csv", type=Path, required=True, help="decisions csv to write (filename,decoded_text)")
    parser.add_argument("--backend", default="faster-whisper")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--device-index", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES, not the physical GPU")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    wavs = sorted(args.in_dir.glob("*.wav"))
    if not wavs:
        print(f"error: no wav files in {args.in_dir}", file=sys.stderr)
        return 1

    transcriber = create_transcriber(
        args.backend,
        model_size_or_path=args.model,
        device=args.device,
        device_index=args.device_index,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "decoded_text"])
        writer.writeheader()
        for i, wav in enumerate(wavs, start=1):
            result = transcriber.transcribe(wav)
            text = " ".join(seg.text.strip() for seg in result.segments if seg.text.strip())
            writer.writerow({"filename": wav.name, "decoded_text": text})
            if i % 100 == 0 or i == len(wavs):
                logger.info("decoded %d/%d", i, len(wavs))

    print(f"decoded {len(wavs)} wavs -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
