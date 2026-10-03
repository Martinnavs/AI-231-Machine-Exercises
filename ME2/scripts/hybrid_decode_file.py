"""Decode wav files with the hybrid: wide CTC + xl classifier heads, both resident (`vcm.hybrid.HybridDecoder`).

  uv run python scripts/hybrid_decode_file.py clip.wav [more.wav ...] \\
      --ctc-checkpoint out/vcm/hybrid-ctcwide-clsxl/ctc-wide/checkpoints/checkpoint.pt \\
      --cls-checkpoint out/vcm/hybrid-ctcwide-clsxl/cls-xl/checkpoints/checkpoint.pt
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import torchaudio

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.hybrid import HybridDecoder
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.pipeline import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--ctc-checkpoint", type=Path, required=True)
    ap.add_argument("--cls-checkpoint", type=Path, required=True)
    ap.add_argument("--cls-threshold", type=float, default=0.8787, help="val-fit threshold of the xl heads (docs/AI231-FIL50.md)")
    a = ap.parse_args()
    ctc, _ = load_checkpoint(a.ctc_checkpoint, device="cpu", weights_only=True)
    cls, _ = load_checkpoint(a.cls_checkpoint, device="cpu", weights_only=True)
    hd = HybridDecoder(ctc, cls, LogMelFeatureExtractor(), OPTIONB_GRAMMAR, cls_threshold=a.cls_threshold, ctc_name="wide", cls_name="xl")
    for w in a.wavs:
        wav, sr = torchaudio.load(str(w))
        wav = wav.mean(0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        dec, trace = hd.decode(wav)
        print(json.dumps({"file": str(w), "intent": dec.intent, "slots": dec.slots, **trace}))
    print("stats", hd.stats)


if __name__ == "__main__":
    main()
