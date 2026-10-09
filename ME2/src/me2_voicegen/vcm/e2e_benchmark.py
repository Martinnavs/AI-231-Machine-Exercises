"""End-to-end (features -> ONNX forward -> log-softmax -> grammar decode)
latency benchmark over real clips, for FP32 and INT8 exports in a run dir.

Read-only against the run dir: it only writes `--out`. Model time and
decode time are timed separately with `time.perf_counter`, decoding at
threshold `-inf` like the streaming runner (policy thresholding is applied
afterward, so it is not part of the cost measured here).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
import time

import numpy as np
import torch

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.benchmark import HARDWARE_LABEL
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.decoder import decode
from me2_voicegen.vcm.evaluate import GRAMMAR_REGISTRY
from me2_voicegen.vcm.streaming.backends import OnnxBackend

VARIANTS = ("fp32", "int8")


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def _fit_window(waveform: torch.Tensor, window_samples: int) -> np.ndarray:
    wav = waveform.numpy().astype(np.float32)
    if wav.shape[0] >= window_samples:
        return wav[:window_samples]
    return np.pad(wav, (0, window_samples - wav.shape[0]))


def run(
    run_dir: Path,
    manifest: Path,
    split: str,
    n_clips: int,
    window_s: float,
    beam_width: int,
    grammar_key: str,
    seed: int,
) -> dict:
    grammar = GRAMMAR_REGISTRY[grammar_key][0]
    dataset = VCMDataset(manifest, split=split)
    target_idx = [i for i, r in enumerate(dataset.rows) if r["bucket"] == "target_commands"]
    rng = random.Random(seed)
    chosen = rng.sample(target_idx, min(n_clips, len(target_idx)))
    window_samples = int(round(window_s * SAMPLE_RATE))
    waves = [_fit_window(dataset._load_waveform(dataset.rows[i]), window_samples) for i in chosen]

    result: dict = {}
    intents: dict[str, list] = {}
    for variant in VARIANTS:
        backend = OnnxBackend(run_dir / "export" / f"vcm_model.{variant}.onnx", ort_threads=1)
        for w in waves[:5]:  # warmup, untimed
            decode(backend.logp_for_waveform(w), grammar, float("-inf"), beam_width=beam_width)
        model_ms, decode_ms, total_ms, won = [], [], [], []
        t_out = 0
        for w in waves:
            t0 = time.perf_counter()
            logp = backend.logp_for_waveform(w)
            t1 = time.perf_counter()
            res = decode(logp, grammar, float("-inf"), beam_width=beam_width)
            t2 = time.perf_counter()
            model_ms.append((t1 - t0) * 1e3)
            decode_ms.append((t2 - t1) * 1e3)
            total_ms.append((t2 - t0) * 1e3)
            won.append(res.intent)
            t_out = int(logp.shape[0])
        intents[variant] = won
        result[variant] = {
            "p50_ms": statistics.median(total_ms),
            "p95_ms": _pct(total_ms, 95),
            "model_p50_ms": statistics.median(model_ms),
            "decode_p50_ms": statistics.median(decode_ms),
            "t_out": t_out,
        }
    agree = sum(a == b for a, b in zip(intents["fp32"], intents["int8"]))
    result["int8_fp32_intent_agreement"] = agree / max(len(waves), 1)
    result["n_clips"] = len(waves)
    result["hardware_label"] = HARDWARE_LABEL
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--window-s", type=float, default=2.5)
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument("--grammar", default="optionb", choices=sorted(GRAMMAR_REGISTRY))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run(
        args.run_dir, args.manifest, args.split, args.n_clips, args.window_s,
        args.beam_width, args.grammar, args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
