"""Score the streaming CLI on wake-word + command replay sessions.

Runs the real entry point (`python -m me2_voicegen.vcm.streaming`) once per
session built by `session_replay`, with the wake-word gate and the policy
under test, and scores the JSONL triggers against the session's label and
its forced-aligned end of speech:

- correct: first trigger has the session's intent;
- wrong_intent: triggers with any other intent (counted per trigger);
- missed: no trigger at all;
- early: first trigger fired before the command's speech ended;
- latency: first-trigger time minus end of speech, correct sessions only.
  This is algorithmic latency (file replay runs in lockstep, no drops);
  add one decode's compute time for wall-clock latency.
- decodes: windows the model actually decoded (`--log-all-windows` records).

    uv run python -m me2_voicegen.vcm.streaming.replay_eval \
        --sessions out/vcm/wakeword-sliding/sessions-val \
        --model out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m --threshold -0.1 \
        --policy endpointed --out out/vcm/wakeword-sliding/val_endpointed.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WAKEWORD = "out/wakeword-sesame-ambient-rir-45m"
FOCUS = ("PAUSE", "STOP", "TIME")
_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"}


def cli_args(cfg: dict) -> list[str]:
    args = [
        "--model", cfg["model"], "--backend", "onnx", "--onnx-variant", "int8", "--ort-threads", "1",
        "--grammar", "optionb", "--beam-width", str(cfg.get("beam_width", 50)),
        f"--threshold={cfg['threshold']}", "--score-mode", cfg.get("score_mode", "mean_frame"),
        "--gate", "wakeword", "--wakeword-model", cfg.get("wakeword_model", DEFAULT_WAKEWORD),
        "--wakeword-backend", "onnx", "--gate-period", str(cfg.get("period_s", 3.0)),
        "--policy", cfg["policy"], "--window-s", "2.5", "--stride-s", "0.25", "--log-all-windows",
    ]
    if cfg.get("wakeword_poll_s") is not None:
        args += ["--wakeword-poll-s", str(cfg["wakeword_poll_s"])]
    if cfg.get("margin") is not None:
        args += ["--required-command-margin", str(cfg["margin"])]
    if cfg["policy"] == "endpointed":
        for flag, key in (("--min-audio", "min_audio_s"), ("--stable-strides", "stable_strides"),
                          ("--hold-ms", "hold_ms"), ("--blank-floor", "blank_floor")):
            if key in cfg:
                args += [flag, str(cfg[key])]
        for flag, key in (("--cls-model", "cls_model"), ("--cls-threshold", "cls_threshold"),
                          ("--cls-hold-ms", "cls_hold_ms"), ("--cls-min-speech-ms", "cls_min_speech_ms")):
            if cfg.get(key) is not None:
                args += [flag, str(cfg[key])]
    return args


def run_session(session: dict, sessions_dir: Path, cfg: dict) -> dict:
    cmd = [sys.executable, "-m", "me2_voicegen.vcm.streaming", *cli_args(cfg),
           "--source", str(sessions_dir / session["file"])]
    p = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_ENV, capture_output=True, text=True)
    records = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
    triggers = [r for r in records if r["event"] == "trigger"]
    return {
        "file": session["file"], "label": session["label"], "speech_end_s": session["speech_end_s"],
        "returncode": p.returncode, "stderr_tail": p.stderr[-400:] if p.returncode else "",
        "decodes": len(records),
        "triggers": [{"t": r["t_seconds"], "intent": r["intent"], "slots": r["slots"],
                      "confidence": r["confidence"]} for r in triggers],
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    first = [r["triggers"][0] if r["triggers"] else None for r in results]
    correct = [r for r, f in zip(results, first) if f and f["intent"] == r["label"]]
    latency = [f["t"] - r["speech_end_s"] for r, f in zip(results, first) if f and f["intent"] == r["label"]]
    pct = lambda q: round(float(np.percentile(latency, q)), 3) if latency else None  # noqa: E731
    per_focus = {}
    for label in FOCUS:
        rs = [(r, f) for r, f in zip(results, first) if r["label"] == label]
        per_focus[label] = [sum(1 for r, f in rs if f and f["intent"] == label), len(rs)]
    return {
        "n": n,
        "correct_first_trigger": len(correct),
        "correct_rate": round(len(correct) / n, 4) if n else None,
        "wrong_intent_triggers": sum(1 for r in results for t in r["triggers"] if t["intent"] != r["label"]),
        "missed": sum(1 for f in first if f is None),
        "extra_triggers": sum(max(0, len(r["triggers"]) - 1) for r in results),
        "early_first_triggers": sum(1 for r, f in zip(results, first) if f and f["t"] < r["speech_end_s"]),
        "latency_s": {"p50": pct(50), "p95": pct(95), "mean": round(float(np.mean(latency)), 3) if latency else None},
        "decodes_per_session_mean": round(float(np.mean([r["decodes"] for r in results])), 2) if n else None,
        "per_focus": per_focus,
        "nonzero_exit": sum(1 for r in results if r["returncode"] != 0),
    }


def evaluate(sessions_dir: Path, cfg: dict, workers: int = 32) -> dict:
    meta = json.loads((sessions_dir / "sessions.json").read_text())
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda s: run_session(s, sessions_dir, cfg), meta["sessions"]))
    return {"config": cfg, "split": meta["split"], "summary": summarize(results), "sessions": results}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--policy", required=True, choices=["single_period", "endpointed"])
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--score-mode", default="mean_frame", choices=["mean_frame", "per_char"])
    p.add_argument("--period-s", type=float, default=3.0)
    p.add_argument("--min-audio", dest="min_audio_s", type=float, default=None)
    p.add_argument("--stable-strides", type=int, default=None)
    p.add_argument("--hold-ms", type=float, default=None)
    p.add_argument("--blank-floor", type=float, default=None)
    p.add_argument("--cls-model", default=None, help="hybrid: heads ONNX/checkpoint for the classifier fallback")
    p.add_argument("--cls-threshold", type=float, default=None)
    p.add_argument("--cls-hold-ms", type=float, default=None)
    p.add_argument("--cls-min-speech-ms", type=float, default=None)
    p.add_argument("--wakeword-model", default=None)
    p.add_argument("--wakeword-poll-s", type=float, default=None)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cfg = {k: v for k, v in {
        "model": a.model, "threshold": a.threshold, "policy": a.policy, "margin": a.margin,
        "score_mode": a.score_mode, "period_s": a.period_s, "min_audio_s": a.min_audio_s,
        "stable_strides": a.stable_strides, "hold_ms": a.hold_ms, "blank_floor": a.blank_floor,
        "cls_model": a.cls_model, "cls_threshold": a.cls_threshold, "cls_hold_ms": a.cls_hold_ms,
        "cls_min_speech_ms": a.cls_min_speech_ms, "wakeword_model": a.wakeword_model, "wakeword_poll_s": a.wakeword_poll_s,
    }.items() if v is not None or k == "margin"}
    report = evaluate(a.sessions, cfg, a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
