"""Run the streaming pipeline over soak sessions (`scripts/build_soak_audio.py`) and score accuracy and latency.

One streaming process per session (wake word gate -> endpointed policy, optional classifier fallback), all windows logged with
wall-clock timing. Works on any machine that has the repo, the model files and the sessions directory, so the same audio can be
replayed on a Raspberry Pi: copy `sessions.json` + `audio/`, run the same command there (`--workers 1` for clean timings).

    # A100, PyTorch fp32 (hybrid; CTC-only with --no-cls)
    uv run python scripts/soak_run.py --sessions out/soak/holdout-wake-gap-v1 --name a100-hybrid \
        --backend torch --device cuda:0 --gpu 5 --workers 6
    # CPU, INT8 ONNX: what the Pi runs
    uv run python scripts/soak_run.py --sessions out/soak/holdout-wake-gap-v1 --name cpu-onnx-int8-hybrid --backend onnx --workers 8

Scoring (per session): a command session is correct if the FIRST trigger has the right intent and slot; missed if there is no trigger;
wrong-intent triggers are counted per trigger; an out-of-scope session must produce no trigger (any trigger is a false accept). Latency =
first trigger time minus the end of the command's speech (algorithmic, file replay runs in lockstep) and the wall-clock compute of every
decoded window (`gate_ms`: wake-word scoring, `decode_ms`: encoder + beam search + policy).
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "out/vcm/hybrid-ctcwide-clsxl"
DEFAULT_WAKEWORD = ROOT / "out/wakeword-sesame-ambient-rir-45m"


def cli_args(a: argparse.Namespace) -> list[str]:
    models = Path(a.models)
    args = ["--model", str(models / "ctc-wide"), "--backend", a.backend, "--grammar", "optionb", "--beam-width", "50",
            "--threshold=-0.1", "--required-command-margin", "4.0", "--gate", "wakeword", "--gate-period", str(a.gate_period), "--wakeword-threshold", str(a.wakeword_threshold),
            "--policy", "endpointed", "--hold-ms", "200", "--stable-strides", "1", "--window-s", "2.5", "--stride-s", str(a.stride_s),
            "--wakeword-model", str(a.wakeword), "--wakeword-backend", a.backend, "--log-all-windows", "--log-timing"]
    if a.backend == "onnx":
        args += ["--onnx-variant", "int8", "--ort-threads", str(a.threads)]
    else:
        args += ["--device", a.device]
    if a.poll_s:
        args += ["--wakeword-poll-s", str(a.poll_s)]
    if not a.no_cls:
        cls = models / "cls-xl" / ("export/vcm_heads.int8.onnx" if a.backend == "onnx" else "checkpoints/checkpoint.pt")
        args += ["--cls-model", str(cls), "--cls-threshold", str(a.cls_threshold), "--cls-slot-threshold", str(a.cls_slot_threshold)]
    return args


def run_session(s: dict, a: argparse.Namespace, raw_dir: Path) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads)}
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    cmd = [sys.executable, "-m", "me2_voicegen.vcm.streaming", *cli_args(a), "--source", str(Path(a.sessions) / s["file"])]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    (raw_dir / (Path(s["file"]).stem + ".jsonl")).write_text("\n".join(lines))
    recs = [json.loads(ln) for ln in lines]
    return {**s, "returncode": p.returncode, "stderr_tail": p.stderr[-300:] if p.returncode else "", "records": recs}


def run_continuous(a: argparse.Namespace, raw_dir: Path) -> list[dict]:
    """Stream `continuous.wav` once and give each session the records whose time falls in its range (times made session-relative)."""
    meta = json.loads((a.sessions / "continuous.json").read_text())
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads)}
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    cmd = [sys.executable, "-m", "me2_voicegen.vcm.streaming", *cli_args(a), "--source", str(a.sessions / "continuous.wav")]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    (raw_dir / "continuous.jsonl").write_text("\n".join(lines))
    recs = [json.loads(ln) for ln in lines]
    out = []
    for s in meta["sessions"][: a.limit]:
        lo, hi = s["offset_s"], s["range_end_s"]
        mine = [{**x, "t_seconds": x["t_seconds"] - lo} for x in recs if lo <= x["t_seconds"] < hi]
        out.append({**s, "returncode": p.returncode, "stderr_tail": p.stderr[-300:] if p.returncode else "", "records": mine})
    return out


def slot_ok(trigger: dict, slot_value: str) -> bool:
    return not slot_value or slot_value.lower() in [str(v).lower() for v in (trigger.get("slots") or {}).values()]


def score(results: list[dict], stride_s: float = 0.25) -> dict:
    cmds = [r for r in results if not r["is_oos"]]
    oos = [r for r in results if r["is_oos"]]
    first = lambda r: next((x for x in r["records"] if x["event"] == "trigger"), None)  # noqa: E731
    triggers = lambda r: [x for x in r["records"] if x["event"] == "trigger"]  # noqa: E731
    correct, wrong, missed, early, lat_ok, lat_all, no_period, by_cls = 0, 0, 0, 0, [], [], 0, 0
    wrong_action = 0
    gap_bins = {"0.0-0.3": [0, 0], "0.4-0.6": [0, 0], "0.7-1.0": [0, 0]}
    for r in cmds:
        f = first(r)
        g = r["gap_s"]
        b = "0.0-0.3" if g <= 0.3 else "0.4-0.6" if g <= 0.6 else "0.7-1.0"
        gap_bins[b][1] += 1
        if not r["records"]:
            no_period += 1
        if f is None:
            missed += 1
            continue
        lat = f["t_seconds"] - r["speech_end_s"]
        lat_all.append(lat)
        wrong_action += not (f["intent"] == r["label"] and slot_ok(f, r["slot_value"]))
        early += f["t_seconds"] < r["speech_end_s"]
        if f["intent"] == r["label"] and slot_ok(f, r["slot_value"]):
            correct += 1
            gap_bins[b][0] += 1
            lat_ok.append(lat)
            by_cls += f.get("text") == "[classifier]"
    gap_trig = sum(1 for r in results if "unit_end_s" in r for x in triggers(r) if x["t_seconds"] > r["unit_end_s"] + 1.5)
    wrong_triggers = sum(1 for r in cmds for x in triggers(r) if x["intent"] != r["label"])
    pct = lambda v, q: round(float(np.percentile(v, q)), 3) if len(v) else None  # noqa: E731
    dec = [x["decode_ms"] for r in results for x in r["records"] if "decode_ms" in x]
    gate = [x["gate_ms"] for r in results for x in r["records"] if "gate_ms" in x]
    tot = [x["decode_ms"] + x["gate_ms"] for r in results for x in r["records"] if "decode_ms" in x]
    ms = lambda v: {"n": len(v), "mean": round(float(np.mean(v)), 2), "p50": round(float(np.percentile(v, 50)), 2),
                    "p95": round(float(np.percentile(v, 95)), 2), "max": round(float(np.max(v)), 2)} if len(v) else None  # noqa: E731
    return {
        "commands": len(cmds), "correct_first_trigger": correct, "correct_rate": round(correct / len(cmds), 4),
        "missed": missed, "missed_no_period_opened": no_period, "wrong_intent_triggers": wrong_triggers,
        "wrong_action_first_triggers": wrong_action,   # the first trigger has the wrong intent OR the wrong slot
        "early_first_triggers": int(early), "answered_by_classifier": int(by_cls),
        "oos_sessions": len(oos), "oos_false_accepts": sum(1 for r in oos if first(r) is not None),
        "objective": correct - 2 * (wrong_action + sum(1 for r in oos if first(r) is not None) + gap_trig),
        "extra_triggers": sum(max(0, len(triggers(r)) - 1) for r in results),
        "triggers_in_ambient_gaps": gap_trig,
        "correct_rate_by_gap_s": {k: [v[0], v[1]] for k, v in gap_bins.items()},
        "latency_after_speech_end_s": {"correct_p50": pct(lat_ok, 50), "correct_p95": pct(lat_ok, 95), "correct_mean":
                                       round(float(np.mean(lat_ok)), 3) if lat_ok else None, "all_triggered_p50": pct(lat_all, 50)},
        "decode_ms": ms(dec), "gate_ms": ms(gate), "gate_plus_decode_ms": ms(tot),
        "rtf_decoded_windows_p95": round(float(np.percentile(tot, 95)) / (1000 * float(stride_s)), 3) if tot else None,
        "nonzero_exit": sum(1 for r in results if r["returncode"] != 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--name", required=True, help="results are written to <sessions>/results/<name>.{json,md} and raw/<name>/")
    ap.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--wakeword", type=Path, default=DEFAULT_WAKEWORD)
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--device", default="cpu", help="torch backend: cpu or cuda:0 (with --gpu to pick the physical card)")
    ap.add_argument("--gpu", type=int, default=None, help="sets CUDA_VISIBLE_DEVICES for each run (never 6 on the shared node)")
    ap.add_argument("--no-cls", action="store_true", help="CTC only (no classifier fallback)")
    ap.add_argument("--cls-threshold", type=float, default=0.8787)
    ap.add_argument("--cls-slot-threshold", type=float, default=0.0, help="slotted intents from the classifier also need this slot-head confidence")
    ap.add_argument("--gate-period", type=float, default=3.0, help="seconds the wake word keeps the period open (live setting: 3)")
    ap.add_argument("--wakeword-threshold", type=float, default=0.9)
    ap.add_argument("--poll-s", type=float, default=0.05, help="wake-word polling step; 0 = once per stride")
    ap.add_argument("--stride-s", type=float, default=0.25, help="decode stride (the live setting is 0.25 s)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--continuous", action="store_true", help="stream continuous.wav once (scripts/build_soak_continuous.py) instead of one process per session")
    a = ap.parse_args()
    if a.gpu == 6:
        raise SystemExit("GPU 6 is off limits on this node")
    meta = json.loads((a.sessions / "sessions.json").read_text())
    sessions = meta["sessions"][: a.limit]
    raw_dir = a.sessions / "raw" / a.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    (a.sessions / "results").mkdir(exist_ok=True)
    if a.continuous:
        results = run_continuous(a, raw_dir)
    else:
        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(lambda s: run_session(s, a, raw_dir), sessions))
    summary = score(results, a.stride_s)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    slim = [{k: v for k, v in r.items() if k != "records"} | {"triggers": [x for x in r["records"] if x["event"] == "trigger"],
            "windows": len(r["records"])} for r in results]
    (a.sessions / "results" / f"{a.name}.json").write_text(json.dumps({"config": config, "summary": summary, "sessions": slim}, indent=1))
    lat, d, g = summary["latency_after_speech_end_s"], summary["decode_ms"], summary["gate_ms"]
    md = [f"# Soak results: {a.name}", "", f"`{json.dumps(config)}`", "",
          f"- commands {summary['commands']}: **{summary['correct_first_trigger']} correct first trigger ({100 * summary['correct_rate']:.1f}%)**, "
          f"{summary['missed']} missed ({summary['missed_no_period_opened']} never opened a period), "
          f"{summary['wrong_action_first_triggers']} wrong-action first triggers (intent or slot), {summary['wrong_intent_triggers']} wrong-intent triggers in all, {summary['early_first_triggers']} early first triggers, "
          f"{summary['answered_by_classifier']} answered by the classifier",
          f"- triggers in the ambient gaps (more than 1.5 s after a unit ended): {summary['triggers_in_ambient_gaps']}",
          f"- out-of-scope sessions {summary['oos_sessions']}: {summary['oos_false_accepts']} false accepts",
          f"- correct by gap: " + ", ".join(f"{k} s {v[0]}/{v[1]}" for k, v in summary["correct_rate_by_gap_s"].items()),
          f"- latency after end of speech (correct): p50 {lat['correct_p50']} s, p95 {lat['correct_p95']} s",
          f"- decode per window: " + (f"mean {d['mean']} ms, p50 {d['p50']}, p95 {d['p95']}, max {d['max']} ms ({d['n']} windows)" if d else "n/a"),
          f"- wake-word gate per decoded window: " + (f"mean {g['mean']} ms, p95 {g['p95']} ms" if g else "n/a"),
          f"- RTF (p95 gate+decode over the stride): {summary['rtf_decoded_windows_p95']}"]
    (a.sessions / "results" / f"{a.name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
