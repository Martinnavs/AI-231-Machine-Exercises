"""Run the VCM benchmark ON the Pi against the ME2 solution, by file replay instead of speaker -> microphone.

Everything that defines the test comes from benchmark.py / vcmbench (holdout download, trial building with a wake word
take + gap + command, shuffle seed, false-wake trials, event-to-trial assignment, scoring, report). Only the transport is
different: the trials are laid end to end in one wav (10-15 s apart, like the live gaps), streamed once through
me2_voicegen.vcm.streaming, and each trigger line becomes a log event at its stream time.

    python replay_bench.py build  --run runs/replay --seed 1 --size full
    python replay_bench.py stream --run runs/replay          # slow: ~real time on a Pi 4
    python replay_bench.py score  --run runs/replay
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark as B                                   # noqa: E402
from vcmbench import audio as A                         # noqa: E402
from vcmbench import schema as S                        # noqa: E402
from vcmbench.pi import Event                           # noqa: E402
from vcmbench.report import score, write_outputs        # noqa: E402

ME2 = HERE.parent / "ME2"
SOAK = ME2 / "soak/holdout-wake-gap-v1"
LEAD_S, TAIL_S = 1.0, 4.0
NOISE_DBFS = -60.0
STREAM_ARGS = dict(backend="onnx", threads=1, beam_width=50, gate="wakeword", gate_period=3.0, stride_s=0.25,
                   wakeword_threshold=0.8, cls_slot_threshold=0.6, cls_threshold=0.8787, poll_s=0.05, no_cls=False,
                   device="cpu", models=str(ME2 / "out/vcm/hybrid-ctcwide-clsxl"), wakeword=str(ME2 / "out/wakeword-sesame-ambient-rir-45m"))


def wake_takes(run: Path, n: int = 3) -> list[str]:
    """Wake-word takes ('sesame') cut from the soak recording (distinct speakers)."""
    meta = json.loads((SOAK / "continuous.json").read_text())
    wav = A.load(SOAK / "continuous.wav")
    seen, paths = set(), []
    for s in meta["sessions"]:
        if s["wakeword_filename"] in seen or s["wake_end_s"] < 0.5:
            continue
        seen.add(s["wakeword_filename"])
        a = int(s["offset_s"] * A.SR)
        p = run / "wake_src" / f"{len(paths) + 1}.wav"
        A.save(p, wav[a: a + int(s["wake_end_s"] * A.SR)])
        paths.append(str(p))
        if len(paths) == n:
            break
    return paths


def build(a) -> None:
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    cfg = {"mode": "replay", "student": "ME2 sesame solution (file replay on the Pi)", "started": time.strftime("%Y%m%d-%H%M%S"),
           "id_order": "manifest"}
    B.set_id_order(cfg)
    ns = SimpleNamespace(wake_word="sesame", wake_files=wake_takes(run), holdout=None, size=a.size, seed=a.seed, limit=a.limit,
                         wake_gap=0.8, gap_min=10.0, gap_max=15.0, wake_takes=None, mic=None, speaker=None)
    word, takes = B.record_wake(ns, cfg, run)
    trials = B.build_trials(ns, cfg, run, takes)
    rng, noise_rng = random.Random(a.seed + 1), np.random.default_rng(a.seed)
    amp = 10 ** (NOISE_DBFS / 20)
    pieces, pos = [np.zeros(int(LEAD_S * A.SR), np.float32)], LEAD_S
    for t in trials:
        x = A.load(run / t["audio_file"])
        gap = rng.uniform(10.0, 15.0)                      # same spacing rule as run_trials: after the command ends
        t["offset"] = pos
        t["cmd_end_stream"] = pos + t["cmd_end"]
        t["window_end"] = pos + t["cmd_end"] + gap
        keep = int(round((t["cmd_end"] + gap) * A.SR))
        seg = np.zeros(keep, np.float32)
        seg[: min(len(x), keep)] = x[:keep]
        pieces.append(seg)
        pos += keep / A.SR
    pieces.append(np.zeros(int(TAIL_S * A.SR), np.float32))
    audio = np.concatenate(pieces)
    audio = audio + (noise_rng.standard_normal(len(audio)) * amp).astype(np.float32)
    A.save(run / "stream.wav", audio)
    (run / "plan.json").write_text(json.dumps(trials, indent=2))
    cfg.update(seed=a.seed, size=a.size, wake_word=word, wake_gap=0.8)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"stream.wav: {len(audio) / A.SR / 60:.1f} min, {len(trials)} trials")


def stream(a) -> None:
    run = Path(a.run)
    sys.path.insert(0, str(ME2 / "scripts"))
    import soak_run
    argv = soak_run.cli_args(SimpleNamespace(**STREAM_ARGS)) + ["--source", str((run / "stream.wav").resolve())]
    samples, specs = [], {}
    agent = subprocess.Popen([sys.executable, str(HERE / "pi_agent.py"), "--proc", r"me2_voicegen\.vcm\.streaming", "--interval", "1"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def reader():
        for line in agent.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("type") == "specs":
                specs.update(m)
            elif m.get("type") == "metrics":
                samples.append(m)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    time.sleep(2)
    t0 = time.time()
    env = {**__import__("os").environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    p = subprocess.run([str(ME2 / ".venv/bin/python"), "-m", "me2_voicegen.vcm.streaming", *argv], cwd=ME2, env=env,
                       capture_output=True, text=True)
    t1 = time.time()
    agent.terminate()
    th.join(timeout=3)
    (run / "stream_raw.jsonl").write_text("\n".join(l for l in p.stdout.splitlines() if l.startswith("{")))
    (run / "stream_stderr.txt").write_text(p.stderr[-4000:])
    (run / "pi_specs.json").write_text(json.dumps(specs, indent=2))
    (run / "pi_samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
    (run / "stream_wall.json").write_text(json.dumps({"t0": t0, "t1": t1, "returncode": p.returncode, "argv": argv}))
    print(f"streamed in {(t1 - t0) / 60:.1f} min, exit {p.returncode}")


def slot_text(slots: dict) -> str:
    return " ".join(str(v) for v in (slots or {}).values())


def do_score(a) -> None:
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    trials = json.loads((run / "plan.json").read_text())
    wall = json.loads((run / "stream_wall.json").read_text())
    t0 = wall["t0"]
    recs = [json.loads(l) for l in (run / "stream_raw.jsonl").read_text().splitlines() if l]
    B.set_id_order(cfg)
    aliases = S.build_alias_table(cfg.get("aliases"))
    win_ms = 2500.0                                     # --window-s of the streaming runner: audio the model sees per decision
    events = [Event("command", t0 + r["t_seconds"], r["intent"], slot_text(r.get("slots")), r["gate_ms"] + r["decode_ms"], win_ms,
                    json.dumps({k: r[k] for k in ("intent", "slots", "text", "t_seconds")}))
              for r in recs if r["event"] == "trigger" and r.get("intent")]
    for t in trials:
        t["play_t0"] = t0 + t["offset"]
        t["cmd_end_abs"] = t0 + t["cmd_end_stream"]
        until = t0 + t["window_end"]
        evs = [e for e in events if t["play_t0"] - 0.2 <= e.t < until]
        t["n_command_events"] = len(evs)
        t["wake_logged"] = False
        if evs:
            e = evs[0]
            intent, slot, _known, variation = S.resolve_prediction(e.intent, e.slot, "", aliases, B.ID_ORDER or S.id_orders()["manifest"])
            t.update(pred_intent=intent, pred_slot=slot, pred_variation=variation, pred_variation_id=None, pred_raw=e.raw,
                     infer_ms=e.infer_ms, audio_ms=e.audio_ms, latency_s=e.t - t["cmd_end_abs"])
        else:
            t.update(pred_intent=S.NONE, pred_slot="", pred_variation="", pred_variation_id=None, pred_raw="",
                     infer_ms=None, audio_ms=None, latency_s=None)
    (run / "trials.jsonl").write_text("\n".join(json.dumps(t, default=str) for t in trials))
    stray = [e for e in events if not any(t["play_t0"] - 0.2 <= e.t < t0 + t["window_end"] for t in trials)]
    samples = [json.loads(l) for l in (run / "pi_samples.jsonl").read_text().splitlines() if l]
    specs = json.loads((run / "pi_specs.json").read_text())
    meta = {k: cfg.get(k) for k in ("student", "started", "mode", "wake_word", "wake_gap", "size", "seed", "id_order")}
    meta.update(id_orders=B.current_id_orders(cfg), holdout="huggingface", gap_s=[10.0, 15.0], pi_log_file="(file replay: streaming stdout)")
    m = score(trials, samples, specs, None, t0, wall["t1"], meta)
    m["replay"] = {"stray_triggers_outside_any_trial": len(stray), "wall_s": wall["t1"] - t0}
    report = write_outputs(run, m, trials, samples)
    print(report.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stream", "score"])
    ap.add_argument("--run", default=str(HERE / "runs/replay"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", default="full")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"build": build, "stream": stream, "score": do_score}[a.cmd](a)
