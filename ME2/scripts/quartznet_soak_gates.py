#!/usr/bin/env python
"""QuartzNet cascade soak gates (quartznet-promotion ticket 03), mirroring
scripts/dense_pilot_gates.py stages P3 / P3b / P4 with the QuartzNet-5x3-tiny
stride-2 run dir and ticket 02's chosen operating point (mean_frame,
threshold -0.1, margin 20.0). Replays the identical soak audio
(raw_datasets/ambient-noise/) and the same 821 composed test clips the
optiond dense pilot used, so every count is directly comparable to the
optiond baselines in docs/archive/CASCADE-SOAK-TEST.md / docs/archive/DENSE-SCORING-DECISION.md.

Optiond baselines on this exact audio/clips (reference only):
  P3  cascade `period: ACCEPT` total: 0 (vacuous -- wakeword never opened)
  P3b VCM-only `trigger` total:       303 (B: mean_frame -0.1, margin 4.0)
  P4  correct-first-trigger:          720/821, wrong-intent 111
        PAUSE 87/? STOP 129/? TIME 86/? (per-focus in dense_gates_stream_summary_B.json)

Stages: p3 | p3b | p4 | summary.  Threads are pinned to 1 (torch/ORT
oversubscribe otherwise). File order honors the ticket: the short COFFEE SHOP
file first, then the three ~1 h files, <= 2 concurrent.
"""
import argparse, json, os, re, subprocess, sys, wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m"
CLIP_SRC = ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot/cascade_stream"
SOAK = ROOT / MODEL / "soak"
AMBIENT = ROOT / "raw_datasets" / "ambient-noise"
CFG = dict(threshold=-0.1, margin=20.0, mode="mean_frame")
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"}
BASE_CMD = [sys.executable, "-m", "me2_voicegen.vcm.streaming", "--model", MODEL,
            "--backend", "onnx", "--onnx-variant", "int8", "--ort-threads", "1",
            "--grammar", "optionb", "--beam-width", "50"]
VCM_ARGS = [f"--threshold={CFG['threshold']}", "--score-mode", CFG["mode"],
            "--required-command-margin", str(CFG["margin"])]
DROP_RE = re.compile(r"dropped=(\d+)")


def vcm_args():
    return list(VCM_ARGS)


def run(cmd, out, err, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, env=ENV, stdout=out, stderr=err, text=True)


def dropped_of(log_text):
    m = [int(x) for x in DROP_RE.findall(log_text)]
    return m[-1] if m else None


def ordered_files():
    files = sorted(AMBIENT.glob("*.wav"))
    short = [f for f in files if f.name.startswith("COFFEE SHOP AMBIENCE")]
    rest = [f for f in files if f not in short]
    return short + rest


def hours_of(files):
    total = 0
    for f in files:
        with wave.open(str(f), "rb") as w:
            total += w.getnframes() / w.getframerate()
    return total / 3600.0


def p3(files):
    out_dir = SOAK / "p3"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(f):
        tag = f"{files.index(f):02d}_{f.name[:40].replace('/', '_')}"
        out, err = out_dir / f"{tag}.jsonl", out_dir / f"{tag}.log"
        with open(out, "w") as o, open(err, "w") as e:
            p = run(BASE_CMD + vcm_args() + ["--gate", "wakeword", "--policy", "single_period",
                    "--gate-period", "3", "--wakeword-model", "out/wakeword-sesame-ambient-rir-45m",
                    "--wakeword-backend", "onnx", "--log-periods", "--source", str(f)], out=o, err=e)
        text = err.read_text(errors="replace")
        return {"file": f.name, "returncode": p.returncode,
                "accepts": text.count("period: ACCEPT"),
                "rejects": text.count("period: REJECT"),
                "opens": text.count("gate: open") + text.count("gate: reopened"),
                "dropped": dropped_of(text), "log": err.name}

    results = []
    for f in files[:1]:
        results.append(one(f))
    with ThreadPoolExecutor(2) as ex:
        results += list(ex.map(one, files[1:]))
    summ = {"config": f"quartznet {CFG}", "stage": "p3", "files": results,
            "accepts_total": sum(r["accepts"] for r in results),
            "rejects_total": sum(r["rejects"] for r in results),
            "opens_total": sum(r["opens"] for r in results),
            "dropped_max": max((r["dropped"] for r in results if r["dropped"] is not None), default=None),
            "nonzero_exit": sum(r["returncode"] != 0 for r in results),
            "audio_hours": round(hours_of(files), 3)}
    (SOAK / "p3_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def p3b(files):
    out_dir = SOAK / "p3b"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(f):
        tag = f"{files.index(f):02d}_{f.name[:40].replace('/', '_')}"
        out, err = out_dir / f"{tag}.jsonl", out_dir / f"{tag}.log"
        with open(out, "w") as o, open(err, "w") as e:
            p = run(BASE_CMD + vcm_args() + ["--gate", "none", "--policy", "threshold",
                    "--refractory-s", "1.5", "--window-s", "2.5", "--stride-s", "0.25",
                    "--source", str(f)], out=o, err=e)
        trig = [json.loads(l) for l in out.read_text().splitlines() if l.strip().startswith("{") and '"trigger"' in l]
        return {"file": f.name, "returncode": p.returncode, "triggers": len(trig),
                "intents": sorted({t["intent"] for t in trig if t.get("intent")}),
                "dropped": dropped_of(err.read_text(errors="replace")), "log": err.name}

    results = []
    for f in files[:1]:
        results.append(one(f))
    with ThreadPoolExecutor(2) as ex:
        results += list(ex.map(one, files[1:]))
    summ = {"config": f"quartznet {CFG}", "stage": "p3b", "files": results,
            "triggers_total": sum(r["triggers"] for r in results),
            "dropped_max": max((r["dropped"] for r in results if r["dropped"] is not None), default=None),
            "nonzero_exit": sum(r["returncode"] != 0 for r in results),
            "audio_hours": round(hours_of(files), 3)}
    (SOAK / "p3b_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def p4(workers=64):
    manifest = json.loads((CLIP_SRC / "clips.json").read_text())
    out_dir = SOAK / "p4"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(m):
        out, err = out_dir / f"{Path(m['file']).stem}.jsonl", out_dir / f"{Path(m['file']).stem}.log"
        with open(out, "w") as o, open(err, "w") as e:
            p = run(BASE_CMD + vcm_args() + ["--gate", "none", "--policy", "threshold",
                    "--refractory-s", "1.5", "--window-s", "2.5", "--stride-s", "0.25",
                    "--source", str(CLIP_SRC / "clips" / m["file"])], out=o, err=e)
        trig = [json.loads(l) for l in out.read_text().splitlines() if l.strip().startswith("{") and '"trigger"' in l]
        return {**m, "returncode": p.returncode, "intents": [t["intent"] for t in trig]}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, manifest))
    n = len(res)
    correct = sum(1 for r in res if r["intents"] and r["intents"][0] == r["label"])
    wrong = sum(1 for r in res for i in r["intents"] if i != r["label"])
    per = {}
    for lab in ("PAUSE", "STOP", "TIME"):
        rs = [r for r in res if r["label"] == lab]
        per[lab] = [sum(1 for r in rs if r["intents"] and r["intents"][0] == lab), len(rs)]
    summ = {"config": f"quartznet {CFG}", "stage": "p4", "n": n,
            "correct_first_trigger": correct, "wrong_intent_triggers": wrong,
            "per_focus": per, "nonzero_exit": sum(r["returncode"] != 0 for r in res)}
    (SOAK / "p4_summary.json").write_text(json.dumps(summ, indent=1))
    (SOAK / "p4_detail.json").write_text(json.dumps(res))
    print(json.dumps(summ, indent=1))


def summary():
    p3s = json.loads((SOAK / "p3_summary.json").read_text())
    p3bs = json.loads((SOAK / "p3b_summary.json").read_text())
    p4s = json.loads((SOAK / "p4_summary.json").read_text())
    optiond = {"p3_accepts": 0, "p3b_triggers": 303, "p4_correct": 720, "p4_wrong": 111, "p4_n": 821}
    out = {"quartznet": {"p3": p3s, "p3b": p3bs, "p4": p4s}, "optiond_baseline": optiond,
           "promotion_bar": {
               "p3b_le_optiond": p3bs["triggers_total"] <= optiond["p3b_triggers"],
               "p3_zero_accepts": p3s["accepts_total"] == 0,
               "p4_recall_within_1pp": (p4s["correct_first_trigger"] / p4s["n"]) >= (optiond["p4_correct"] / optiond["p4_n"]) - 0.01,
               "dropped_zero": (p3s["dropped_max"] in (0, None) and p3bs["dropped_max"] in (0, None)),
           }}
    (SOAK / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("p3", "p3b", "p4", "summary"))
    a = ap.parse_args()
    if a.stage in ("p3", "p3b"):
        files = ordered_files()
        (SOAK).mkdir(parents=True, exist_ok=True)
        {"p3": p3, "p3b": p3b}[a.stage](files)
    elif a.stage == "p4":
        p4()
    else:
        summary()
