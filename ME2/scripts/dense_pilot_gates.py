#!/usr/bin/env python
"""Task-5 validation gates for D2 per_char (dense-d2-loose-impl), run SEQUENTIALLY for the
production baseline B, then the conservative candidate C1, then the current candidate C2.

  B  = mean_frame, threshold -0.1, margin 4.0   (production; reference only)
  C1 = per_char,   threshold -1.1,  no margin   (conservative; = the real harness's val-Youden pick)
  C2 = per_char,   threshold -1.204, no margin  (current; val FA-budget rule)
C1 was fixed BEFORE this run from val-only information (harness Youden); C2 was fixed earlier.
Disclosure: the earlier loosening curve printed test numbers near -1.1, so C1 is not test-blind.

Pre-registered rule (frozen before any P1-P4 number is computed). Candidate PASSES iff ALL hold vs B (P1, P2, P3, P3b, P4):
  P1 offline strict-per-cell (val AND test, clean AND noisy, same records as the pilot): reject-probe
     FA <= B (test); exact-correct >= B; every intent with >= 50 targets has exact >= B - 3pp
     PER CELL (strict; no pooling).
  P2 ghost FA (141 independent clips) <= B's ghost FA.
  P3 ambient soak, REAL wakeword->VCM cascade (streaming CLI, 4 raw_datasets/ambient-noise
     recordings, INT8 ONNX, single_period, gate-period 3): `period: ACCEPT` count <= B's on the same audio.
  P3b (ADDED 2026-09-30 after B's cascade soak showed P3 is vacuous -- 0 wakeword periods on all 4 ambient
     files, so the VCM never ran; amended BEFORE any candidate number existed): VCM-only streaming soak
     (gate none, threshold policy, same 4 files, INT8 ONNX): `trigger` event count <= B's. This skips the
     wakeword filter, so it upper-bounds compound false actions with real power (~3.5 h of babble).
  P4 streaming VCM recall, gate none, threshold policy, same INT8 ONNX, on composed clips
     [1.0s silence][test command clip][1.5s silence]; ALL test PAUSE/STOP/TIME clips + first 30 of each
     other intent: per-intent recall for PAUSE, STOP, TIME >= B - 3pp each; overall recall >= B - 0.5pp;
     wrong-intent triggers <= B's count.
Stages: offline | stream | soak | soakvcm | summary.   Threads are pinned to 1 (torch/ORT oversubscribe otherwise).
"""
import argparse, csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot"
CLIPS = D / "cascade_stream/clips"
CONFIGS = {
    "B": dict(threshold=-0.1, margin=4.0, mode="mean_frame"),
    "C1": dict(threshold=-1.1, margin=None, mode="per_char"),
    "C2": dict(threshold=-1.204, margin=None, mode="per_char"),
}
AMBIENT = ROOT / "raw_datasets/ambient-noise"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"}
BASE_CMD = [sys.executable, "-m", "me2_voicegen.vcm.streaming", "--model", "out/vcm/option-d-fil50-ambient-rir-135m",
            "--backend", "onnx", "--onnx-variant", "int8", "--ort-threads", "1", "--grammar", "optionb",
            "--beam-width", "50"]
FOCUS = ("PAUSE", "STOP", "TIME")


def vcm_args(cfg):
    a = [f"--threshold={cfg['threshold']}", "--score-mode", cfg["mode"]]
    if cfg["margin"] is not None:
        a += ["--required-command-margin", str(cfg["margin"])]
    return a


def run(cmd, out=None, err=None):
    return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=out or subprocess.PIPE, stderr=err or subprocess.PIPE, text=True)


# ---------------------------------------------------------------- offline (P1, P2)
def offline(name):
    import importlib.util
    sys.path.insert(0, str(ROOT / "src"))
    from me2_voicegen.vcm import dense_pilot as dp
    recs = dp._load_records(D / "records")
    base = {"default": dp.BASELINE_THRESHOLD}
    B = {(s, c): dp.metrics(recs[(s, c)], "baseline", base, dp.PRODUCTION_MARGIN) for s in ("val", "test") for c in dp.CONDITIONS}
    cfg = CONFIGS[name]
    tau = {"default": cfg["threshold"]}
    S = {(s, c): dp.metrics(recs[(s, c)], "d2", tau, None) for s in ("val", "test") for c in dp.CONDITIONS}
    fa_ok = all(S[("test", c)]["reject_fa"] <= B[("test", c)]["reject_fa"] for c in dp.CONDITIONS)
    exact_ok = all(S[k]["exact"] >= B[k]["exact"] for k in S)
    fails = []
    for (s, c), b in B.items():
        for label, n in b["per_intent_n"].items():
            if n >= dp.R6_MIN_TARGETS and Fraction(S[(s, c)]["per_intent_exact"].get(label, 0) - b["per_intent_exact"].get(label, 0), n) < -dp.R6_ACC_BUDGET:
                fails.append(f"{label}@{s}/{c}")
    # ghosts (cached score records)
    gpath = D / "records/ghost.jsonl"
    if not gpath.exists():
        spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/dense_pilot_ghosts.py")
        g = importlib.util.module_from_spec(spec); sys.modules["g"] = g; spec.loader.exec_module(g)
        from concurrent.futures import ProcessPoolExecutor
        rows = list(csv.DictReader(open(g.MANIFEST, newline="")))
        idx = list(enumerate(rows))
        with ProcessPoolExecutor(32) as pool:
            parts = list(pool.map(g._ghost_worker, [(str(g.CHECKPOINT), str(g.MANIFEST), idx[k::32]) for k in range(32)]))
        by = {i: lp for p in parts for i, lp in p}
        gs = dp.score_rows([by[i] for i in range(len(rows))], [{"index": i, "bucket": "ghost", "label": "x"} for i in range(len(rows))], workers=32)
        gpath.write_text("\n".join(json.dumps(r) for r in gs))
    ghost = [json.loads(l) for l in gpath.read_text().splitlines()]
    b_ghost = sum(dp.accepts(r, "baseline", base, dp.PRODUCTION_MARGIN) for r in ghost)
    s_ghost = sum(dp.accepts(r, "d2", tau, None) for r in ghost)
    out = {"config": name, "P1": fa_ok and exact_ok and not fails, "P1_fa_ok": fa_ok, "P1_exact_ok": exact_ok,
           "P1_per_cell_fails": fails, "P2": s_ghost <= b_ghost, "ghost_fa": s_ghost, "B_ghost_fa": b_ghost,
           "S": {f"{s}/{c}": [v["exact"], v["reject_fa"]] for (s, c), v in S.items()},
           "B": {f"{s}/{c}": [v["exact"], v["reject_fa"]] for (s, c), v in B.items()}}
    (D / f"metadata/dense_gates_offline_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- P4 streaming recall
def build_clips():
    import torch, torchaudio
    mp = ROOT / "out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv"
    rows = [r for r in csv.DictReader(open(mp, newline="")) if r["split"] == "test" and r["bucket"] == "target_commands"]
    seen, chosen = {}, []
    for r in rows:
        n = seen.get(r["label"], 0)
        if r["label"] in FOCUS or n < 30:
            chosen.append(r)
        seen[r["label"]] = n + 1
    CLIPS.mkdir(parents=True, exist_ok=True)
    z = lambda s: torch.zeros(1, int(16000 * s))
    manifest = []
    for i, r in enumerate(chosen):
        w, sr = torchaudio.load(str(mp.parent / r["path"]))
        w = w.mean(0, keepdim=True)
        assert sr == 16000, sr
        f = CLIPS / f"{i:04d}_{r['label']}.wav"
        torchaudio.save(str(f), torch.cat([z(1.0), w, z(1.5)], 1), 16000)
        manifest.append({"file": f.name, "label": r["label"]})
    (CLIPS.parent / "clips.json").write_text(json.dumps(manifest))
    return manifest


def stream(name, workers=64):
    manifest = json.loads((CLIPS.parent / "clips.json").read_text()) if (CLIPS.parent / "clips.json").exists() else build_clips()
    cfg = CONFIGS[name]

    def one(m):
        p = run(BASE_CMD + vcm_args(cfg) + ["--gate", "none", "--policy", "threshold", "--refractory-s", "1.5", "--window-s", "2.5", "--stride-s", "0.25", "--source", str(CLIPS / m["file"])])
        trig = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{") and '"trigger"' in l]
        return {**m, "returncode": p.returncode, "intents": [t["intent"] for t in trig]}
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, manifest))
    (D / f"metadata/dense_gates_stream_{name}.json").write_text(json.dumps(res))
    bad = [r for r in res if r["returncode"] != 0]
    n = len(res)
    correct = sum(1 for r in res if r["intents"] and r["intents"][0] == r["label"])
    wrong = sum(1 for r in res for i in r["intents"] if i != r["label"])
    per = {}
    for lab in FOCUS:
        rs = [r for r in res if r["label"] == lab]
        per[lab] = [sum(1 for r in rs if r["intents"] and r["intents"][0] == lab), len(rs)]
    summ = {"config": name, "n": n, "correct_first_trigger": correct, "wrong_intent_triggers": wrong, "per_focus": per, "nonzero_exit": len(bad)}
    (D / f"metadata/dense_gates_stream_summary_{name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


# ---------------------------------------------------------------- P3 soak
def soak(name):
    cfg = CONFIGS[name]
    files = sorted(AMBIENT.glob("*.wav"))
    (D / "soak").mkdir(exist_ok=True)

    def one(f):
        tag = f"{name}_{abs(hash(f.name)) % 10**6}"
        out, err = D / f"soak/{tag}.jsonl", D / f"soak/{tag}.log"
        with open(out, "w") as o, open(err, "w") as e:
            p = run(BASE_CMD + vcm_args(cfg) + ["--gate", "wakeword", "--policy", "single_period", "--gate-period", "3",
                    "--wakeword-model", "out/wakeword-sesame-ambient-rir-45m", "--wakeword-backend", "onnx",
                    "--log-periods", "--source", str(f)], out=o, err=e)
        text = err.read_text(errors="replace")
        return {"file": f.name, "returncode": p.returncode, "accepts": text.count("period: ACCEPT"),
                "periods": text.count("period: ACCEPT") + text.count("period: REJECT"), "log": err.name}
    with ThreadPoolExecutor(len(files)) as ex:
        res = list(ex.map(one, files))
    summ = {"config": name, "files": res, "accepts_total": sum(r["accepts"] for r in res),
            "periods_total": sum(r["periods"] for r in res), "nonzero_exit": sum(r["returncode"] != 0 for r in res)}
    (D / f"metadata/dense_gates_soak_{name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def soakvcm(name):
    cfg = CONFIGS[name]
    files = sorted(AMBIENT.glob("*.wav"))
    (D / "soak").mkdir(exist_ok=True)

    def one(f):
        tag = f"vcmonly_{name}_{files.index(f)}"
        out, err = D / f"soak/{tag}.jsonl", D / f"soak/{tag}.log"
        with open(out, "w") as o, open(err, "w") as e:
            p = run(BASE_CMD + vcm_args(cfg) + ["--gate", "none", "--policy", "threshold", "--refractory-s", "1.5",
                    "--window-s", "2.5", "--stride-s", "0.25", "--source", str(f)], out=o, err=e)
        trig = [json.loads(l) for l in out.read_text().splitlines() if '"trigger"' in l]
        return {"file": f.name, "returncode": p.returncode, "triggers": len(trig),
                "intents": sorted({t["intent"] for t in trig if t.get("intent")}), "log": err.name}
    with ThreadPoolExecutor(len(files)) as ex:
        res = list(ex.map(one, files))
    summ = {"config": name, "files": res, "triggers_total": sum(r["triggers"] for r in res),
            "nonzero_exit": sum(r["returncode"] != 0 for r in res)}
    (D / f"metadata/dense_gates_soakvcm_{name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def summary():
    rows = {}
    B_stream = json.loads((D / "metadata/dense_gates_stream_summary_B.json").read_text())
    B_soak = json.loads((D / "metadata/dense_gates_soak_B.json").read_text())
    B_soakvcm = json.loads((D / "metadata/dense_gates_soakvcm_B.json").read_text())
    for name in ("C1", "C2"):
        off = json.loads((D / f"metadata/dense_gates_offline_{name}.json").read_text())
        st = json.loads((D / f"metadata/dense_gates_stream_summary_{name}.json").read_text())
        sk = json.loads((D / f"metadata/dense_gates_soak_{name}.json").read_text())
        p3 = sk["accepts_total"] <= B_soak["accepts_total"] and sk["nonzero_exit"] == 0
        sv = json.loads((D / f"metadata/dense_gates_soakvcm_{name}.json").read_text())
        p3b = sv["triggers_total"] <= B_soakvcm["triggers_total"] and sv["nonzero_exit"] == 0
        p4_focus = all(Fraction(st["per_focus"][l][0], st["per_focus"][l][1]) >= Fraction(B_stream["per_focus"][l][0], B_stream["per_focus"][l][1]) - Fraction(3, 100) for l in FOCUS)
        p4_all = Fraction(st["correct_first_trigger"], st["n"]) >= Fraction(B_stream["correct_first_trigger"], B_stream["n"]) - Fraction(5, 1000)
        p4 = p4_focus and p4_all and st["wrong_intent_triggers"] <= B_stream["wrong_intent_triggers"] and st["nonzero_exit"] == 0
        rows[name] = {"P1": off["P1"], "P2": off["P2"], "P3": p3, "P3_periods_vacuous": sk["periods_total"] == 0, "P3b": p3b, "P3b_triggers": [sv["triggers_total"], B_soakvcm["triggers_total"]], "P4": p4, "P4_focus": p4_focus, "P4_overall": p4_all,
                      "verdict": "PASS" if (off["P1"] and off["P2"] and p3 and p3b and p4) else "FAIL"}
    out = {"B_stream": B_stream, "B_soak_accepts": B_soak["accepts_total"], "B_soakvcm_triggers": B_soakvcm["triggers_total"], "candidates": rows}
    (D / "metadata/dense_gates_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("offline", "stream", "soak", "soakvcm", "summary", "build"))
    ap.add_argument("--config", choices=tuple(CONFIGS))
    a = ap.parse_args()
    {"offline": lambda: offline(a.config), "stream": lambda: stream(a.config), "soak": lambda: soak(a.config), "soakvcm": lambda: soakvcm(a.config),
     "summary": summary, "build": build_clips}[a.stage]()
