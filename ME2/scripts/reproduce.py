"""One-command reproduction of the hybrid's published numbers, on CPU, from public data.

Stages (default: data, manifest, verify, soak; `train` only with --train --gpu N):
  data      download the pinned ai231 dataset (DOI 10.57967/hf/10723) and the gap-fill / numeral-wordings shards
  manifest  rebuild the ai231-fil50-supp training manifest; must equal the committed one (scripts/rebuild_training_manifest.py)
  verify    score the checked-in INT8 ONNX hybrid on ai231 test (clean) and holdout (scripts/hybrid_score.py)
  soak      stream the holdout soak recording once and score it (scripts/soak_run.py)
  train     retrain wide + XL, export, score; deltas are reported against a band, never failed (A100, ~2.5 h)

Each stage writes <out>/results/<stage>.json and is skipped when that file exists (--force redoes it). The report is <out>/REPORT.md;
the exit code is non-zero when any check FAILs. `--dry-run` prints the commands without running them.

    uv run python scripts/reproduce.py                      # or: make reproduce
    .venv-pi/bin/python scripts/reproduce.py --stages soak  # Raspberry Pi (no make, no uv)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["data", "manifest", "verify", "soak", "train"]
DEFAULT_STAGES = STAGES[:4]
FORBIDDEN_GPU = 6
CLS_THRESHOLD = "0.8787"
WAKEWORD_THRESHOLD = "0.8"     # Makefile HYBRID_WAKEWORD_THRESHOLD
CLS_SLOT_THRESHOLD = "0.6"     # Makefile HYBRID_CLS_SLOT_THRESHOLD
LICENSE_NOTE = ("Trained on the public airimonda/ai231-me2-voice-commands v2 dataset plus the public "
                "martinnavs/ai231-fil-supplemental-data persona clips (research and education only)")
TRAIN_FLAGS = ["--seed", "0", "--p-rir", "0.7", "--p-timestretch", "0.25", "--p-noise", "0.5", "--p-babble", "0.15", "--noise-source", "dataset",
               "--perturbation-plan", "table", "--dump-plan", "--skip-prenoised", "--noise-random-offset",
               "--onecycle-epochs", "60", "--max-epochs", "70", "--patience", "20", "--max-minutes", "135"]
RETRAIN = [("ctc-wide", "quartznet5x3-wide-heads", False), ("cls-xl", "quartznet5x3-xl-heads", True)]


class StageError(RuntimeError):
    pass


def parse_summary(text: str) -> dict:
    """`hybrid_score.py summarize` output -> the `fallback` policy line (the hybrid's policy A) and the row counts."""
    rows = re.search(r"(\d+) rows, (\d+) targets, (\d+) babble/silence", text)
    fb = re.search(r"^fallback: ([\d.]+)% \((\d+)/(\d+)\), FA (\d+)/(\d+)", text, re.M)
    if not rows or not fb:
        raise StageError("could not parse the hybrid_score summarize output")
    return {"rows": int(rows[1]), "correct": int(fb[2]), "targets": int(fb[3]), "pct": float(fb[1]), "false_accepts": int(fb[4]), "non_targets": int(fb[5])}


def parse_soak(md: str) -> dict:
    """The soak result markdown (`scripts/soak_run.py`) -> counts. `wrong_action` is None for result files from before that counter existed."""
    cmd = re.search(r"commands (\d+): \*\*(\d+) correct first trigger", md)
    oos = re.search(r"out-of-scope sessions (\d+): (\d+) false accepts", md)
    gap = re.search(r"triggers in the ambient gaps[^:]*: (\d+)", md)
    wrong = re.search(r"(\d+) wrong-action first triggers", md)
    if not (cmd and oos and gap):
        raise StageError("could not parse the soak result markdown")
    return {"commands": int(cmd[1]), "correct": int(cmd[2]), "oos_sessions": int(oos[1]), "oos_false_accepts": int(oos[2]),
            "gap_triggers": int(gap[1]), "wrong_action": int(wrong[1]) if wrong else None}


def parse_manifest(text: str) -> dict:
    m = re.search(r"OK: (\d+) rows = the committed manifest \((\d+)\) minus the (\d+) unpublished", text)
    if not m:
        raise StageError("rebuild_training_manifest.py did not print its OK line")
    return {"rebuilt_rows": int(m[1]), "committed_rows": int(m[2]), "unpublished": int(m[3])}


def check(name: str, expected, got, mode: str = "exact", band: float = 0.0) -> dict:
    """mode exact: PASS/FAIL. mode band (retrained models): WITHIN/OUTSIDE a +-`band` interval around `expected`; never a FAIL."""
    if got is None:
        status = "FAIL" if mode == "exact" else "OUTSIDE"
    elif mode == "exact":
        status = "PASS" if got == expected else "FAIL"
    else:
        status = "WITHIN" if abs(got - expected) <= band else "OUTSIDE"
    row = {"name": name, "status": status, "expected": expected, "got": got}
    if mode == "band" and got is not None:
        row["delta"] = round(got - expected, 2)
        row["band"] = band
    return row


def pct(c: int, t: int) -> float:
    return round(100 * c / t, 2)


def build_checks(stage: str, values: dict, exp: dict) -> list[dict]:
    if stage == "data":
        return [check("ai231 revision", exp["data"]["ai231"]["revision"], values.get("ai231_revision")),
                check("gap-fill / numeral-wordings revision", exp["data"]["supplemental"]["revision"] if values.get("supplemental_source") != "ai231" else values.get("ai231_revision"),
                      values.get("supplemental_revision"))]
    if stage == "manifest":
        e = exp["manifest"]
        got = f"{values['rebuilt_rows'] + values['unpublished']}/{values['committed_rows']}"
        return [check("rebuilt manifest rows equal the committed manifest", f"{e['rows']}/{e['rows']}", got),
                check("unpublished clips dropped", e["unpublished"], values["unpublished"])]
    if stage == "verify":
        t, h, e = values["test"], values["holdout"], exp["verify"]
        return [check("test clean: correct / targets", f"{e['test']['correct']}/{e['test']['targets']}", f"{t['correct']}/{t['targets']}"),
                check("test clean: false accepts / non-commands", f"{e['test']['false_accepts']}/{e['test']['non_targets']}", f"{t['false_accepts']}/{t['non_targets']}"),
                check("holdout: correct / targets", f"{e['holdout']['correct']}/{e['holdout']['targets']}", f"{h['correct']}/{h['targets']}"),
                check("holdout: false accepts / out-of-scope", f"{e['holdout']['false_accepts']}/{e['holdout']['non_targets']}", f"{h['false_accepts']}/{h['non_targets']}")]
    if stage == "soak":
        s, e = values, exp["soak"]
        return [check("soak: correct first trigger / commands", f"{e['correct']}/{e['commands']}", f"{s['correct']}/{s['commands']}"),
                check("soak: wrong-action first triggers", e["wrong_action"], s["wrong_action"]),
                check("soak: out-of-scope false accepts / sessions", f"{e['oos_false_accepts']}/{e['oos_sessions']}", f"{s['oos_false_accepts']}/{s['oos_sessions']}"),
                check("soak: triggers in the ambient gaps", e["gap_triggers"], s["gap_triggers"])]
    if stage == "train":
        band, e = exp["retrain"]["band_points"], exp["verify"]
        t, h = values["test"], values["holdout"]
        return [check("retrained test clean accuracy (%)", pct(e["test"]["correct"], e["test"]["targets"]), pct(t["correct"], t["targets"]), "band", band),
                check("retrained holdout accuracy (%)", pct(e["holdout"]["correct"], e["holdout"]["targets"]), pct(h["correct"], h["targets"]), "band", band)]
    raise ValueError(stage)


class Ctx:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.dry = a.dry_run
        self.out = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
        self.data_dir = ROOT / a.data_dir if not Path(a.data_dir).is_absolute() else Path(a.data_dir)
        self.model_dir = ROOT / a.model_dir if not Path(a.model_dir).is_absolute() else Path(a.model_dir)
        self.results = self.out / "results"
        self.logs = self.out / "logs"
        self.expected = json.loads((ROOT / "recipes/reproduce/expected.json").read_text())
        self.env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}

    @property
    def imported(self) -> Path:
        return self.out / "ai231-v2/manifest.csv"

    @property
    def eval_manifest(self) -> Path:
        return self.out / "ai231-v2/manifest.eval-exact.csv"

    def show(self, cmd: list, env_extra: dict | None = None) -> None:
        prefix = " ".join(f"{k}={v}" for k, v in (env_extra or {}).items())
        line = " ".join(str(c) for c in cmd).replace(sys.executable, "python").replace(str(ROOT) + "/", "")
        print("$ " + (prefix + " " if prefix else "") + line, flush=True)

    def run(self, cmd: list, log: str, env_extra: dict | None = None) -> str:
        """Run one command, logging its output to <out>/logs/<log>; returns the output ('' in a dry run)."""
        self.show(cmd, env_extra)
        if self.dry:
            return ""
        self.logs.mkdir(parents=True, exist_ok=True)
        p = subprocess.run([str(c) for c in cmd], cwd=ROOT, env={**self.env, **(env_extra or {})}, capture_output=True, text=True)
        text = p.stdout + (("\n" + p.stderr) if p.stderr else "")
        (self.logs / log).write_text(text)
        if p.returncode:
            raise StageError(f"`{' '.join(str(c) for c in cmd[:3])} ...` exited {p.returncode}; see {self.logs / log}\n{text[-600:]}")
        return text

    def run_parallel(self, cmds: list[tuple[list, dict]], log: str) -> None:
        self.show(*cmds[0])
        if len(cmds) > 1:
            print(f"  ... and the same for shards 1-{len(cmds) - 1}", flush=True)
        if self.dry:
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        procs = [(subprocess.Popen([str(x) for x in c], cwd=ROOT, env={**self.env, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), c) for c, env in cmds]
        outs, bad = [], []
        for p, c in procs:
            o = p.communicate()[0]
            outs.append(o)
            if p.returncode:
                bad.append(p.returncode)
        (self.logs / log).write_text("\n".join(outs))
        if bad:
            raise StageError(f"{len(bad)} of {len(procs)} shard processes failed; see {self.logs / log}\n" + next(o for o in outs if "Traceback" in o)[-600:])

    def load(self, stage: str) -> dict | None:
        f = self.results / f"{stage}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def need(self, stage: str) -> dict:
        r = self.load(stage)
        if r is None:
            if self.dry:
                return {"values": {"ai231_dir": str(self.data_dir), "supplemental_dir": str(self.data_dir), "supplemental_source": "ai231"}}
            raise StageError(f"stage `{stage}` has not run (no {self.results / (stage + '.json')}); run it first")
        return r


def py(*args) -> list:
    return [sys.executable, *args]


def stage_data(ctx: Ctx) -> dict | None:
    ai = ctx.expected["data"]["ai231"]
    sup = ctx.expected["data"]["supplemental"]
    print(f"snapshot_download({ai['repo']!r}, revision={ai['revision']!r}, local_dir={str(ctx.data_dir)!r})", flush=True)
    if ctx.dry:
        print(f"then, unless it holds gap_fill/ and numeral_wordings/: snapshot_download({sup['repo']!r}, revision={sup['revision']!r}, allow_patterns=gap_fill/*, numeral_wordings/*)")
        return None
    from huggingface_hub import snapshot_download
    snapshot_download(ai["repo"], repo_type="dataset", revision=ai["revision"], local_dir=str(ctx.data_dir))
    values = {"ai231_dir": str(ctx.data_dir), "ai231_repo": ai["repo"], "ai231_revision": ai["revision"]}
    if list(ctx.data_dir.glob("gap_fill/train-*.parquet")) and list(ctx.data_dir.glob("numeral_wordings/train-*.parquet")):
        values |= {"supplemental_source": "ai231", "supplemental_dir": str(ctx.data_dir), "supplemental_repo": ai["repo"], "supplemental_revision": ai["revision"]}
    else:
        sdir = ctx.data_dir.parent / "ai231-fil-supplemental-gap-fill"
        snapshot_download(sup["repo"], repo_type="dataset", revision=sup["revision"], allow_patterns=["gap_fill/*", "numeral_wordings/*"], local_dir=str(sdir))
        values |= {"supplemental_source": "supplemental", "supplemental_dir": str(sdir), "supplemental_repo": sup["repo"], "supplemental_revision": sup["revision"]}
    print(f"gap_fill / numeral_wordings source: {values['supplemental_repo']} @ {values['supplemental_revision']}", flush=True)
    return values


def ensure_import(ctx: Ctx, d: dict) -> None:
    if ctx.imported.exists() and not ctx.dry:
        return
    ctx.run(py("-m", "me2_voicegen.vcm.optionb.import_ai231", "--src", d["ai231_dir"], "--out", ctx.out / "ai231-v2"), "import_ai231.log")


def stage_manifest(ctx: Ctx) -> dict | None:
    d = ctx.need("data")["values"]
    ensure_import(ctx, d)
    text = ctx.run(py("scripts/rebuild_training_manifest.py", "--ai231", d["ai231_dir"], "--out", ctx.out / "rebuilt", "--ai231-v2-manifest", ctx.imported,
                      "--persona-dir", Path(d["ai231_dir"]) / "supplemental_fil", "--numerals-dir", Path(d["supplemental_dir"]) / "numeral_wordings",
                      "--gap-fill-dir", Path(d["supplemental_dir"]) / "gap_fill"), "manifest.log")
    return None if ctx.dry else parse_manifest(text)


def score_split(ctx: Ctx, ctc: Path, cls: Path, split: str, tag: str, manifest: Path) -> dict | None:
    sdir = ctx.out / "score" / tag
    n = ctx.a.shards
    cmds = [(py("scripts/hybrid_score.py", "score", "--manifest", manifest, "--split", split, "--shard", k, "--n-shards", n, "--ctc-checkpoint", ctc,
                "--cls-checkpoint", cls, "--cls-threshold", CLS_THRESHOLD, "--out-dir", sdir), {"OMP_NUM_THREADS": "2"}) for k in range(n)]
    ctx.run_parallel(cmds, f"score-{tag}-{split}.log")
    text = ctx.run(py("scripts/hybrid_score.py", "summarize", "--dir", sdir, "--split", split, "--cond", "clean"), f"summarize-{tag}-{split}.log")
    return None if ctx.dry else parse_summary(text)


def verify_models(ctx: Ctx, ctc: Path, cls: Path, tag: str) -> dict | None:
    d = ctx.need("data")["values"]
    ensure_import(ctx, d)
    if not ctx.eval_manifest.exists() or ctx.dry:
        ctx.run(py("-m", "me2_voicegen.vcm.semantic_eval", "build-manifest", "--ai231", ctx.imported, "--out", ctx.eval_manifest), "build_eval_manifest.log")
    # test: the exact-variation rows (no dataset-only noise); holdout: every row of the imported manifest (186 commands + 16 out-of-scope)
    out = {"test": score_split(ctx, ctc, cls, "test", tag, ctx.eval_manifest), "holdout": score_split(ctx, ctc, cls, "holdout", tag, ctx.imported)}
    return None if ctx.dry else out


def stage_verify(ctx: Ctx) -> dict | None:
    return verify_models(ctx, ctx.model_dir / "ctc-wide/export/vcm_model.int8.onnx", ctx.model_dir / "cls-xl/export/vcm_heads.int8.onnx", "checked-in")


def stage_soak(ctx: Ctx) -> dict | None:
    src = ROOT / "soak/holdout-wake-gap-v1"
    sdir = ctx.out / "soak"
    if not ctx.dry:
        sdir.mkdir(parents=True, exist_ok=True)
        for f in ("continuous.wav", "continuous.json"):
            if not (sdir / f).exists():
                (sdir / f).symlink_to(src / f)
    ctx.run(py("scripts/soak_run.py", "--sessions", sdir, "--continuous", "--name", "reproduce", "--backend", "onnx", "--models", ctx.model_dir,
               "--wakeword-threshold", WAKEWORD_THRESHOLD, "--cls-slot-threshold", CLS_SLOT_THRESHOLD), "soak.log")
    return None if ctx.dry else parse_soak((sdir / "results/reproduce.md").read_text())


def train_commands(ctx: Ctx, gpu: int) -> list[tuple[str, list, dict]]:
    manifest = ctx.out / "rebuilt/ai231-fil50-supp/manifest.csv"
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    cmds = []
    for name, preset, heads_only in RETRAIN:
        run = ctx.out / "retrain" / name
        cmds.append((f"train-{name}.log", py("-m", "me2_voicegen.vcm.train", "--device", "cuda:0", "--manifest", manifest, "--preset", preset, *TRAIN_FLAGS,
                                              "--out-dir", run, "--license-note", LICENSE_NOTE), env))
        cmds.append((f"export-{name}.log", py("-m", "me2_voicegen.vcm.export_onnx", "--checkpoint", run / "checkpoints/checkpoint.pt", "--out-dir", run, "--manifest", manifest,
                                               *(["--heads-only"] if heads_only else [])), env))
    return cmds


def stage_train(ctx: Ctx) -> dict | None:
    gpu = ctx.a.gpu
    for log, cmd, env in train_commands(ctx, gpu):
        ctx.run(cmd, log, env)
    run = ctx.out / "retrain"
    return verify_models(ctx, run / "ctc-wide/export/vcm_model.int8.onnx", run / "cls-xl/export/vcm_heads.int8.onnx", "retrained")


RUNNERS = {"data": stage_data, "manifest": stage_manifest, "verify": stage_verify, "soak": stage_soak, "train": stage_train}


def pick_stages(a: argparse.Namespace) -> list[str]:
    if a.stages:
        picked = [s.strip() for s in a.stages.split(",") if s.strip()]
        bad = [s for s in picked if s not in STAGES]
        if bad:
            raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")
    else:
        picked = DEFAULT_STAGES + (["train"] if a.train else [])
    if "train" in picked:
        if not a.train:
            raise SystemExit("the train stage needs --train (and --gpu N); it is a ~2.5 h A100 retrain")
        if a.gpu is None:
            raise SystemExit("--gpu N is required for the train stage (an explicit GPU, never a default)")
        if a.gpu == FORBIDDEN_GPU:
            raise SystemExit(f"GPU {FORBIDDEN_GPU} is off limits on this node")
    return [s for s in STAGES if s in picked]


def fmt(v) -> str:
    return "n/m" if v is None else str(v)


def write_report(ctx: Ctx, picked: list[str], done: dict, failure: str | None) -> bool:
    rows = [c for s in picked if s in done for c in done[s]["checks"]]
    ok = failure is None and not any(c["status"] == "FAIL" for c in rows)
    data = ctx.load("data")
    md = ["# Reproduction report", "", f"Result: **{'PASS' if ok else 'FAIL'}**" + (f" ({failure.splitlines()[0]})" if failure else ""), "",
          "| Stage | Check | Status | Expected | Got |", "|---|---|---|---|---|"]
    for s in picked:
        for c in done.get(s, {}).get("checks", []):
            got = fmt(c["got"]) + (f" (delta {c['delta']:+}, band +-{c['band']})" if "delta" in c else "")
            md.append(f"| {s} | {c['name']} | {c['status']} | {c['expected']} | {got} |")
    md += ["", "| Stage | Wall-clock | |", "|---|---:|---|"]
    for s in picked:
        if s in done:
            sec = done[s]["seconds"]
            md.append(f"| {s} | {int(sec // 60)} min {int(sec % 60)} s | {'cached from an earlier run' if done[s].get('cached') else ''} |")
    md.append(f"| total | {int(sum(done[s]['seconds'] for s in picked if s in done) // 60)} min | |")
    if data:
        v = data["values"]
        md += ["", "Data:", f"- ai231: `{v['ai231_repo']}` @ `{v['ai231_revision']}` (DOI 10.57967/hf/10723)",
               f"- persona clips: that revision's `supplemental_fil/`",
               f"- gap_fill and numeral_wordings: `{v['supplemental_repo']}` @ `{v['supplemental_revision']}`"]
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--train", action="store_true", help="add the opt-in retrain stage (needs --gpu N)")
    ap.add_argument("--gpu", type=int, default=None, help="physical GPU for the train stage (6 is refused)")
    ap.add_argument("--force", action="store_true", help="redo stages whose result already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--out", default="out/reproduce")
    ap.add_argument("--data-dir", default="raw_datasets/ai231-me2-voice-commands-v2")
    ap.add_argument("--model-dir", default="out/vcm/hybrid-ctcwide-clsxl")
    ap.add_argument("--shards", type=int, default=8, help="parallel CPU shards for scoring")
    a = ap.parse_args(argv)
    picked = pick_stages(a)
    ctx = Ctx(a)
    done: dict = {}
    failure = None
    for s in picked:
        cached = ctx.load(s)
        if cached and not a.force and not a.dry_run:
            print(f"== {s}: skipped, result exists ({ctx.results / (s + '.json')}); --force to redo", flush=True)
            done[s] = {**cached, "cached": True}
            continue
        print(f"== {s}", flush=True)
        t0 = time.monotonic()
        try:
            values = RUNNERS[s](ctx)
        except StageError as e:
            failure = f"stage {s} failed: {e}"
            print(failure, file=sys.stderr)
            break
        if a.dry_run:
            continue
        rec = {"seconds": round(time.monotonic() - t0, 1), "values": values, "checks": build_checks(s, values, ctx.expected)}
        ctx.results.mkdir(parents=True, exist_ok=True)
        (ctx.results / f"{s}.json").write_text(json.dumps(rec, indent=1))
        done[s] = rec
    if a.dry_run:
        return 0
    return 0 if write_report(ctx, picked, done, failure) else 1


if __name__ == "__main__":
    sys.exit(main())
