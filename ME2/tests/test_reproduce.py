"""`scripts/reproduce.py`: stage selection, skipping, output parsing, comparison with expected.json, GPU guard, dry run (no network, no audio, subprocess mocked)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "scripts/reproduce.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

EXPECTED = json.loads((ROOT / "recipes/reproduce/expected.json").read_text())
SUMMARY = """4149 rows, 3823 targets, 326 babble/silence, mean decode 120 ms
fallback: 98.82% (3778/3823), FA 10/326, classifier answered 145 targets
agree: 97.78% (3738/3823), FA 3/326, classifier answered 0 targets
"""
SOAK = """# Soak results: reproduce

`{"x": 1}`

- commands 186: **151 correct first trigger (81.2%)**, 26 missed (5 never opened a period), 9 wrong-action first triggers (intent or slot), 4 wrong-intent triggers in all, 17 early first triggers, 18 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
"""


def ns(**kw):
    base = dict(stages=None, train=False, gpu=None, force=False, dry_run=False, out="out/reproduce", data_dir="raw_datasets/x",
                model_dir="out/vcm/hybrid-ctcwide-clsxl", shards=8)
    return rp.argparse.Namespace(**(base | kw))


def test_parse_summary_takes_the_fallback_line():
    s = rp.parse_summary(SUMMARY)
    assert (s["correct"], s["targets"], s["false_accepts"], s["non_targets"]) == (3778, 3823, 10, 326)


def test_parse_summary_rejects_garbage():
    with pytest.raises(rp.StageError):
        rp.parse_summary("nothing here")


def test_parse_soak_md_and_old_file_without_wrong_action():
    s = rp.parse_soak(SOAK)
    assert s == {"commands": 186, "correct": 151, "oos_sessions": 16, "oos_false_accepts": 3, "gap_triggers": 0, "wrong_action": 9}
    assert rp.parse_soak(SOAK.replace("9 wrong-action first triggers (intent or slot), ", ""))["wrong_action"] is None


def test_parse_manifest_ok_line():
    m = rp.parse_manifest("x\nOK: 28858 rows = the committed manifest (28858) minus the 0 unpublished clips\n")
    assert m == {"rebuilt_rows": 28858, "committed_rows": 28858, "unpublished": 0}


def test_exact_checks_pass_and_fail():
    ok = rp.build_checks("verify", {"test": rp.parse_summary(SUMMARY), "holdout": {"correct": 143, "targets": 186, "false_accepts": 1, "non_targets": 16}}, EXPECTED)
    assert all(c["status"] == "PASS" for c in ok)
    bad = rp.build_checks("verify", {"test": rp.parse_summary(SUMMARY), "holdout": {"correct": 142, "targets": 186, "false_accepts": 1, "non_targets": 16}}, EXPECTED)
    assert [c["status"] for c in bad] == ["PASS", "PASS", "FAIL", "PASS"]
    soak = rp.build_checks("soak", rp.parse_soak(SOAK), EXPECTED)
    assert all(c["status"] == "PASS" for c in soak)
    assert rp.build_checks("soak", rp.parse_soak(SOAK.replace("9 wrong-action", "8 wrong-action")), EXPECTED)[1]["status"] == "FAIL"


def test_manifest_check_fails_on_dropped_clips():
    ok = rp.build_checks("manifest", {"rebuilt_rows": 28858, "committed_rows": 28858, "unpublished": 0}, EXPECTED)
    assert all(c["status"] == "PASS" for c in ok)
    bad = rp.build_checks("manifest", {"rebuilt_rows": 28850, "committed_rows": 28858, "unpublished": 4}, EXPECTED)
    assert bad[1]["status"] == "FAIL"


def test_retrain_is_banded_never_fail():
    near = {"test": {"correct": 3750, "targets": 3823}, "holdout": {"correct": 130, "targets": 186}}
    c = rp.build_checks("train", near, EXPECTED)
    assert [x["status"] for x in c] == ["WITHIN", "OUTSIDE"]
    assert c[0]["delta"] == pytest.approx(-0.73, abs=0.01) and c[1]["band"] == 2
    assert not any(x["status"] == "FAIL" for x in c)


def test_check_missing_value_is_a_fail_when_exact():
    assert rp.check("x", 1, None)["status"] == "FAIL"


def test_stage_selection_defaults_and_subset():
    assert rp.pick_stages(ns()) == ["data", "manifest", "verify", "soak"]
    assert rp.pick_stages(ns(stages="soak,verify")) == ["verify", "soak"]
    assert rp.pick_stages(ns(train=True, gpu=0))[-1] == "train"
    with pytest.raises(SystemExit):
        rp.pick_stages(ns(stages="bogus"))


@pytest.mark.parametrize("kw", [dict(stages="train"), dict(stages="train", train=True), dict(stages="train", train=True, gpu=6)])
def test_train_refused_without_flag_explicit_gpu_or_on_gpu_6(kw):
    with pytest.raises(SystemExit):
        rp.pick_stages(ns(**kw))


def test_main_refuses_gpu_6_before_doing_anything(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(SystemExit):
        rp.main(["--stages", "train", "--train", "--gpu", "6", "--out", str(tmp_path)])
    assert not calls and not list(tmp_path.iterdir())


def test_dry_run_prints_the_recipe_and_runs_nothing(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(rp.subprocess, "Popen", lambda *a, **k: calls.append(a))
    assert rp.main(["--stages", "train", "--train", "--gpu", "3", "--dry-run", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert not calls and not list(tmp_path.iterdir())
    assert out.count("-m me2_voicegen.vcm.train") == 2 and out.count("--device cuda:0") == 2 and "CUDA_VISIBLE_DEVICES=3" in out
    assert "--preset quartznet5x3-wide-heads" in out and "--preset quartznet5x3-xl-heads" in out
    assert out.count("-m me2_voicegen.vcm.export_onnx") == 2 and out.count("--heads-only") == 1
    assert "--perturbation-plan table" in out and "--onecycle-epochs 60" in out and "--max-minutes 135" in out
    assert "hybrid_score.py score" in out and "retrain/cls-xl/export/vcm_heads.int8.onnx" in out


def fake_run_factory(log):
    class R:
        returncode, stdout, stderr = 0, "", ""

    def fake(cmd, **kw):
        log.append([str(c) for c in cmd])
        r = R()
        if "summarize" in cmd:
            r.stdout = SUMMARY
        if any("soak_run.py" in str(c) for c in cmd):
            out = Path(cmd[cmd.index("--sessions") + 1]) / "results"
            out.mkdir(parents=True, exist_ok=True)
            (out / "reproduce.md").write_text(SOAK)
        return r
    return fake


def test_stage_skip_force_and_report(tmp_path, monkeypatch):
    log: list = []
    monkeypatch.setattr(rp.subprocess, "run", fake_run_factory(log))
    out = tmp_path / "o"
    (out / "results").mkdir(parents=True)
    (out / "results/data.json").write_text(json.dumps({"seconds": 1, "checks": [], "values": {
        "ai231_dir": "d", "ai231_repo": "r", "ai231_revision": "rev", "supplemental_dir": "d", "supplemental_repo": "r", "supplemental_revision": "rev"}}))
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    (tmp_path / "recipes/reproduce").mkdir(parents=True)
    (tmp_path / "recipes/reproduce/expected.json").write_text(json.dumps(EXPECTED))
    (tmp_path / "soak/holdout-wake-gap-v1").mkdir(parents=True)
    for f in ("continuous.wav", "continuous.json"):
        (tmp_path / "soak/holdout-wake-gap-v1" / f).write_text("x")
    argv = ["--stages", "data,soak", "--out", str(out)]
    assert rp.main(argv) == 0
    assert sum("soak_run.py" in " ".join(c) for c in log) == 1          # data skipped, soak ran
    report = (out / "REPORT.md").read_text()
    assert "Result: **PASS**" in report and "cached from an earlier run" in report and "soak: wrong-action first triggers | PASS" in report
    assert rp.main(argv) == 0 and sum("soak_run.py" in " ".join(c) for c in log) == 1   # soak now skipped too
    assert rp.main(["--stages", "soak", "--force", "--out", str(out)]) == 0 and sum("soak_run.py" in " ".join(c) for c in log) == 2


def test_failed_check_gives_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(rp.subprocess, "run", fake_run_factory([]))
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    (tmp_path / "recipes/reproduce").mkdir(parents=True)
    changed = json.loads(json.dumps(EXPECTED))
    changed["soak"]["correct"] = 152
    (tmp_path / "recipes/reproduce/expected.json").write_text(json.dumps(changed))
    (tmp_path / "soak/holdout-wake-gap-v1").mkdir(parents=True)
    for f in ("continuous.wav", "continuous.json"):
        (tmp_path / "soak/holdout-wake-gap-v1" / f).write_text("x")
    assert rp.main(["--stages", "soak", "--out", str(tmp_path / "o")]) == 1
    assert "FAIL" in (tmp_path / "o/REPORT.md").read_text()


def test_subprocess_error_stops_and_fails(tmp_path, monkeypatch):
    class R:
        returncode, stdout, stderr = 2, "", "boom"
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    (tmp_path / "recipes/reproduce").mkdir(parents=True)
    (tmp_path / "recipes/reproduce/expected.json").write_text(json.dumps(EXPECTED))
    (tmp_path / "soak/holdout-wake-gap-v1").mkdir(parents=True)
    for f in ("continuous.wav", "continuous.json"):
        (tmp_path / "soak/holdout-wake-gap-v1" / f).write_text("x")
    assert rp.main(["--stages", "soak", "--out", str(tmp_path / "o")]) == 1
    assert "stage soak failed" in (tmp_path / "o/REPORT.md").read_text()
