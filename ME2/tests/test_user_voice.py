"""Tests for me2_voicegen.user_voice (user-voice weak-phrase augmentation)."""

import csv
from collections import Counter

import pytest

from me2_voicegen.user_voice import collate as uc
from me2_voicegen.user_voice import plan as up
from me2_voicegen.user_voice import prep as upp
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError


# ---- prep ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name,text,take", [
    ("Call 1.mp3", "Call", 1), ("Shut off the lights 3.mp3", "Shut off the lights", 3),
    ("Change color to blue.mp3", "Change color to blue", 1), ("What time is it 2.mp3", "What time is it", 2),
])
def test_parse_clip_name(name, text, take):
    assert upp.parse_clip_name(name) == (text, take)


def test_clip_id_is_slug_and_take():
    assert upp.clip_id_for("Shut off the lights", 2) == "uv_shut_off_the_lights_2"


@pytest.mark.parametrize("text,label", [
    ("Call", "CALL"), ("Place a call", "CALL"), ("Make a phone call", "CALL"), ("Set color to green", "COLOR"),
    ("Switch color to red", "COLOR"), ("Send my message", "MESSAGE"), ("Show my reminders", "LIST_REMINDERS"),
    ("Shut off the lights", "LIGHT_OFF"), ("What time is it", "TIME"),
])
def test_label_for_uses_grammar(text, label):
    assert upp.label_for(text)[0] == label


def test_label_for_color_keeps_slot():
    assert upp.label_for("Change color to blue") == ("COLOR", {"COLOR": "blue"})


def test_label_for_rejects_out_of_grammar():
    with pytest.raises(upp.PrepError, match="not accepted"):
        upp.label_for("Open the pod bay doors")


# ---- plan: personas -----------------------------------------------------------------------
def _voices():
    v = [{"voice_id": f"fsc_{i:03d}", "prompt_source": "sapinsapin", "split": "train"} for i in range(86)]
    v += [{"voice_id": f"fsc_v{i:03d}", "prompt_source": "sapinsapin", "split": "val"} for i in range(18)]
    v += [{"voice_id": f"fsc_t{i:03d}", "prompt_source": "sapinsapin", "split": "test"} for i in range(19)]
    v += [{"voice_id": f"ref_{i:02d}", "prompt_source": "references", "split": "train"} for i in range(17)]
    return v


def test_sample_personas_train_only_stratified_and_deterministic():
    a = up.sample_personas(_voices(), 50, seed=0)
    assert len(a) == 50 and len({p["voice_id"] for p in a}) == 50
    assert {p["split"] for p in a} == {"train"}
    assert Counter(p["prompt_source"] for p in a) == {"sapinsapin": 42, "references": 8}
    assert a == up.sample_personas(_voices(), 50, seed=0)
    assert a != up.sample_personas(_voices(), 50, seed=1)


def test_sample_personas_too_many_raises():
    with pytest.raises(ValueError):
        up.sample_personas(_voices(), 500)


# ---- plan: jobs ---------------------------------------------------------------------------
def _clips():
    spec = {"CALL": 3, "COLOR": 8, "MESSAGE": 3, "LIGHT_OFF": 3, "TIME": 2, "LIST_REMINDERS": 1}
    return [{"clip_id": f"{lab.lower()}_{i}", "text": f"{lab} {i}", "label": lab, "path": f"/x/{lab}_{i}.wav"}
            for lab, n in spec.items() for i in range(n)]


def test_plan_jobs_caps_per_intent_and_spreads_evenly():
    personas = up.sample_personas(_voices(), 50, seed=0)
    jobs = up.plan_jobs(_clips(), personas, cap=150, seed=0)
    per = Counter(j["label"] for j in jobs)
    assert per == {"CALL": 150, "COLOR": 150, "MESSAGE": 150, "LIGHT_OFF": 150, "TIME": 100, "LIST_REMINDERS": 50}
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert {j["split"] for j in jobs} == {"train"}
    color = Counter(j["source_row_ref"] for j in jobs if j["label"] == "COLOR")
    assert max(color.values()) - min(color.values()) <= 1  # 150 over 8 clips -> 18/19 each
    # no (clip, persona) pair repeats
    assert len({(j["source_row_ref"], j["voice_id"]) for j in jobs}) == len(jobs)


def test_plan_jobs_is_deterministic_and_noisy_rate_plausible():
    personas = up.sample_personas(_voices(), 50, seed=0)
    a, b = up.plan_jobs(_clips(), personas, seed=0), up.plan_jobs(_clips(), personas, seed=0)
    assert a == b
    rate = sum(j["noisy_target"] == "1" for j in a) / len(a)
    assert abs(rate - up.NOISY_RATE) < 0.08


# ---- collate ------------------------------------------------------------------------------
def _job(split="train"):
    return {"job_id": "uv_x__fsc_001", "voice_id": "fsc_001", "text": "Call", "label": "CALL", "split": split,
            "source_row_ref": "uv_call_1", "duration": "1.3", "path": "out/x/uv_x__fsc_001.wav", "noisy_target": "0"}


def test_build_new_row_fields(tmp_path):
    r = uc.build_new_row(_job(), tmp_path)
    assert (r["source_dataset"], r["split"], r["bucket"], r["label"], r["transcript"]) == (
        "user_voice_persona", "train", "target_commands", "CALL", "Call")
    assert r["group_id"] == "fsc_001" and r["original_dataset"] == "sapinsapin" and r["sample_rate"] == "16000"


def test_build_new_row_rejects_non_train(tmp_path):
    with pytest.raises(ManifestValidationError, match="train-only"):
        uc.build_new_row(_job("val"), tmp_path)


def _write_manifest(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _row(fn, label, split, src="optionb", **kw):
    r = {"filename": fn, "path": f"audio/{fn}", "bucket": "target_commands", "label": label, "duration": "1.0",
         "sample_rate": "16000", "resampled": "False", "source_dataset": src, "source_relpath": fn, "group_id": f"g_{fn}",
         "split": split, "transcript": "call", "original_dataset": "x", "noise_source_file": "", "noise_chunk_id": "", "snr_db": ""}
    r.update(kw)
    return r


def test_assemble_preserves_base_adds_train_only_and_reports_balance(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, "run_fail_loud_checks", lambda rows, out_dir: None)
    base_dir = tmp_path / "base"; base_dir.mkdir()
    base = [_row("a.wav", "CALL", "train"), _row("b.wav", "TIME", "val"), _row("c.wav", "TIME", "test")]
    _write_manifest(base_dir / "manifest.csv", base)
    out = tmp_path / "out"; out.mkdir()
    new = [_row("n1.wav", "CALL", "train", src="user_voice_persona", path="audio_uv/n1.wav")]
    _write_manifest(out / "new_rows.csv", new)
    amb = tmp_path / "amb"; amb.mkdir()
    _write_manifest(amb / "manifest.csv", [_row("n1.wav", "CALL", "train", src="user_voice_persona"),
                                             _row("n1_amb.wav", "CALL", "train", src="user_voice_persona_ambient", snr_db="12.0")])
    dest = uc.assemble(base_dir / "manifest.csv", out / "new_rows.csv", amb / "manifest.csv", out)
    rows = list(csv.DictReader(dest.open()))
    assert len(rows) == 3 + 2  # base + new clean + new ambient (the clean copy in the ambient manifest is not re-added)
    assert [r["filename"] for r in rows[:3]] == ["a.wav", "b.wav", "c.wav"]
    assert [r for r in rows if r["split"] in ("val", "test")] == [r for r in rows[:3] if r["split"] in ("val", "test")]
    assert Counter(r["source_dataset"] for r in rows[3:]) == {"user_voice_persona": 1, "user_voice_persona_ambient": 1}
    assert "| CALL | 1 | 2 | 3 |" in (out / "assemble_report.md").read_text()


def test_assemble_rejects_added_non_train_row(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, "run_fail_loud_checks", lambda rows, out_dir: None)
    base_dir = tmp_path / "base"; base_dir.mkdir()
    _write_manifest(base_dir / "manifest.csv", [_row("a.wav", "CALL", "train")])
    out = tmp_path / "out"; out.mkdir()
    _write_manifest(out / "new_rows.csv", [_row("n1.wav", "CALL", "val", src="user_voice_persona")])
    with pytest.raises(ManifestValidationError, match="must be train"):
        uc.assemble(base_dir / "manifest.csv", out / "new_rows.csv", None, out)


def test_resolve_transcript_knows_user_voice_sources():
    """New sources resolve exactly like fil50_persona (own `transcript` column)."""
    from me2_voicegen.vcm.text import resolve_transcript

    row = {"transcript": "Shut off the lights", "label": "LIGHT_OFF"}
    expect = resolve_transcript({**row, "source_dataset": "fil50_persona"})
    assert expect
    for src in ("user_voice_persona", "user_voice_persona_ambient"):
        assert resolve_transcript({**row, "source_dataset": src}) == expect
