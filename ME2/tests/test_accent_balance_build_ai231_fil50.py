import csv
from random import Random

from me2_voicegen.accent_balance.build_ai231_fil50 import assign_voices, build, pick, rescued_job_ids


def _write(path, fields, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def test_pick_is_round_robin_over_voices_and_capped():
    pool = [{"filename": f"{v}{i}.wav", "voice": v} for v in "ab" for i in range(5)]
    rows = pick(pool, 4, Random(0))
    assert len(rows) == 4 and sum(r["voice"] == "a" for r in rows) == 2


def test_assign_voices_is_disjoint_and_close_to_the_target_shares():
    sizes = {f"v{i}": 100 for i in range(10)}
    split_of = assign_voices(sizes, n_test=3, n_val=1)
    assert sorted(split_of.values()).count("test") == 3 and sorted(split_of.values()).count("val") == 1
    assert assign_voices(sizes, 3, 1) == split_of  # deterministic


def test_build_splits_voices_caps_each_variation_and_keeps_every_base_row(tmp_path):
    _write(tmp_path / "variations.csv", ["label", "variation", "value", "phrase"],
           [{"label": "STOP", "variation": "1", "value": "", "phrase": "Stop"},
            {"label": "TIMER", "variation": "1", "value": "10 seconds", "phrase": "Timer 10 seconds"}])
    base = tmp_path / "base" / "manifest.csv"
    base.parent.mkdir()
    base_fields = ["filename", "path", "bucket", "label", "split", "transcript", "variation", "slot_value", "source_dataset",
                   "variation_match"]
    base_rows = [{"filename": f"{s}.wav", "path": f"audio/{s}.wav", "bucket": "target_commands", "label": "STOP", "split": s,
                  "transcript": "Stop", "variation": "Stop", "slot_value": "", "source_dataset": "optionb",
                  "variation_match": "exact"} for s in ("train", "val", "test")]
    _write(base, base_fields, base_rows)
    persona = tmp_path / "old" / "manifest.csv"
    persona.parent.mkdir()
    pf = ["filename", "path", "source_dataset", "split", "label", "transcript", "group_id", "duration", "source_relpath"]
    rows = [{"filename": f"{lab}{i}_{v}.wav", "path": f"a/{lab}{i}_{v}.wav", "source_dataset": "fil50_persona", "split": "train",
             "label": lab, "transcript": text, "group_id": f"v{v}", "duration": "1.0", "source_relpath": "r.wav"}
            for lab, text in (("STOP", "Stop"), ("TIMER", "Timer 10 seconds")) for i in range(6) for v in range(4)]
    rows.append({**rows[0], "filename": "noisy_noisy.wav"})  # noisy siblings are never candidates
    _write(persona, pf, rows)
    out = tmp_path / "out" / "manifest.csv"
    result = build(base, persona, None, None, tmp_path / "variations.csv", out, {"train": 3, "val": 2, "test": 2}, seed=0,
                   n_test_voices=1, n_val_voices=1)
    got = list(csv.DictReader(out.open()))
    persona_rows = [r for r in got if r["source_dataset"] == "fil50_persona"]
    assert [r for r in got if r["source_dataset"] == "optionb"] and len(got) == 3 + len(persona_rows)  # nothing removed
    by_split = {s: {r["group_id"] for r in persona_rows if r["split"] == s} for s in ("train", "val", "test")}
    assert not (by_split["train"] & by_split["val"] or by_split["train"] & by_split["test"] or by_split["val"] & by_split["test"])
    assert result["persona_added"] == {"train": 6, "val": 4, "test": 4}
    assert not any("noisy" in r["filename"] for r in persona_rows)


def test_percent_format_false_flags_are_rescued_but_misheard_words_are_not(tmp_path):
    shim = tmp_path / "shim" / "vcm_references"
    shim.mkdir(parents=True)
    _write(shim / "index.csv", ["filename", "job_id"], [{"filename": "a.wav", "job_id": "j1"}, {"filename": "b.wav", "job_id": "j2"}])
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "r.md").write_text(
        "| File | Expected | Transcribed | Score | Flagged |\n|---|---|---|---|---|\n"
        "| a.wav | Brightness 100 percent | Brightness, 100%. | 0.778 | no |\n"
        "| b.wav | Reminder Study | Recon Study. | 0.720 | no |\n")
    assert rescued_job_ids(reports, tmp_path / "shim") == {"j1"}
