from collections import Counter

from me2_voicegen.accent_balance.plan_gap_jobs import phrase_key, plan, spoken


def test_alarm_time_is_spoken_without_minutes():
    assert spoken("Alarm 6:00 AM") == "Alarm 6 AM"
    assert phrase_key("ALARM", "Alarm 6:00 AM") == phrase_key("ALARM", "alarm 6 AM")


def test_plan_fills_only_the_shortfall_with_overgeneration():
    variations = {("PAUSE", "pause song"): {"text": "Pause song", "label": "PAUSE"},
                  ("STOP", "stop"): {"text": "Stop", "label": "STOP"}}
    jobs = plan(variations, Counter({("STOP", "stop"): 200}), ["v1", "v2"], target=10, overgen=1.5, seed=0)
    assert len(jobs) == 15 and {j["label"] for j in jobs} == {"PAUSE"}
    assert {j["voice_id"] for j in jobs} == {"v1", "v2"} and len({j["job_id"] for j in jobs}) == 15


def test_noisy_siblings_are_not_counted_as_clips(tmp_path):
    import csv

    from me2_voicegen.accent_balance.plan_gap_jobs import count_existing

    rows = [("a.wav", "STOP", "Stop"), ("a_noisy.wav", "STOP", "Stop"), ("b.wav", "STOP", "Stop")]
    path = tmp_path / "m.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["filename", "source_dataset", "split", "label", "transcript"])
        w.writeheader()
        for fn, label, text in rows:
            w.writerow({"filename": fn, "source_dataset": "fil50_persona", "split": "train", "label": label, "transcript": text})
    assert count_existing(path, {("STOP", "stop"): {}})[("STOP", "stop")] == 2
