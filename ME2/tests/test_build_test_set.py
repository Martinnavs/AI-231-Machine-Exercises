"""Unit tests for build_test_set.py, against small synthetic fixture trees
under tmp_path (never the real multi-GB corpus - see test_build_test_set_slow
skip pattern conventions in this repo for anything that would need the real
data)."""

from __future__ import annotations

import csv
import struct
import wave
from pathlib import Path

import pytest

import me2_voicegen.dataset_tools.build_test_set as bts


# ---------------------------------------------------------------------------
# wav helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, sample_rate: int, n_samples: int, channels: int = 1, sampwidth: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        frame = [0] * (n_samples * channels)
        w.writeframes(struct.pack(f"<{len(frame)}h", *frame))


# ---------------------------------------------------------------------------
# largest_remainder / compute_allocation
# ---------------------------------------------------------------------------


def test_largest_remainder_exact_split():
    result = bts.largest_remainder({"a": 1, "b": 1, "c": 1}, 10)
    assert sum(result.values()) == 10
    assert result == {"a": 4, "b": 3, "c": 3}  # tie-break alphabetical on equal remainders


def test_largest_remainder_zero_total():
    assert bts.largest_remainder({"a": 1, "b": 2}, 0) == {"a": 0, "b": 0}


def test_compute_allocation_matches_established_real_values():
    intent_pools = {
        "CALL": 73, "PAUSE": 89, "STOP": 89, "DIM_DOWN": 105, "DIM_UP": 106,
        "TEMP_UP": 111, "TIME": 112, "NEXT": 115, "WEATHER": 122, "ALARM": 124,
        "LIGHT_OFF": 126, "LIGHT_ON": 126, "MESSAGE": 126, "TEMP_DOWN": 126,
        "VOLUME_DOWN": 127, "VOLUME_UP": 129, "LIST_REMINDERS": 130, "TIMER": 134,
        "SET_REMINDER": 137, "PLAY_MUSIC": 140,
    }
    bg_class_pools = {
        "glass_breaking": 79, "door_wood_knock": 97, "can_opening": 101,
        "mouse_click": 120, "door_wood_creaks": 131, "clock_alarm": 132,
        "washing_machine": 136, "wind": 136, "clock_tick": 137, "rain": 140,
        "thunderstorm": 141, "vacuum_cleaner": 141, "keyboard_typing": 142,
    }
    alloc = bts.compute_allocation(intent_pools, bg_class_pools, yt_ambient_pool=258)

    assert alloc["commands_per_intent"] == 73
    assert alloc["target_commands_total"] == 1460
    assert alloc["total"] == 2246
    assert alloc["babble_total"] == 561
    assert alloc["silence_total"] == 225
    assert alloc["babble_split"] == {
        "common_voice_negative": 187,
        "filipino_speech_corpus": 187,
        "youtube_institutional": 187,
    }
    assert alloc["silence_split"] == {"background_noise": 194, "youtube_institutional": 31}
    assert sum(alloc["esc_split"].values()) == 194
    counts = sorted(alloc["esc_split"].values())
    assert counts == [14] + [15] * 12


def test_compute_allocation_synthetic_small_scale():
    intent_pools = {"A": 6, "B": 9, "C": 7, "D": 6, "E": 8}
    bg_class_pools = {"c1": 10, "c2": 10, "c3": 9}
    alloc = bts.compute_allocation(intent_pools, bg_class_pools, yt_ambient_pool=12)

    assert alloc["commands_per_intent"] == 6
    assert alloc["target_commands_total"] == 30
    assert alloc["total"] == 46
    assert alloc["babble_total"] == 11
    assert alloc["silence_total"] == 5
    assert alloc["babble_split"] == {
        "common_voice_negative": 4,
        "filipino_speech_corpus": 4,
        "youtube_institutional": 3,
    }
    assert alloc["silence_split"] == {"background_noise": 4, "youtube_institutional": 1}
    assert alloc["esc_split"] == {"c1": 2, "c2": 1, "c3": 1}


def test_compute_allocation_esc_class_pool_exhaustion():
    intent_pools = {"A": 100, "B": 100}
    bg_class_pools = {"tiny": 1, "big": 500}
    with pytest.raises(bts.PoolExhaustedError):
        bts.compute_allocation(intent_pools, bg_class_pools, yt_ambient_pool=500)


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


def test_build_clean_candidates_parses_filename_tokens(tmp_path):
    clean_dir = tmp_path / "sanitized" / "clean"
    for name in ["CALL_s01_p1_v1.wav", "CALL_s01_p1_v2.wav", "CALL_s02_p1_v1.wav"]:
        _write_wav(clean_dir / "CALL" / name, 16000, 16000)  # 1.0s

    cands = bts.build_clean_candidates(clean_dir, "CALL")
    assert len(cands) == 3
    by_name = {c.filename: c for c in cands}
    assert by_name["CALL_s01_p1_v1.wav"].group_id == "s01"
    assert by_name["CALL_s02_p1_v1.wav"].group_id == "s02"
    assert all(abs(c.duration - 1.0) < 1e-6 for c in cands)
    assert all(c.needs_resample is False for c in cands)
    assert all(c.source_dataset == "sanitized_clean" for c in cands)


def test_build_clean_candidates_rejects_bad_filename(tmp_path):
    clean_dir = tmp_path / "sanitized" / "clean"
    _write_wav(clean_dir / "CALL" / "not_a_valid_name.wav", 16000, 100)
    with pytest.raises(ValueError):
        bts.build_clean_candidates(clean_dir, "CALL")


def _write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_cv_candidates_uses_csv_dictreader_for_comma_bearing_fields(tmp_path):
    source_dir = tmp_path / "common_voice_negative"
    fieldnames = ["filename", "accent", "duration", "transcript", "sentence", "client_id", "source_file"]
    rows = [
        {
            "filename": "cv1.wav",
            "accent": "philippines",
            "duration": "1.218",
            "transcript": "Hello, world, this has commas",
            "sentence": "the visual acuity is astonishing, apparently",
            "client_id": "client-a",
            "source_file": "raw1.mp3",
        }
    ]
    _write_manifest(source_dir / "manifest.csv", fieldnames, rows)
    _write_wav(source_dir / "audio" / "cv1.wav", 48000, 48000)

    cands = bts.build_cv_candidates(source_dir)
    assert len(cands) == 1
    c = cands[0]
    assert c.group_id == "client-a"
    assert c.needs_resample is True
    assert c.orig_sample_rate == 48000
    assert c.duration == pytest.approx(1.218)
    assert c.source_dataset == "common_voice_negative"


def test_build_fsc_candidates_speaker_group_id_with_comma_bearing_sentence(tmp_path):
    source_dir = tmp_path / "filipino_speech_corpus"
    fieldnames = ["filename", "sentence", "duration", "speech_type", "source_file", "speaker_id", "gender", "age_group"]
    rows = [
        {
            "filename": "f1.wav",
            "sentence": "manggugulo, kumusta ka",
            "duration": "1.2",
            "speech_type": "machine",
            "source_file": "",
            "speaker_id": "94",
            "gender": "male",
            "age_group": "20-27",
        }
    ]
    _write_manifest(source_dir / "manifest.csv", fieldnames, rows)
    _write_wav(source_dir / "audio" / "f1.wav", 16000, 16000)

    cands = bts.build_fsc_candidates(source_dir)
    assert len(cands) == 1
    assert cands[0].group_id == "94"
    assert cands[0].needs_resample is False
    assert cands[0].orig_sample_rate == 16000


def test_build_yt_candidates_resolves_nested_bucket_path(tmp_path):
    source_dir = tmp_path / "youtube_institutional"
    fieldnames = ["filename", "bucket", "video_id", "start_s", "end_s", "duration", "transcript"]
    rows = [
        {"filename": "vid1_babble_0.wav", "bucket": "babble", "video_id": "vid1", "start_s": "0", "end_s": "1.5", "duration": "1.5", "transcript": ""},
        {"filename": "vid1_ambient_0.wav", "bucket": "ambient", "video_id": "vid1", "start_s": "0", "end_s": "1.5", "duration": "1.5", "transcript": ""},
    ]
    _write_manifest(source_dir / "manifest.csv", fieldnames, rows)
    _write_wav(source_dir / "clips" / "babble" / "vid1_babble_0.wav", 16000, 16000)
    _write_wav(source_dir / "clips" / "ambient" / "vid1_ambient_0.wav", 16000, 16000)

    babble = bts.build_yt_candidates(source_dir, "babble")
    ambient = bts.build_yt_candidates(source_dir, "ambient")
    assert len(babble) == 1 and len(ambient) == 1
    assert babble[0].audio_path == source_dir / "clips" / "babble" / "vid1_babble_0.wav"
    assert babble[0].source_relpath == "clips/babble/vid1_babble_0.wav"
    assert babble[0].group_id == "vid1"
    assert babble[0].audio_path.is_file()


def test_build_bg_candidates_groups_by_category(tmp_path):
    source_dir = tmp_path / "background_noise"
    fieldnames = ["filename", "category", "esc_group", "fold", "duration", "rms", "source_file"]
    rows = [
        {"filename": "n1.wav", "category": "rain", "esc_group": "x", "fold": "1", "duration": "1.0", "rms": "0.1", "source_file": "src1.wav"},
        {"filename": "n2.wav", "category": "rain", "esc_group": "x", "fold": "1", "duration": "1.0", "rms": "0.1", "source_file": "src2.wav"},
        {"filename": "n3.wav", "category": "wind", "esc_group": "x", "fold": "1", "duration": "1.0", "rms": "0.1", "source_file": "src3.wav"},
    ]
    _write_manifest(source_dir / "manifest.csv", fieldnames, rows)
    for name in ["n1.wav", "n2.wav", "n3.wav"]:
        _write_wav(source_dir / "audio" / name, 16000, 16000)

    by_class = bts.build_bg_candidates(source_dir)
    assert set(by_class) == {"rain", "wind"}
    assert len(by_class["rain"]) == 2
    assert len(by_class["wind"]) == 1
    assert by_class["rain"][0].group_id == "src1.wav"


# ---------------------------------------------------------------------------
# grouped_sample
# ---------------------------------------------------------------------------


def _make_candidates(group_counts: dict[str, int]) -> list[bts.Candidate]:
    items = []
    for group, count in group_counts.items():
        for i in range(count):
            items.append(
                bts.Candidate(
                    filename=f"{group}_{i}.wav",
                    audio_path=Path(f"/nonexistent/{group}_{i}.wav"),
                    duration=1.0,
                    orig_sample_rate=16000,
                    group_id=group,
                    source_dataset="test",
                    source_relpath=f"{group}_{i}.wav",
                    needs_resample=False,
                )
            )
    return items


def test_grouped_sample_caps_per_group_when_feasible():
    items = _make_candidates({"g1": 50, "g2": 50})
    rng = __import__("random").Random(42)
    selected, max_group = grouped_sample_result = bts.grouped_sample(rng, items, 20)
    assert len(selected) == 20
    assert max_group == 10  # ceil(20/2), evenly cappable


def test_grouped_sample_falls_back_flat_when_groups_too_few():
    items = _make_candidates({"only_group": 30})
    rng = __import__("random").Random(1)
    selected, max_group = bts.grouped_sample(rng, items, 10)
    assert len(selected) == 10
    assert max_group == 10  # only one group -> cap is meaningless, all 10 come from it


def test_grouped_sample_raises_when_pool_too_small():
    items = _make_candidates({"g1": 3})
    rng = __import__("random").Random(1)
    with pytest.raises(bts.PoolExhaustedError):
        bts.grouped_sample(rng, items, 10)


def test_grouped_sample_deterministic_for_fixed_seed():
    items = _make_candidates({"g1": 20, "g2": 20, "g3": 20})
    selected_a, _ = bts.grouped_sample(__import__("random").Random(7), items, 15)
    selected_b, _ = bts.grouped_sample(__import__("random").Random(7), items, 15)
    assert [c.filename for c in selected_a] == [c.filename for c in selected_b]


# ---------------------------------------------------------------------------
# end-to-end main() on a small synthetic corpus
# ---------------------------------------------------------------------------


def _build_synthetic_corpus(root: Path) -> None:
    # matches test_compute_allocation_synthetic_small_scale's hand-verified numbers:
    # min pool 6 across 5 intents -> commands_per_intent=6, target_commands_total=30
    intents = {"A": 6, "B": 9, "C": 7, "D": 6, "E": 8}
    for intent, pool in intents.items():
        for s in range(pool):
            _write_wav(root / "sanitized" / "clean" / intent / f"{intent}_s{s:02d}_p1_v1.wav", 16000, 16640)  # 1.04s

    cv_dir = root / "common_voice_negative"
    cv_fields = ["filename", "accent", "duration", "transcript", "sentence", "client_id", "source_file"]
    cv_rows = []
    for i in range(20):
        name = f"cv_{i}.wav"
        cv_rows.append({
            "filename": name, "accent": "x", "duration": "1.5",
            "transcript": "hi, there", "sentence": "a sentence, with a comma",
            "client_id": f"client{i % 5}", "source_file": "raw.mp3",
        })
        _write_wav(cv_dir / "audio" / name, 48000, int(48000 * 1.5))
    _write_manifest(cv_dir / "manifest.csv", cv_fields, cv_rows)

    fsc_dir = root / "filipino_speech_corpus"
    fsc_fields = ["filename", "sentence", "duration", "speech_type", "source_file", "speaker_id", "gender", "age_group"]
    fsc_rows = []
    for i in range(20):
        name = f"fsc_{i}.wav"
        fsc_rows.append({
            "filename": name, "sentence": "kumusta, ka", "duration": "1.4",
            "speech_type": "machine", "source_file": "", "speaker_id": f"spk{i % 5}",
            "gender": "male", "age_group": "20-27",
        })
        _write_wav(fsc_dir / "audio" / name, 16000, int(16000 * 1.4))
    _write_manifest(fsc_dir / "manifest.csv", fsc_fields, fsc_rows)

    yt_dir = root / "youtube_institutional"
    yt_fields = ["filename", "bucket", "video_id", "start_s", "end_s", "duration", "transcript"]
    yt_rows = []
    for i in range(20):
        name = f"vidA_babble_{i}.wav"
        yt_rows.append({"filename": name, "bucket": "babble", "video_id": "vidA", "start_s": str(i), "end_s": str(i + 1.5), "duration": "1.5", "transcript": ""})
        _write_wav(yt_dir / "clips" / "babble" / name, 16000, int(16000 * 1.5))
    for i in range(12):
        name = f"vidA_ambient_{i}.wav"
        yt_rows.append({"filename": name, "bucket": "ambient", "video_id": "vidA", "start_s": str(i), "end_s": str(i + 1.5), "duration": "1.5", "transcript": ""})
        _write_wav(yt_dir / "clips" / "ambient" / name, 16000, int(16000 * 1.5))
    _write_manifest(yt_dir / "manifest.csv", yt_fields, yt_rows)

    bg_dir = root / "background_noise"
    bg_fields = ["filename", "category", "esc_group", "fold", "duration", "rms", "source_file"]
    bg_rows = []
    for cls, pool in {"c1": 10, "c2": 10, "c3": 9}.items():
        for i in range(pool):
            name = f"{cls}_{i}.wav"
            bg_rows.append({"filename": name, "category": cls, "esc_group": "g", "fold": "1", "duration": "1.6", "rms": "0.1", "source_file": name})
            _write_wav(bg_dir / "audio" / name, 16000, int(16000 * 1.6))
    _write_manifest(bg_dir / "manifest.csv", bg_fields, bg_rows)


def test_main_end_to_end_synthetic_corpus_matches_allocation(tmp_path):
    corpus_root = tmp_path / "corpus"
    _build_synthetic_corpus(corpus_root)
    out_dir = tmp_path / "test_set"

    rc = bts.main(["--seed", "123", "--corpus-root", str(corpus_root), "--out-dir", str(out_dir)])
    assert rc == 0

    manifest_path = out_dir / "manifest.csv"
    assert manifest_path.is_file()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # matches test_compute_allocation_synthetic_small_scale's hand-verified numbers
    assert len(rows) == 46
    assert sum(1 for r in rows if r["bucket"] == "target_commands") == 30
    assert sum(1 for r in rows if r["bucket"] == "babble") == 11
    assert sum(1 for r in rows if r["bucket"] == "silence") == 5

    for r in rows:
        emitted = out_dir / r["path"]
        assert emitted.is_file()
        with wave.open(str(emitted), "rb") as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2

    assert (out_dir / "summary.md").is_file()
    summary_text = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "seed" in summary_text.lower()
    assert "CC-BY-NC-SA-4.0" in summary_text
    assert "single-persona" in summary_text.lower() or "p1" in summary_text.lower()
    assert "distinct video_ids" in summary_text


def test_main_seed_determinism_byte_identical_manifest(tmp_path):
    corpus_root = tmp_path / "corpus"
    _build_synthetic_corpus(corpus_root)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    rc_a = bts.main(["--seed", "99", "--corpus-root", str(corpus_root), "--out-dir", str(out_a)])
    rc_b = bts.main(["--seed", "99", "--corpus-root", str(corpus_root), "--out-dir", str(out_b)])
    assert rc_a == 0 and rc_b == 0

    manifest_a = (out_a / "manifest.csv").read_bytes()
    manifest_b = (out_b / "manifest.csv").read_bytes()
    assert manifest_a == manifest_b


def test_main_different_seeds_can_diverge(tmp_path):
    corpus_root = tmp_path / "corpus"
    _build_synthetic_corpus(corpus_root)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    bts.main(["--seed", "1", "--corpus-root", str(corpus_root), "--out-dir", str(out_a)])
    bts.main(["--seed", "2", "--corpus-root", str(corpus_root), "--out-dir", str(out_b)])

    rows_a = (out_a / "manifest.csv").read_bytes()
    rows_b = (out_b / "manifest.csv").read_bytes()
    assert rows_a != rows_b


def test_main_no_duplicate_filenames_emitted(tmp_path):
    corpus_root = tmp_path / "corpus"
    _build_synthetic_corpus(corpus_root)
    out_dir = tmp_path / "test_set"
    bts.main(["--seed", "5", "--corpus-root", str(corpus_root), "--out-dir", str(out_dir)])

    with (out_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        filenames = [r["filename"] for r in csv.DictReader(f)]
    assert len(filenames) == len(set(filenames))


def test_main_pool_exhaustion_exits_nonzero_with_actionable_message(tmp_path, capsys):
    corpus_root = tmp_path / "corpus"
    _build_synthetic_corpus(corpus_root)
    # shrink one babble source below what the allocator will request
    import shutil as _shutil
    fsc_dir = corpus_root / "filipino_speech_corpus"
    for wav in sorted((fsc_dir / "audio").glob("*.wav"))[2:]:
        wav.unlink()
    with (fsc_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    kept_names = {p.name for p in (fsc_dir / "audio").glob("*.wav")}
    rows = [r for r in rows if r["filename"] in kept_names]
    _write_manifest(fsc_dir / "manifest.csv", fieldnames, rows)

    out_dir = tmp_path / "test_set"
    rc = bts.main(["--seed", "1", "--corpus-root", str(corpus_root), "--out-dir", str(out_dir)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "filipino_speech_corpus" in err
    assert "exceeds pool" in err
