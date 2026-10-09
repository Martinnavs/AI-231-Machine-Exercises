"""Unit tests for `me2_voicegen.vcm.vcmx_merge` (Phase 2 of feature
`optionb-v3-vcmx`). Fast, synthetic fixtures under `tmp_path` with real
16 kHz WAVs -- no real `raw_datasets/`/`out/conversions/v2/optionb-v3/`
tree is touched here (see the `@pytest.mark.slow` test at the bottom of
this file for the real-data load path).
"""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import pytest

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm import vcmx_merge as vx
from me2_voicegen.vcm.dataset import VCMDataset, collate_fn
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.text import normalize_text, resolve_transcript

REQUIRED_SR = 16000

# ---------------------------------------------------------------------------
# Pure-function tests: alias normalization, label map, UNKNOWN handling.
# ---------------------------------------------------------------------------


def test_alias_normalize_lowercases_and_strips_punctuation():
    assert vx.alias_normalize("Play  Music!") == "play music"


def test_alias_normalize_curly_apostrophe():
    assert vx.alias_normalize("What’s the weather") == "what's the weather"


def test_alias_normalize_percent():
    assert vx.alias_normalize("Set the brightness to 60%") == "set the brightness to 60 percent"


def test_alias_normalize_ampm_dotted():
    assert vx.alias_normalize("Wake me up at 6 a.m.") == "wake me up at 6 am"


def test_alias_normalize_hhmm00():
    assert vx.alias_normalize("Set an alarm for 6:00 am") == "set an alarm for 6 am"


def test_alias_normalize_digit_ampm_no_space():
    assert vx.alias_normalize("alarm 6am") == "alarm 6 am"


@pytest.mark.parametrize(
    "text,intent,slot_value",
    [
        ("Set the temperature to 22 degrees.", "TEMPERATURE", {"DEGREES": "22 degrees"}),
        ("wake me up at six am", "ALARM", {"ALARM_TIME": "6 AM"}),
        ("Wake me up at 6 a.m.", "ALARM", {"ALARM_TIME": "6 AM"}),
    ],
)
def test_alias_normalize_digit_and_word_slot_forms_accepted(text, intent, slot_value):
    normalized = vx.alias_normalize(text)
    result = OPTIONB_GRAMMAR.accepts(normalized)
    assert result == [(intent, slot_value)]


def test_label_map_known_entries():
    assert vx.LABEL_MAP == {
        "MEDIA_NEXT": "NEXT",
        "MEDIA_PAUSE": "PAUSE",
        "MEDIA_STOP": "STOP",
        "SET_TIMER": "TIMER",
        "SET_ALARM": "ALARM",
        "SET_TEMPERATURE": "TEMPERATURE",
        "LIGHT_DIM": "BRIGHTNESS",
    }


def test_normalize_unknown_text_at_and_ampersand():
    assert vx.normalize_unknown_text("call me @ home & work") == "call me at home and work"


def test_normalize_unknown_text_drops_hash():
    assert vx.normalize_unknown_text("ticket #42") is None


def test_normalize_unknown_text_drops_digit_overflow():
    assert vx.normalize_unknown_text("order 101 items") is None


def test_normalize_unknown_text_keeps_digit_at_ceiling():
    assert vx.normalize_unknown_text("order 100 items") == "order 100 items"


def test_normalize_unknown_text_no_digits_passthrough():
    assert vx.normalize_unknown_text("what's happening around the world") == "what's happening around the world"


# ---------------------------------------------------------------------------
# Step 1: select_b_rows -- both drop reasons, UNKNOWN, SILENCE.
# ---------------------------------------------------------------------------


def _b_row(row_id, class_, original_source, speaker_id="spk1", source_dataset="FluentSpeechCommands", original_or_augmented="original"):
    return {
        "id": row_id,
        "filepath": f"audio/{row_id}.wav",
        "class": class_,
        "speaker_id": speaker_id,
        "source_dataset": source_dataset,
        "source_type": "real",
        "original_or_augmented": original_or_augmented,
        "augmentation_type": "",
        "original_source": original_source,
        "original_id": original_source,
        "group_id": original_source,
    }


def test_select_b_rows_keeps_exact_in_grammar_command():
    b_rows = [_b_row("B1", "PLAY_MUSIC", "a/1")]
    a_text = {"a/1": "Play music"}
    result = vx.select_b_rows(b_rows, a_text)
    assert len(result.kept) == 1
    kept = result.kept[0]
    assert kept.bucket == "target_commands"
    assert kept.label == "PLAY_MUSIC"
    assert kept.transcript == "play music"
    assert not result.drop_counts


def test_select_b_rows_label_mapped_command_kept():
    b_rows = [_b_row("B1", "MEDIA_NEXT", "a/1")]
    a_text = {"a/1": "Next song"}
    result = vx.select_b_rows(b_rows, a_text)
    assert result.kept[0].label == "NEXT"


def test_select_b_rows_drops_out_of_grammar():
    b_rows = [_b_row("B1", "PLAY_MUSIC", "a/1")]
    a_text = {"a/1": "play the music please"}
    result = vx.select_b_rows(b_rows, a_text)
    assert result.kept == []
    assert result.drop_counts["out_of_grammar"] == 1


def test_select_b_rows_drops_accepted_other_intent():
    # "pause" is grammar-accepted, but under the PAUSE intent, not STOP.
    b_rows = [_b_row("B1", "MEDIA_STOP", "a/1")]
    a_text = {"a/1": "Pause"}
    result = vx.select_b_rows(b_rows, a_text)
    assert result.kept == []
    assert result.drop_counts["accepted_other_intent"] == 1


def test_select_b_rows_unknown_kept_with_symbol_handling():
    b_rows = [_b_row("B1", "UNKNOWN", "a/1")]
    a_text = {"a/1": "email me @ home & at work"}
    result = vx.select_b_rows(b_rows, a_text)
    assert len(result.kept) == 1
    assert result.kept[0].bucket == "babble"
    assert result.kept[0].label == "unknown"
    assert result.kept[0].transcript == "email me at home and at work"


def test_select_b_rows_unknown_dropped_for_hash():
    b_rows = [_b_row("B1", "UNKNOWN", "a/1")]
    a_text = {"a/1": "ticket #7"}
    result = vx.select_b_rows(b_rows, a_text)
    assert result.kept == []
    assert result.drop_counts["unknown_contains_hash"] == 1


def test_select_b_rows_unknown_dropped_for_digit_overflow():
    b_rows = [_b_row("B1", "UNKNOWN", "a/1")]
    a_text = {"a/1": "order 250 units"}
    result = vx.select_b_rows(b_rows, a_text)
    assert result.kept == []
    assert result.drop_counts["unknown_digit_overflow"] == 1


def test_select_b_rows_silence_forces_empty_transcript():
    b_rows = [_b_row("B1", "SILENCE", "a/1")]
    a_text = {"a/1": ""}
    result = vx.select_b_rows(b_rows, a_text)
    assert result.kept[0].bucket == "silence"
    assert result.kept[0].label == "silence"
    assert result.kept[0].transcript == ""


def test_select_b_rows_missing_a_join_dropped():
    b_rows = [_b_row("B1", "PLAY_MUSIC", "missing/path")]
    result = vx.select_b_rows(b_rows, {})
    assert result.kept == []
    assert result.drop_counts["no_a_join"] == 1


# ---------------------------------------------------------------------------
# Step 2: B speaker split -- determinism, per-family weighting.
# ---------------------------------------------------------------------------


def _synthetic_family_rows(source_dataset: str, n_speakers: int, rows_per_speaker: int) -> list[dict]:
    rows = []
    for s in range(n_speakers):
        for r in range(rows_per_speaker):
            rows.append(
                _b_row(
                    f"{source_dataset}_{s}_{r}",
                    "PLAY_MUSIC",
                    f"a/{source_dataset}_{s}_{r}",
                    speaker_id=f"spk{s}",
                    source_dataset=source_dataset,
                )
            )
    return rows


def test_assign_b_speaker_splits_is_deterministic():
    rows = _synthetic_family_rows("FAM", 20, 5)
    a = vx.assign_b_speaker_splits(rows, seed=7)
    b = vx.assign_b_speaker_splits(rows, seed=7)
    assert a == b


def test_assign_b_speaker_splits_different_seed_can_differ():
    rows = _synthetic_family_rows("FAM", 20, 5)
    a = vx.assign_b_speaker_splits(rows, seed=1)
    b = vx.assign_b_speaker_splits(rows, seed=2)
    assert a != b


def test_assign_b_speaker_splits_covers_all_three_splits_for_small_family():
    # 16 speakers, uneven row counts -- like the real Multi-Sensor family --
    # a naive hash-based split was measured to produce 0 val / 60 test for
    # this shape; the weighted greedy assignment must not degenerate to it.
    rows = _synthetic_family_rows("MultiSensorLike", 16, 60)
    assignment = vx.assign_b_speaker_splits(rows, seed=0)
    splits = {v for k, v in assignment.items() if k[0] == "MultiSensorLike"}
    assert splits == {"train", "val", "test"}


def test_assign_b_speaker_splits_speaker_disjoint_within_family():
    rows = _synthetic_family_rows("FAM", 30, 4)
    assignment = vx.assign_b_speaker_splits(rows, seed=3)
    # Each (family, speaker) key maps to exactly one split by construction
    # (it's a dict), but assert the invariant explicitly for clarity.
    keys = list(assignment)
    assert len(keys) == len(set(keys))


def test_assign_b_speaker_splits_approx_80_10_10_by_row_count():
    rows = _synthetic_family_rows("FAM", 50, 10)
    assignment = vx.assign_b_speaker_splits(rows, seed=0)
    totals = {"train": 0, "val": 0, "test": 0}
    for (family, speaker), split in assignment.items():
        totals[split] += 10  # rows_per_speaker
    total = sum(totals.values())
    assert 0.7 <= totals["train"] / total <= 0.9
    assert 0.03 <= totals["val"] / total <= 0.2
    assert 0.03 <= totals["test"] / total <= 0.2


# ---------------------------------------------------------------------------
# ESC-50 fold -> split mapping + Freesound disjointness guard.
# ---------------------------------------------------------------------------


def _esc_row(filename, fold, source_file):
    return {
        "filename": filename,
        "category": "vacuum_cleaner",
        "esc_group": "interior_domestic",
        "fold": fold,
        "duration": "1.0",
        "rms": "0.2",
        "source_file": source_file,
    }


def test_assign_esc50_splits_fold_mapping():
    rows = [
        _esc_row("a.wav", "1", "1-100-A-1.wav"),
        _esc_row("b.wav", "2", "2-200-A-1.wav"),
        _esc_row("c.wav", "3", "3-300-A-1.wav"),
        _esc_row("d.wav", "4", "4-400-A-1.wav"),
        _esc_row("e.wav", "5", "5-500-A-1.wav"),
    ]
    fid_splits = vx.assign_esc50_splits(rows)
    assert fid_splits == {"100": "train", "200": "train", "300": "train", "400": "val", "500": "test"}


def test_assign_esc50_splits_resolves_cross_fold_freesound_id_by_majority():
    # Same real-data shape found against the actual pool: one Freesound
    # recording split across val (fold 4) and test (fold 5) folds.
    rows = [
        _esc_row("a.wav", "4", "4-999-A-1.wav"),
        _esc_row("b.wav", "5", "5-999-A-2.wav"),
        _esc_row("c.wav", "5", "5-999-A-3.wav"),
    ]
    fid_splits = vx.assign_esc50_splits(rows)
    # 2 test-fold chunks outvote 1 val-fold chunk -> resolved to test.
    assert fid_splits == {"999": "test"}


def test_freesound_id_parses_second_dash_field():
    assert vx.freesound_id("1-100210-A-36.wav") == "100210"


def test_check_freesound_disjoint_raises_on_violation():
    rows = [
        {"source_dataset": "background_noise", "group_id": "999", "split": "val"},
        {"source_dataset": "background_noise", "group_id": "999", "split": "test"},
    ]
    with pytest.raises(vx.VCMXMergeError, match="Freesound"):
        vx._check_freesound_disjoint(rows)


def test_check_freesound_disjoint_passes_when_disjoint():
    rows = [
        {"source_dataset": "background_noise", "group_id": "999", "split": "val"},
        {"source_dataset": "background_noise", "group_id": "111", "split": "test"},
    ]
    vx._check_freesound_disjoint(rows)  # must not raise


def test_check_group_id_disjoint_covers_every_family_with_a_group_id():
    # Widened check (post-`resolve_m0_group_splits` fix): a cross-split
    # group_id in ANY family with a real group_id -- including the two
    # M0-carried reject-probe families that used to be exempted -- must
    # raise.
    rows = [
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "train"},
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "val"},
    ]
    with pytest.raises(vx.VCMXMergeError, match="group_id"):
        vx._check_group_id_disjoint(rows)

    rows2 = [
        {"source_dataset": "youtube_institutional", "group_id": "g1", "split": "train"},
        {"source_dataset": "youtube_institutional", "group_id": "g1", "split": "test"},
    ]
    with pytest.raises(vx.VCMXMergeError, match="group_id"):
        vx._check_group_id_disjoint(rows2)

    rows3 = [
        {"source_dataset": "vcm_balanced", "group_id": "g1", "split": "train"},
        {"source_dataset": "vcm_balanced", "group_id": "g1", "split": "val"},
    ]
    with pytest.raises(vx.VCMXMergeError, match="group_id"):
        vx._check_group_id_disjoint(rows3)


def test_check_group_id_disjoint_ignores_falsy_group_id():
    # common_voice_negative's group_id is not a real per-speaker id (same
    # convention as evaluate.classify_speaker_group: falsy group_id is
    # excluded from grouping, not treated as one shared group).
    rows = [
        {"source_dataset": "common_voice_negative", "group_id": "", "split": "train"},
        {"source_dataset": "common_voice_negative", "group_id": "", "split": "val"},
        {"source_dataset": "common_voice_negative", "group_id": "", "split": "test"},
    ]
    vx._check_group_id_disjoint(rows)  # must not raise


def test_check_group_id_disjoint_ref_prefix_exempt_across_splits():
    # Phase-1 scoped exception: `ref_`-prefixed group ids (accent-balance
    # references voices, usable in every split) are exempt from the
    # cross-split disjointness check, in any source_dataset family.
    rows = [
        {"source_dataset": "fil50_persona", "group_id": "ref_tagalog1", "split": "train"},
        {"source_dataset": "fil50_persona", "group_id": "ref_tagalog1", "split": "test"},
        {"source_dataset": "optionb", "group_id": "ref_stella", "split": "val"},
        {"source_dataset": "optionb", "group_id": "ref_stella", "split": "train"},
    ]
    vx._check_group_id_disjoint(rows)  # must not raise


@pytest.mark.parametrize("gid", ["fsc_1", "s68", "g1"])
def test_check_group_id_disjoint_still_raises_for_non_ref_ids(gid):
    # Regression guard: the exemption is scoped to the ref_ prefix only.
    rows = [
        {"source_dataset": "filipino_speech_corpus", "group_id": gid, "split": "train"},
        {"source_dataset": "filipino_speech_corpus", "group_id": gid, "split": "test"},
    ]
    with pytest.raises(vx.VCMXMergeError, match="group_id"):
        vx._check_group_id_disjoint(rows)


def test_resolve_m0_group_splits_majority_and_tiebreak():
    rows = [
        # 1-1 tie between train and val -> tie-break picks val.
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "train"},
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "val"},
        # Clear majority (train) -> majority wins, no tie-break needed.
        {"source_dataset": "youtube_institutional", "group_id": "g2", "split": "train"},
        {"source_dataset": "youtube_institutional", "group_id": "g2", "split": "train"},
        {"source_dataset": "youtube_institutional", "group_id": "g2", "split": "val"},
        # Already single-split -> untouched.
        {"source_dataset": "youtube_institutional", "group_id": "g3", "split": "test"},
    ]
    resolved = vx.resolve_m0_group_splits(rows)
    g1_splits = {r["split"] for r in resolved if r["group_id"] == "g1"}
    g2_splits = {r["split"] for r in resolved if r["group_id"] == "g2"}
    g3_splits = {r["split"] for r in resolved if r["group_id"] == "g3"}
    assert g1_splits == {"val"}
    assert g2_splits == {"train"}
    assert g3_splits == {"test"}


def test_resolve_m0_group_splits_tiebreak_prefers_test_over_train_when_no_val():
    rows = [
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "train"},
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "test"},
    ]
    resolved = vx.resolve_m0_group_splits(rows)
    assert {r["split"] for r in resolved} == {"test"}


def test_resolve_m0_group_splits_excludes_common_voice_negative():
    rows = [
        {"source_dataset": "common_voice_negative", "group_id": "", "split": "train"},
        {"source_dataset": "common_voice_negative", "group_id": "", "split": "val"},
    ]
    resolved = vx.resolve_m0_group_splits(rows)
    assert [r["split"] for r in resolved] == ["train", "val"]


def test_resolve_m0_group_splits_result_is_disjoint_by_construction():
    rows = [
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "train"},
        {"source_dataset": "filipino_speech_corpus", "group_id": "g1", "split": "val"},
        {"source_dataset": "youtube_institutional", "group_id": "g2", "split": "val"},
        {"source_dataset": "youtube_institutional", "group_id": "g2", "split": "test"},
        {"source_dataset": "youtube_institutional", "group_id": "g2", "split": "test"},
    ]
    resolved = vx.resolve_m0_group_splits(rows)
    vx._check_group_id_disjoint(resolved)  # must not raise


# ---------------------------------------------------------------------------
# CTC feasibility.
# ---------------------------------------------------------------------------


def test_ctc_feasible_true_for_ample_duration():
    ids = alphabet.encode("call")  # has one adjacent repeat ("ll")
    assert vx._ctc_feasible(2.0, ids) is True


def test_ctc_feasible_false_for_too_short_duration():
    ids = alphabet.encode("call")
    assert vx._ctc_feasible(0.001, ids) is False


def test_check_ctc_feasible_raises_on_infeasible_row():
    row = {
        "filename": "x.wav",
        "source_dataset": "vcm_balanced",
        "transcript": "call",
        "duration": "0.001",
    }
    with pytest.raises(vx.VCMXMergeError, match="CTC-infeasible"):
        vx._check_ctc_feasible([row])


def test_check_ctc_feasible_passes_for_feasible_row():
    row = {
        "filename": "x.wav",
        "source_dataset": "vcm_balanced",
        "transcript": "call",
        "duration": "2.0",
    }
    vx._check_ctc_feasible([row])  # must not raise


# ---------------------------------------------------------------------------
# End-to-end `build()` over synthetic fixtures (real tiny 16 kHz WAVs).
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def vcmx_fixture(tmp_path, vcm_wav_factory):
    """Builds a full synthetic fixture tree: an M0 (optionb-v3-like)
    manifest, VCM Dataset B (balanced) metadata + audio, and an ESC-50
    pool -- everything `vcmx_merge.build()` needs, with real tiny 16 kHz
    WAVs so file-existence/WAV-header checks exercise real code paths."""
    root = tmp_path

    # --- M0 (optionb-v3) manifest ---
    m0_dir = root / "optionb-v3"
    m0_audio = m0_dir / "audio"
    m0_wav_1 = vcm_wav_factory(m0_audio / "call.wav", duration_s=1.0)
    m0_wav_2 = vcm_wav_factory(m0_audio / "bg.wav", duration_s=1.0)
    m0_rows = [
        {
            "filename": "call.wav",
            "path": "audio/call.wav",
            "bucket": "target_commands",
            "label": "CALL",
            "duration": "1.000000",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "optionb",
            "source_relpath": "CALL/call.wav",
            "group_id": "s1",
            "split": "train",
            "transcript": "Call",
        },
        {
            "filename": "bg.wav",
            "path": "../test_set/audio/background_noise/bg.wav",
            "bucket": "silence",
            "label": "silence",
            "duration": "1.000000",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "background_noise",
            "source_relpath": "bg.wav",
            "group_id": "bgclip",
            "split": "val",
            "transcript": "",
        },
    ]
    # No audio file is created for the "../test_set/..." bg.wav row: it has
    # `source_dataset == "background_noise"`, which `build_m0_schema_rows`
    # drops before any file-existence check ever runs (Step 2b) -- so its
    # path is deliberately never resolved.
    optionb_manifest = m0_dir / "manifest.csv"
    _write_csv(optionb_manifest, m0_rows)

    # --- B (VCM_BALANCED) ---
    b_root = root / "VCM_BALANCED"
    b_audio = b_root / "audio"
    b_meta = root / "VCM_BALANCED_METADATA"

    b_wav_play = vcm_wav_factory(b_audio / "VCM_BAL_000001.wav", duration_s=1.0)
    b_wav_next = vcm_wav_factory(b_audio / "VCM_BAL_000002.wav", duration_s=1.0)
    b_wav_bad = vcm_wav_factory(b_audio / "VCM_BAL_000003.wav", duration_s=1.0)
    b_wav_unknown = vcm_wav_factory(b_audio / "VCM_BAL_000004.wav", duration_s=1.0)
    b_wav_silence = vcm_wav_factory(b_audio / "VCM_BAL_000005.wav", duration_s=1.0)
    b_wav_aug = vcm_wav_factory(b_audio / "VCM_BAL_000006.wav", duration_s=1.0)

    b_train_rows = [
        _b_row("VCM_BAL_000001", "PLAY_MUSIC", "audio/A_000001.wav", speaker_id="spk1", source_dataset="FluentSpeechCommands"),
        _b_row("VCM_BAL_000002", "MEDIA_NEXT", "audio/A_000002.wav", speaker_id="spk2", source_dataset="Multi-Sensor"),
        _b_row("VCM_BAL_000003", "PLAY_MUSIC", "audio/A_000003.wav", speaker_id="spk1", source_dataset="FluentSpeechCommands"),
        _b_row("VCM_BAL_000004", "UNKNOWN", "audio/A_000004.wav", speaker_id="spk3", source_dataset="SLURP"),
        _b_row("VCM_BAL_000005", "SILENCE", "audio/A_000005.wav", speaker_id="doing_the_dishes.wav", source_dataset="GSC_background_noise"),
        _b_row(
            "VCM_BAL_000006",
            "PLAY_MUSIC",
            "audio/A_000001.wav",
            speaker_id="spk1",
            source_dataset="FluentSpeechCommands",
            original_or_augmented="augmented",
        ),
    ]
    _write_csv(b_meta / "manifests" / "train.csv", b_train_rows)

    provenance_rows = [
        {
            "final_filename": f"VCM_BAL_00000{i}.wav",
            "original_vcm_master_path": f"audio/A_00000{i}.wav",
            "original_dataset": "x",
            "original_id": "x",
            "class": "x",
            "speaker": "x",
            "original_or_augmented": "original",
            "augmentation_type": "",
            "sha256": "x",
            "duration": "1.000000",
            "sample_rate": "16000",
            "channels": "1",
            "bit_depth": "16",
        }
        for i in range(1, 7)
    ]
    _write_csv(b_meta / "manifests" / "audio_provenance.csv", provenance_rows)

    # --- A (VCM_MASTER_METADATA) ---
    a_meta = root / "VCM_MASTER_METADATA"
    a_rows = [
        {"filepath": "audio/A_000001.wav", "label": "PLAY_MUSIC", "speaker": "x", "source": "real", "split": "train", "transcript": "Play music", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
        {"filepath": "audio/A_000002.wav", "label": "MEDIA_NEXT", "speaker": "x", "source": "real", "split": "train", "transcript": "Next song", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
        {"filepath": "audio/A_000003.wav", "label": "PLAY_MUSIC", "speaker": "x", "source": "real", "split": "train", "transcript": "play the music please", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
        {"filepath": "audio/A_000004.wav", "label": "UNKNOWN", "speaker": "x", "source": "real", "split": "train", "transcript": "email me @ home", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
        {"filepath": "audio/A_000005.wav", "label": "SILENCE", "speaker": "x", "source": "real", "split": "train", "transcript": "", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
    ]
    _write_csv(a_meta / "manifests" / "all.csv", a_rows)

    # --- ESC-50 pool ---
    esc_root = root / "background_noise"
    esc_audio = esc_root / "audio"
    esc_rows = []
    fold_by_idx = ["1", "2", "3", "4", "5"]
    for i, fold in enumerate(fold_by_idx, start=1):
        filename = f"clip{i}.wav"
        vcm_wav_factory(esc_audio / filename, duration_s=1.317)
        esc_rows.append(_esc_row(filename, fold, f"{fold}-{1000 + i}-A-1.wav"))
    esc_manifest = esc_root / "manifest.csv"
    _write_csv(esc_manifest, esc_rows)

    return {
        "root": root,
        "optionb_manifest": optionb_manifest,
        "b_root": b_root,
        "b_metadata_root": b_meta,
        "a_metadata_root": a_meta,
        "esc50_manifest": esc_manifest,
    }


def test_build_end_to_end_produces_both_manifests_and_reports(vcmx_fixture, tmp_path):
    out_treatment = tmp_path / "out-treatment"
    out_control = tmp_path / "out-control"

    vx.build(
        optionb_manifest=vcmx_fixture["optionb_manifest"],
        b_root=vcmx_fixture["b_root"],
        b_metadata_root=vcmx_fixture["b_metadata_root"],
        a_metadata_root=vcmx_fixture["a_metadata_root"],
        esc50_manifest=vcmx_fixture["esc50_manifest"],
        out_treatment=out_treatment,
        out_control=out_control,
        seed=0,
        dry_run=False,
    )

    assert (out_treatment / "manifest.csv").is_file()
    assert (out_control / "manifest.csv").is_file()
    assert (out_treatment / "report.md").is_file()
    assert (out_control / "report.md").is_file()

    with (out_treatment / "manifest.csv").open(newline="", encoding="utf-8") as f:
        treatment_rows = list(csv.DictReader(f))

    # Every referenced file must resolve from the output dir.
    for row in treatment_rows:
        assert (out_treatment / row["path"]).resolve().is_file(), row["path"]

    labels = {(r["source_dataset"], r["bucket"], r["label"]) for r in treatment_rows}
    assert ("vcm_balanced", "target_commands", "PLAY_MUSIC") in labels
    assert ("vcm_balanced", "target_commands", "NEXT") in labels
    assert ("vcm_balanced", "babble", "unknown") in labels
    assert ("vcm_balanced", "silence", "silence") in labels
    assert ("background_noise", "silence", "silence") in labels
    assert ("optionb", "target_commands", "CALL") in labels
    # The M0 background_noise row was dropped, replaced by the ESC-50 pool.
    assert not any(r["source_dataset"] == "background_noise" and r["filename"] == "bg.wav" for r in treatment_rows)


def test_build_b_silence_forced_to_train(vcmx_fixture, tmp_path):
    out_treatment = tmp_path / "out-treatment"
    out_control = tmp_path / "out-control"
    vx.build(
        optionb_manifest=vcmx_fixture["optionb_manifest"],
        b_root=vcmx_fixture["b_root"],
        b_metadata_root=vcmx_fixture["b_metadata_root"],
        a_metadata_root=vcmx_fixture["a_metadata_root"],
        esc50_manifest=vcmx_fixture["esc50_manifest"],
        out_treatment=out_treatment,
        out_control=out_control,
        seed=0,
        dry_run=False,
    )
    with (out_treatment / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    b_silence = [r for r in rows if r["source_dataset"] == "vcm_balanced" and r["bucket"] == "silence"]
    assert b_silence
    assert all(r["split"] == "train" for r in b_silence)


def test_build_augmented_row_dropped_when_speaker_not_train(tmp_path, vcm_wav_factory):
    """A minimal fixture where the single FluentSpeechCommands speaker is
    forced (via a 1-speaker family) to land in a non-train split, so its
    augmented row must be dropped rather than kept."""
    root = tmp_path
    m0_dir = root / "optionb-v3"
    vcm_wav_factory(m0_dir / "audio" / "call.wav", duration_s=1.0)
    _write_csv(
        m0_dir / "manifest.csv",
        [
            {
                "filename": "call.wav", "path": "audio/call.wav", "bucket": "target_commands",
                "label": "CALL", "duration": "1.000000", "sample_rate": "16000", "resampled": "False",
                "source_dataset": "optionb", "source_relpath": "x", "group_id": "s1", "split": "train",
                "transcript": "Call",
            }
        ],
    )

    b_root = root / "VCM_BALANCED"
    vcm_wav_factory(b_root / "audio" / "VCM_BAL_000001.wav", duration_s=1.0)
    vcm_wav_factory(b_root / "audio" / "VCM_BAL_000002.wav", duration_s=1.0)
    b_meta = root / "VCM_BALANCED_METADATA"
    _write_csv(
        b_meta / "manifests" / "train.csv",
        [
            _b_row("VCM_BAL_000001", "PLAY_MUSIC", "audio/A_000001.wav", speaker_id="only_speaker", source_dataset="OneSpeakerFamily"),
            _b_row(
                "VCM_BAL_000002", "PLAY_MUSIC", "audio/A_000001.wav", speaker_id="only_speaker",
                source_dataset="OneSpeakerFamily", original_or_augmented="augmented",
            ),
        ],
    )
    _write_csv(
        b_meta / "manifests" / "audio_provenance.csv",
        [
            {
                "final_filename": f"VCM_BAL_00000{i}.wav", "original_vcm_master_path": "x", "original_dataset": "x",
                "original_id": "x", "class": "x", "speaker": "x", "original_or_augmented": "x", "augmentation_type": "",
                "sha256": "x", "duration": "1.000000", "sample_rate": "16000", "channels": "1", "bit_depth": "16",
            }
            for i in (1, 2)
        ],
    )
    a_meta = root / "VCM_MASTER_METADATA"
    _write_csv(
        a_meta / "manifests" / "all.csv",
        [
            {"filepath": "audio/A_000001.wav", "label": "PLAY_MUSIC", "speaker": "x", "source": "real", "split": "train", "transcript": "Play music", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
        ],
    )
    esc_root = root / "background_noise"
    vcm_wav_factory(esc_root / "audio" / "clip1.wav", duration_s=1.0)
    _write_csv(esc_root / "manifest.csv", [_esc_row("clip1.wav", "1", "1-9-A-1.wav")])

    # A single speaker in its own family always lands in "train" by the
    # greedy assignment (only one bucket to fill), so instead assert the
    # *rule*: with only one speaker, both the original and augmented rows
    # follow that speaker's split (both kept, both "train") -- and
    # separately unit-test the drop rule directly via
    # `build_b_schema_rows` with a forced non-train split (below).
    out_treatment = root / "out-treatment"
    out_control = root / "out-control"
    vx.build(
        optionb_manifest=m0_dir / "manifest.csv",
        b_root=b_root,
        b_metadata_root=b_meta,
        a_metadata_root=a_meta,
        esc50_manifest=esc_root / "manifest.csv",
        out_treatment=out_treatment,
        out_control=out_control,
        seed=0,
        dry_run=False,
    )
    with (out_treatment / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    b_rows = [r for r in rows if r["source_dataset"] == "vcm_balanced"]
    assert len(b_rows) == 2
    assert all(r["split"] == "train" for r in b_rows)


def test_build_b_schema_rows_drops_augmented_row_outside_train():
    selection = vx.SelectionResult(
        kept=[
            vx.SelectedBRow(
                b_row=_b_row(
                    "VCM_BAL_000002", "PLAY_MUSIC", "audio/A_000001.wav",
                    speaker_id="spk1", source_dataset="FAM", original_or_augmented="augmented",
                ),
                bucket="target_commands", label="PLAY_MUSIC", transcript="play music",
            )
        ]
    )
    provenance = {
        "VCM_BAL_000002.wav": {
            "duration": "1.0", "sample_rate": "16000", "channels": "1", "bit_depth": "16",
        }
    }
    rows, drop_counts = vx.build_b_schema_rows(
        selection,
        b_root=Path("/tmp/does-not-need-to-exist"),
        b_provenance=provenance,
        speaker_splits={("FAM", "spk1"): "val"},
        out_dir=Path("/tmp/out"),
    )
    assert rows == []
    assert drop_counts["augmented_dropped_val_test"] == 1


# ---------------------------------------------------------------------------
# Control/treatment val/test equality + path rewriting.
# ---------------------------------------------------------------------------


def test_build_control_val_test_identical_to_treatment(vcmx_fixture, tmp_path):
    out_treatment = tmp_path / "out-treatment"
    out_control = tmp_path / "out-control"
    vx.build(
        optionb_manifest=vcmx_fixture["optionb_manifest"],
        b_root=vcmx_fixture["b_root"],
        b_metadata_root=vcmx_fixture["b_metadata_root"],
        a_metadata_root=vcmx_fixture["a_metadata_root"],
        esc50_manifest=vcmx_fixture["esc50_manifest"],
        out_treatment=out_treatment,
        out_control=out_control,
        seed=0,
        dry_run=False,
    )
    with (out_treatment / "manifest.csv").open(newline="", encoding="utf-8") as f:
        treatment_rows = list(csv.DictReader(f))
    with (out_control / "manifest.csv").open(newline="", encoding="utf-8") as f:
        control_rows = list(csv.DictReader(f))

    t_valtest = [r for r in treatment_rows if r["split"] in ("val", "test")]
    c_valtest = [r for r in control_rows if r["split"] in ("val", "test")]
    assert t_valtest == c_valtest

    control_vcmb_train = [r for r in control_rows if r["source_dataset"] == "vcm_balanced" and r["split"] == "train"]
    assert control_vcmb_train == []


def test_build_resolves_leaking_m0_groups_identically_across_treatment_and_control(tmp_path, vcm_wav_factory):
    """A leaking M0 filipino_speech_corpus group_id (present under two
    different splits in the raw M0 manifest, mirroring the real
    `test_set/`-inherited leak) must resolve to exactly one split, and
    both treatment and control manifests must agree on which one."""
    root = tmp_path
    m0_dir = root / "optionb-v3"
    vcm_wav_factory(m0_dir / "audio" / "call.wav", duration_s=1.0)
    vcm_wav_factory(m0_dir / "audio" / "fsc_a.wav", duration_s=1.0)
    vcm_wav_factory(m0_dir / "audio" / "fsc_b.wav", duration_s=1.0)
    _write_csv(
        m0_dir / "manifest.csv",
        [
            {
                "filename": "call.wav", "path": "audio/call.wav", "bucket": "target_commands",
                "label": "CALL", "duration": "1.000000", "sample_rate": "16000", "resampled": "False",
                "source_dataset": "optionb", "source_relpath": "x", "group_id": "s1", "split": "train",
                "transcript": "Call",
            },
            {
                "filename": "fsc_a.wav", "path": "audio/fsc_a.wav", "bucket": "babble",
                "label": "unknown", "duration": "1.000000", "sample_rate": "16000", "resampled": "False",
                "source_dataset": "filipino_speech_corpus", "source_relpath": "x", "group_id": "leak1",
                "split": "train", "transcript": "",
            },
            {
                "filename": "fsc_b.wav", "path": "audio/fsc_b.wav", "bucket": "babble",
                "label": "unknown", "duration": "1.000000", "sample_rate": "16000", "resampled": "False",
                "source_dataset": "filipino_speech_corpus", "source_relpath": "x", "group_id": "leak1",
                "split": "val", "transcript": "",
            },
        ],
    )

    b_root = root / "VCM_BALANCED"
    vcm_wav_factory(b_root / "audio" / "VCM_BAL_000001.wav", duration_s=1.0)
    b_meta = root / "VCM_BALANCED_METADATA"
    _write_csv(
        b_meta / "manifests" / "train.csv",
        [_b_row("VCM_BAL_000001", "PLAY_MUSIC", "audio/A_000001.wav", speaker_id="spk1", source_dataset="FAM")],
    )
    _write_csv(
        b_meta / "manifests" / "audio_provenance.csv",
        [
            {
                "final_filename": "VCM_BAL_000001.wav", "original_vcm_master_path": "x", "original_dataset": "x",
                "original_id": "x", "class": "x", "speaker": "x", "original_or_augmented": "x", "augmentation_type": "",
                "sha256": "x", "duration": "1.000000", "sample_rate": "16000", "channels": "1", "bit_depth": "16",
            }
        ],
    )
    a_meta = root / "VCM_MASTER_METADATA"
    _write_csv(
        a_meta / "manifests" / "all.csv",
        [
            {"filepath": "audio/A_000001.wav", "label": "PLAY_MUSIC", "speaker": "x", "source": "real", "split": "train", "transcript": "Play music", "original_dataset": "x", "original_id": "x", "original_filepath": "x"},
        ],
    )
    esc_root = root / "background_noise"
    vcm_wav_factory(esc_root / "audio" / "clip1.wav", duration_s=1.0)
    _write_csv(esc_root / "manifest.csv", [_esc_row("clip1.wav", "1", "1-9-A-1.wav")])

    out_treatment = root / "out-treatment"
    out_control = root / "out-control"
    vx.build(
        optionb_manifest=m0_dir / "manifest.csv",
        b_root=b_root,
        b_metadata_root=b_meta,
        a_metadata_root=a_meta,
        esc50_manifest=esc_root / "manifest.csv",
        out_treatment=out_treatment,
        out_control=out_control,
        seed=0,
        dry_run=False,
    )

    with (out_treatment / "manifest.csv").open(newline="", encoding="utf-8") as f:
        treatment_rows = list(csv.DictReader(f))
    with (out_control / "manifest.csv").open(newline="", encoding="utf-8") as f:
        control_rows = list(csv.DictReader(f))

    t_leak_splits = {r["split"] for r in treatment_rows if r["group_id"] == "leak1"}
    c_leak_splits = {r["split"] for r in control_rows if r["group_id"] == "leak1"}
    # 1-1 tie between train and val -> resolved to val (tie-break).
    assert t_leak_splits == {"val"}
    assert c_leak_splits == {"val"}

    t_valtest = [r for r in treatment_rows if r["split"] in ("val", "test")]
    c_valtest = [r for r in control_rows if r["split"] in ("val", "test")]
    assert t_valtest == c_valtest


def test_rel_path_rewrites_m0_relative_probe_path():
    m0_dir = Path("/repo/out/conversions/v2/optionb-v3")
    out_dir = Path("/repo/out/conversions/v2/optionb-v3-vcmx")
    m0_rows = [
        {
            "filename": "bg.wav",
            "path": "../test_set/audio/background_noise/bg.wav",
            "bucket": "silence",
            "label": "silence",
            "duration": "1.0",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "youtube_institutional",
            "source_relpath": "x",
            "group_id": "g",
            "split": "val",
            "transcript": "",
        }
    ]
    rows = vx.build_m0_schema_rows(m0_rows, m0_dir, out_dir)
    assert rows[0]["path"] == "../test_set/audio/background_noise/bg.wav"


def test_rel_path_rewrites_m0_audio_subdir_path():
    m0_dir = Path("/repo/out/conversions/v2/optionb-v3")
    out_dir = Path("/repo/out/conversions/v2/optionb-v3-vcmx")
    m0_rows = [
        {
            "filename": "call.wav", "path": "audio/CALL/call.wav", "bucket": "target_commands",
            "label": "CALL", "duration": "1.0", "sample_rate": "16000", "resampled": "False",
            "source_dataset": "optionb", "source_relpath": "x", "group_id": "s1", "split": "train",
            "transcript": "Call",
        }
    ]
    rows = vx.build_m0_schema_rows(m0_rows, m0_dir, out_dir)
    assert rows[0]["path"] == "../optionb-v3/audio/CALL/call.wav"


def test_build_m0_schema_rows_drops_background_noise_rows():
    m0_dir = Path("/repo/out/conversions/v2/optionb-v3")
    out_dir = Path("/repo/out/conversions/v2/optionb-v3-vcmx")
    m0_rows = [
        {
            "filename": "bg.wav", "path": "../test_set/x.wav", "bucket": "silence", "label": "silence",
            "duration": "1.0", "sample_rate": "16000", "resampled": "False",
            "source_dataset": "background_noise", "source_relpath": "x", "group_id": "g", "split": "val",
            "transcript": "",
        }
    ]
    assert vx.build_m0_schema_rows(m0_rows, m0_dir, out_dir) == []


# ---------------------------------------------------------------------------
# Real-data slow test: both built manifests load under VCMDataset for every
# split; every row resolves and encodes; collate_fn runs one batch/split.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_TREATMENT_MANIFEST = PROJECT_ROOT / "out" / "conversions" / "v2" / "optionb-v3-vcmx" / "manifest.csv"
REAL_CONTROL_MANIFEST = PROJECT_ROOT / "out" / "conversions" / "v2" / "optionb-v3-vcmx-control" / "manifest.csv"


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_TREATMENT_MANIFEST.exists() and REAL_CONTROL_MANIFEST.exists()),
    reason="real vcmx manifests not built locally (run `vcmx_merge build` first)",
)
@pytest.mark.parametrize("manifest_path", [REAL_TREATMENT_MANIFEST, REAL_CONTROL_MANIFEST])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_real_manifests_load_under_vcmdataset_every_split(manifest_path, split):
    ds = VCMDataset(manifest_path=manifest_path, split=split)
    assert len(ds) > 0

    for i, row in enumerate(ds.rows):
        text = resolve_transcript(row)
        if text is not None:
            alphabet.encode(normalize_text(text))

    batch = [ds[i] for i in range(min(4, len(ds)))]
    collated = collate_fn(batch)
    assert collated["features"].shape[0] == len(batch)
    assert collated["input_lengths"].shape[0] == len(batch)
