"""Fast, CPU-only, checkpoint-free tests for `vcm.evaluate`.

Ticket 05's own Execution Log flagged this module as untested by any
dedicated unit test file (its correctness was only checked via the one
real evaluation run against the real checkpoint) -- this file is
ticket 08's independent gap-audit closing that gap. Covers the pure
threshold-sweep/confusion/false-accept arithmetic, `render_markdown`'s
grammar-coverage framing, and -- the two paths ticket 05 explicitly left
unexercised -- `main()`'s graceful skip when Task 07's slot-eval-set
manifest is absent, and its non-fatal catch-and-continue when that
manifest is present but evaluation against it fails.

Uses Task 02's shared `vcm_stub_model_factory`/`vcm_fake_manifest_factory`
fixtures throughout; no real checkpoint or GPU is ever touched here.
"""

from __future__ import annotations

import csv
import json

import pytest

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.evaluate import (
    RowResult,
    choose_operating_threshold,
    confusion_counts,
    false_accept_stats,
    main,
    render_markdown,
    slot_accuracy_breakdown,
    speaker_group_breakdown,
    sweep_margins,
    sweep_thresholds,
)
from me2_voicegen.vcm.optiona.grammar import TOY_GRAMMAR
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

CALL_IDS = alphabet.encode("call")


# ---------------------------------------------------------------------------
# --grammar selection (ticket 04, D11): GRAMMAR_REGISTRY / _intent_labels_for
# and main()'s CLI wiring for spec/toy/optionb.
# ---------------------------------------------------------------------------


def test_intent_labels_for_spec_and_toy_use_intent_phrases():
    from me2_voicegen.vcm.evaluate import _intent_labels_for
    from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

    assert _intent_labels_for("spec") == sorted(INTENT_PHRASES)
    assert _intent_labels_for("toy") == sorted(INTENT_PHRASES)


def test_intent_labels_for_optionb_uses_optionb_grammars_own_intents():
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
    from me2_voicegen.vcm.evaluate import _intent_labels_for
    from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

    labels = _intent_labels_for("optionb")
    assert labels == sorted({intent for _, intent, _ in OPTIONB_GRAMMAR.all_phrases()})
    # Option B's label vocabulary is disjoint from vcm's -- this must not
    # silently fall back to INTENT_PHRASES.
    assert labels != sorted(INTENT_PHRASES)
    assert "BRIGHTNESS" in labels
    assert "DIM_UP" not in labels


# ---------------------------------------------------------------------------
# sweep_thresholds / choose_operating_threshold
# ---------------------------------------------------------------------------


def _row(bucket, label, intent, confidence, group_id=None, source_dataset=None, index=0, slots=None):
    return RowResult(
        index=index,
        bucket=bucket,
        label=label,
        text="",
        intent=intent,
        confidence=confidence,
        group_id=group_id,
        source_dataset=source_dataset,
        slots={} if slots is None else slots,
    )


def test_sweep_thresholds_computes_target_and_false_accept_rates():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1),
        _row("target_commands", "CALL", "CALL", -0.4),
        _row("target_commands", "STOP", None, None),  # never reached any grammar terminal
        _row("babble", "n/a", "CALL", -0.2),
        _row("silence", "n/a", None, None),
    ]
    sweep = sweep_thresholds(results, thresholds=(0.0, -0.3, -1.0))

    by_threshold = {row["threshold"]: row for row in sweep}
    assert by_threshold[0.0]["target_accept_rate"] == pytest.approx(0.0)
    assert by_threshold[0.0]["false_accept_rate"] == pytest.approx(0.0)

    assert by_threshold[-0.3]["target_accept_rate"] == pytest.approx(1 / 3)
    assert by_threshold[-0.3]["false_accept_rate"] == pytest.approx(0.5)
    assert by_threshold[-0.3]["youden_j"] == pytest.approx(1 / 3 - 0.5)

    assert by_threshold[-1.0]["target_accept_rate"] == pytest.approx(2 / 3)
    assert by_threshold[-1.0]["false_accept_rate"] == pytest.approx(0.5)


def test_sweep_thresholds_empty_bucket_yields_zero_rate_not_zerodiv():
    results = [_row("target_commands", "CALL", "CALL", -0.1)]
    sweep = sweep_thresholds(results, thresholds=(0.0,))
    assert sweep[0]["false_accept_rate"] == 0.0
    assert sweep[0]["n_reject_probes"] == 0


def test_choose_operating_threshold_picks_max_youden_j():
    sweep = [
        {"threshold": 0.0, "youden_j": 0.2},
        {"threshold": -0.1, "youden_j": 0.9},
        {"threshold": -0.2, "youden_j": 0.5},
    ]
    chosen = choose_operating_threshold(sweep)
    assert chosen["threshold"] == -0.1


def test_choose_operating_threshold_ties_break_toward_least_permissive():
    # least-permissive == highest / least-negative threshold among ties.
    sweep = [
        {"threshold": -0.5, "youden_j": 0.7},
        {"threshold": -0.1, "youden_j": 0.7},
        {"threshold": -0.9, "youden_j": 0.7},
    ]
    chosen = choose_operating_threshold(sweep)
    assert chosen["threshold"] == -0.1


# ---------------------------------------------------------------------------
# RowResult's incomplete-prefix-gate diagnostic fields (R2-2) / sweep_margins
# ---------------------------------------------------------------------------


def test_row_result_incomplete_gate_fields_default_to_none():
    row = _row("target_commands", "CALL", "CALL", -0.1)
    assert row.incomplete_prefix is None
    assert row.incomplete_gap is None
    assert row.command_raw_score is None
    assert row.incomplete_raw_score is None


def test_row_result_accepts_incomplete_gate_fields():
    row = RowResult(
        index=0,
        bucket="target_commands",
        label="CALL",
        text="call",
        intent="CALL",
        confidence=-0.1,
        incomplete_prefix="cal",
        incomplete_gap=0.5,
        command_raw_score=-1.0,
        incomplete_raw_score=-1.5,
    )
    assert row.incomplete_prefix == "cal"
    assert row.incomplete_gap == 0.5
    assert row.command_raw_score == -1.0
    assert row.incomplete_raw_score == -1.5


def _margin_row(bucket, confidence, incomplete_gap, index=0):
    return RowResult(
        index=index,
        bucket=bucket,
        label="CALL",
        text="",
        intent="CALL" if confidence is not None else None,
        confidence=confidence,
        incomplete_gap=incomplete_gap,
    )


def test_sweep_margins_computes_rates_by_construction():
    results = [
        _margin_row("target_commands", -0.1, 1.0, index=0),
        _margin_row("target_commands", -0.1, 0.5, index=1),
        _margin_row("target_commands", -0.1, 0.2, index=2),
        _margin_row("target_commands", -0.1, None, index=3),
        _margin_row("babble", -0.1, 0.1, index=4),
        _margin_row("silence", -0.1, 0.9, index=5),
    ]

    sweep = sweep_margins(results, margins=(0.0, 0.6), threshold=-0.3)
    by_margin = {row["margin"]: row for row in sweep}

    # margin=0.0: every row's incomplete_gap >= 0.0 (or None), so nothing is
    # rejected by the margin gate -- all 4 target rows and both reject-probe
    # rows pass through unaffected.
    assert by_margin[0.0]["target_accept_rate"] == pytest.approx(1.0)
    assert by_margin[0.0]["false_accept_rate"] == pytest.approx(1.0)
    assert by_margin[0.0]["incomplete_prefix_reject_rate"] == pytest.approx(0.0)

    # margin=0.6: rows with gap 0.5 and 0.2 (2 of 4 target rows) are
    # rejected by the margin gate; gap=None (index 3) is never rejected;
    # gap=1.0 (index 0) passes. -> 2/4 target rows accepted.
    assert by_margin[0.6]["target_accept_rate"] == pytest.approx(2 / 4)
    assert by_margin[0.6]["incomplete_prefix_reject_rate"] == pytest.approx(2 / 4)
    # babble row's gap=0.1 < 0.6 -> rejected; silence row's gap=0.9 >= 0.6 -> accepted.
    assert by_margin[0.6]["false_accept_rate"] == pytest.approx(0.5)


def test_sweep_margins_empty_bucket_yields_zero_rate_not_zerodiv():
    results = [_margin_row("target_commands", -0.1, 1.0)]
    sweep = sweep_margins(results, margins=(0.0,), threshold=-0.3)
    assert sweep[0]["false_accept_rate"] == 0.0
    assert sweep[0]["incomplete_prefix_reject_rate"] == pytest.approx(0.0)


def test_sweep_margins_no_competing_prefix_beam_never_rejected():
    row = _margin_row("target_commands", -0.1, None)
    sweep = sweep_margins([row], margins=(0.0, 5.0, 1000.0), threshold=-0.3)
    assert all(entry["target_accept_rate"] == pytest.approx(1.0) for entry in sweep)
    assert all(entry["incomplete_prefix_reject_rate"] == pytest.approx(0.0) for entry in sweep)


def test_sweep_margins_rejects_input_already_decoded_with_gate_enabled():
    results = [
        _margin_row("target_commands", -0.1, 1.0, index=0),
        RowResult(
            index=1,
            bucket="target_commands",
            label="CALL",
            text="",
            intent=None,
            confidence=None,
            rejection_reason="incomplete_prefix",
        ),
    ]
    with pytest.raises(ValueError):
        sweep_margins(results, margins=(0.0,), threshold=-0.3)


def test_sweep_margins_valid_input_with_rejection_reason_none_is_unaffected():
    results = [
        _margin_row("target_commands", -0.1, 1.0, index=0),
        _margin_row("target_commands", -0.1, 0.5, index=1),
    ]
    assert all(r.rejection_reason is None for r in results)
    sweep = sweep_margins(results, margins=(0.0, 0.6), threshold=-0.3)
    by_margin = {row["margin"]: row for row in sweep}
    assert by_margin[0.0]["target_accept_rate"] == pytest.approx(1.0)
    assert by_margin[0.6]["target_accept_rate"] == pytest.approx(0.5)


def test_sweep_margins_exact_gap_margin_tie_is_accepted():
    # decoder.py's own gate rejects on strict `<` (incomplete_gap <
    # required_command_margin), so an exact tie (gap == margin) must NOT be
    # rejected. sweep_margins's `_margin_ok` uses `>=`, matching that -- but
    # no existing test exercised gap == margin exactly (prior cases only use
    # margins that are strictly above or below every fixture gap).
    row = _margin_row("target_commands", -0.1, 0.6)
    sweep = sweep_margins([row], margins=(0.6,), threshold=-0.3)
    assert sweep[0]["target_accept_rate"] == pytest.approx(1.0)
    assert sweep[0]["incomplete_prefix_reject_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# confusion_counts / false_accept_stats
# ---------------------------------------------------------------------------


def test_confusion_counts_only_covers_target_commands_bucket():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1),
        _row("target_commands", "CALL", "STOP", -0.1),  # confusion
        _row("target_commands", "STOP", None, None),  # rejected
        _row("babble", "n/a", "CALL", -0.1),  # excluded from confusion table
    ]
    counts = confusion_counts(results, threshold=-0.5)
    assert counts["CALL"] == {"CALL": 1, "STOP": 1}
    assert counts["STOP"] == {"REJECTED": 1}
    assert "n/a" not in counts


def test_confusion_counts_below_threshold_counts_as_rejected():
    results = [_row("target_commands", "CALL", "CALL", -5.0)]
    counts = confusion_counts(results, threshold=-0.1)
    assert counts["CALL"] == {"REJECTED": 1}


def test_false_accept_stats_empty_bucket_reports_rate_none_not_zero():
    stats = false_accept_stats([], threshold=0.0, bucket="babble")
    assert stats == {"n": 0, "false_accepts": 0, "rate": None}


def test_false_accept_stats_counts_accepts_at_threshold():
    results = [
        _row("babble", "n/a", "CALL", -0.1),
        _row("babble", "n/a", "CALL", -5.0),
        _row("babble", "n/a", None, None),
    ]
    stats = false_accept_stats(results, threshold=-1.0, bucket="babble")
    assert stats == {"n": 3, "false_accepts": 1, "rate": pytest.approx(1 / 3)}


# ---------------------------------------------------------------------------
# classify_speaker_group / speaker_group_breakdown / _wilson_interval
# ---------------------------------------------------------------------------


def test_classify_speaker_group_filipino_id():
    from me2_voicegen.vcm.evaluate import classify_speaker_group

    assert classify_speaker_group("optionb", "s100") == "filipino_reference"
    assert classify_speaker_group("optionb", "s68") == "filipino_reference"
    assert classify_speaker_group("optionb", "s80") == "filipino_reference"
    assert classify_speaker_group("optionb", "s89") == "filipino_reference"
    assert classify_speaker_group("optionb", "s90") == "filipino_reference"


def test_classify_speaker_group_foreign_id():
    from me2_voicegen.vcm.evaluate import classify_speaker_group

    assert classify_speaker_group("optionb", "s1") == "foreign_reference"
    assert classify_speaker_group("optionb", "s67") == "foreign_reference"
    assert classify_speaker_group("optionb", "s81") == "foreign_reference"


def test_classify_speaker_group_non_optionb_source_is_none():
    from me2_voicegen.vcm.evaluate import classify_speaker_group

    assert classify_speaker_group("background_noise", "s100") is None
    assert classify_speaker_group("youtube_institutional", "abc123") is None
    assert classify_speaker_group("filipino_speech_corpus", "007") is None
    assert classify_speaker_group("common_voice_negative", "") is None


def test_classify_speaker_group_fil50_persona_is_always_filipino():
    """feature accent-balance-fil50 (.scratch/accent-balance-fil50/tickets/
    00-RECAP.md T6): every fil50_persona row is unconditionally
    filipino_reference, including with a non-s<N>-shaped group_id (a
    voice_id like fsc_94 or ref_tagalog3, not an optionb speaker ID) and
    with no group_id at all."""
    from me2_voicegen.vcm.evaluate import classify_speaker_group

    assert classify_speaker_group("fil50_persona", "fsc_94") == "filipino_reference"
    assert classify_speaker_group("fil50_persona", "ref_tagalog3") == "filipino_reference"
    assert classify_speaker_group("fil50_persona", None) == "filipino_reference"


def test_classify_speaker_group_malformed_or_empty_group_id_is_none():
    from me2_voicegen.vcm.evaluate import classify_speaker_group

    assert classify_speaker_group("optionb", None) is None
    assert classify_speaker_group("optionb", "") is None
    assert classify_speaker_group("optionb", "not_a_speaker_id") is None
    assert classify_speaker_group("optionb", "sXY") is None


def test_wilson_interval_brackets_point_estimate():
    from me2_voicegen.vcm.evaluate import _wilson_interval

    interval = _wilson_interval(90, 100)
    assert interval is not None
    lower, upper = interval
    assert lower < 0.9 < upper


def test_wilson_interval_zero_n_is_none():
    from me2_voicegen.vcm.evaluate import _wilson_interval

    assert _wilson_interval(0, 0) is None


def test_speaker_group_breakdown_counts_and_rates():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s100", source_dataset="optionb"),
        _row("target_commands", "CALL", "STOP", -0.1, group_id="s100", source_dataset="optionb"),
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s1", source_dataset="optionb"),
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s2", source_dataset="optionb"),
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s3", source_dataset="optionb"),
    ]
    breakdown = speaker_group_breakdown(results, threshold=-0.5)
    assert breakdown is not None
    assert breakdown["filipino_reference"]["n"] == 2
    assert breakdown["filipino_reference"]["n_accepted"] == 2
    assert breakdown["filipino_reference"]["n_exact_correct"] == 1
    assert breakdown["filipino_reference"]["exact_accuracy"] == pytest.approx(0.5)
    assert breakdown["foreign_reference"]["n"] == 3
    assert breakdown["foreign_reference"]["exact_accuracy"] == pytest.approx(1.0)
    assert breakdown["n_unclassified"] == 0
    assert breakdown["exact_accuracy_gap_foreign_minus_filipino"] == pytest.approx(0.5)


def test_speaker_group_breakdown_none_when_no_optionb_target_rows():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1, group_id="0", source_dataset="sanitized_clean"),
        _row("babble", "n/a", "CALL", -0.1, group_id="0", source_dataset="common_voice_negative"),
    ]
    assert speaker_group_breakdown(results, threshold=-0.5) is None


def test_speaker_group_breakdown_counts_unclassifiable_row():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s100", source_dataset="optionb"),
        _row("target_commands", "CALL", "CALL", -0.1, group_id="bogus", source_dataset="optionb"),
    ]
    breakdown = speaker_group_breakdown(results, threshold=-0.5)
    assert breakdown is not None
    assert breakdown["n_unclassified"] == 1
    assert breakdown["filipino_reference"]["n"] == 1
    assert breakdown["foreign_reference"]["n"] == 0
    assert breakdown["foreign_reference"]["exact_accuracy"] is None
    assert breakdown["foreign_reference"]["exact_accuracy_ci95"] is None


def test_speaker_group_breakdown_excludes_non_target_buckets():
    results = [
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s100", source_dataset="optionb"),
        _row("babble", "n/a", "CALL", -0.1, group_id="s1", source_dataset="optionb"),
        _row("silence", "n/a", None, None, group_id="s2", source_dataset="optionb"),
    ]
    breakdown = speaker_group_breakdown(results, threshold=-0.5)
    assert breakdown is not None
    assert breakdown["filipino_reference"]["n"] == 1
    assert breakdown["foreign_reference"]["n"] == 0


# ---------------------------------------------------------------------------
# _true_slots_for_row / slot_accuracy_breakdown: does the model get the
# SLOT VALUE right (e.g. which hour an ALARM clip named), not just the
# intent? `exact_accuracy`/`confusion_counts` only ever compare
# `intent == label` and cannot answer this.
# ---------------------------------------------------------------------------


def _optionb_row(transcript, label):
    return {"transcript": transcript, "label": label, "source_dataset": "optionb"}


def test_true_slots_for_row_parses_real_transcript():
    from me2_voicegen.vcm.evaluate import _true_slots_for_row

    row = _optionb_row("Alarm 6 AM", "ALARM")
    assert _true_slots_for_row(row, OPTIONB_GRAMMAR) == {"ALARM_TIME": "6 AM"}


def test_true_slots_for_row_digit_and_word_form_agree():
    from me2_voicegen.vcm.evaluate import _true_slots_for_row

    digit_form = _true_slots_for_row(_optionb_row("Wake me up at 9 PM", "ALARM"), OPTIONB_GRAMMAR)
    word_form = _true_slots_for_row(_optionb_row("Wake me up at nine PM", "ALARM"), OPTIONB_GRAMMAR)
    assert digit_form == word_form == {"ALARM_TIME": "9 PM"}


def test_true_slots_for_row_none_when_unparseable():
    from me2_voicegen.vcm.evaluate import _true_slots_for_row

    row = _optionb_row("this is not a real command at all", "ALARM")
    assert _true_slots_for_row(row, OPTIONB_GRAMMAR) is None


def test_true_slots_for_row_none_when_transcript_matches_a_different_intent():
    from me2_voicegen.vcm.evaluate import _true_slots_for_row

    # A row whose (mislabeled, for this test) `label` disagrees with what
    # its own transcript actually parses to -- must not silently return
    # the wrong intent's slots.
    row = _optionb_row("lights on", "ALARM")
    assert _true_slots_for_row(row, OPTIONB_GRAMMAR) is None


def test_slot_accuracy_breakdown_scores_intent_correct_but_slot_wrong():
    raw_rows = [
        _optionb_row("Alarm 6 AM", "ALARM"),  # index 0: intent+slot both right
        _optionb_row("Alarm 8 AM", "ALARM"),  # index 1: intent right, slot wrong
        _optionb_row("Alarm 9 PM", "ALARM"),  # index 2: intent wrong entirely
    ]
    results = [
        _row("target_commands", "ALARM", "ALARM", -0.1, source_dataset="optionb", index=0, slots={"ALARM_TIME": "6 AM"}),
        _row("target_commands", "ALARM", "ALARM", -0.1, source_dataset="optionb", index=1, slots={"ALARM_TIME": "9 PM"}),
        _row("target_commands", "ALARM", "STOP", -0.1, source_dataset="optionb", index=2, slots={}),
    ]
    breakdown = slot_accuracy_breakdown(results, raw_rows, OPTIONB_GRAMMAR, threshold=-0.5)
    assert breakdown is not None
    assert breakdown["n_slot_bearing_target_rows"] == 3
    assert breakdown["n_unparseable_ground_truth"] == 0
    assert breakdown["n_intent_correct"] == 2  # index 2's wrong intent excluded
    assert breakdown["n_intent_and_slots_correct"] == 1  # only index 0
    assert breakdown["slot_exact_match_rate_given_intent_correct"] == pytest.approx(0.5)
    assert breakdown["per_slot_name_accuracy"]["ALARM_TIME"] == {"n": 2, "n_correct": 1, "accuracy": pytest.approx(0.5)}


def test_slot_accuracy_breakdown_none_when_no_optionb_target_rows():
    raw_rows = [{"transcript": "set alarm", "label": "ALARM", "source_dataset": "sanitized_clean"}]
    results = [_row("target_commands", "ALARM", "ALARM", -0.1, source_dataset="sanitized_clean", index=0)]
    assert slot_accuracy_breakdown(results, raw_rows, OPTIONB_GRAMMAR, threshold=-0.5) is None


def test_slot_accuracy_breakdown_none_when_grammar_has_no_slotted_intents():
    from me2_voicegen.common.grammar_core import compile_grammar, literal, with_intent

    no_slot_grammar = compile_grammar("NO_SLOTS", {"$CMD_STOP": with_intent("STOP", literal("stop"))})
    raw_rows = [_optionb_row("stop", "STOP")]
    results = [_row("target_commands", "STOP", "STOP", -0.1, source_dataset="optionb", index=0)]
    assert slot_accuracy_breakdown(results, raw_rows, no_slot_grammar, threshold=-0.5) is None


# ---------------------------------------------------------------------------
# render_markdown: grammar-coverage framing must actually appear, not just
# render without crashing.
# ---------------------------------------------------------------------------


def _minimal_grammar_section(label: str) -> dict:
    sweep = sweep_thresholds(
        [_row("target_commands", "CALL", "CALL", -0.1), _row("babble", "n/a", None, None)],
        thresholds=(0.0, -0.1),
    )
    chosen = choose_operating_threshold(sweep)
    return {
        "grammar": label,
        "threshold_sweep_on_val": sweep,
        "chosen_operating_threshold": chosen["threshold"],
        "chosen_operating_point_val_stats": chosen,
        "test_split": {
            "n_target_commands": 1,
            "n_accepted": 1,
            "n_exact_correct": 1,
            "accept_rate": 1.0,
            "exact_accuracy": 1.0,
            "per_intent_confusion": {"CALL": {"CALL": 1}},
            "false_accept_rate_babble": {"n": 0, "false_accepts": 0, "rate": None},
            "false_accept_rate_silence": {"n": 0, "false_accepts": 0, "rate": None},
        },
    }


def test_render_markdown_includes_spec_grammar_coverage_note():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("SPEC_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "Grammar-coverage note" in md
    assert "12 of the 20" in md
    assert "not because the model failed" in md


def test_render_markdown_toy_grammar_section_has_no_coverage_note():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("TOY_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "Grammar-coverage note" not in md


def test_render_markdown_speaker_group_breakdown_absent_when_no_key():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("TOY_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "Speaker-group breakdown" not in md


def test_render_markdown_speaker_group_breakdown_present_when_key_set():
    section = _minimal_grammar_section("OPTIONB_GRAMMAR")
    results = [
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s100", source_dataset="optionb"),
        _row("target_commands", "CALL", "CALL", -0.1, group_id="s1", source_dataset="optionb"),
    ]
    section["test_split"]["speaker_group_breakdown"] = speaker_group_breakdown(results, threshold=-0.5)

    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [section],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "### Speaker-group breakdown" in md
    assert "single held-out speaker (`s100`, 180 clips)" in md
    assert "13 of the 16 Filipino speakers (2,146 clips)" in md
    assert "filipino_reference" in md
    assert "foreign_reference" in md


def test_render_markdown_slot_accuracy_absent_when_no_key():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("TOY_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "### Slot accuracy" not in md


def test_render_markdown_slot_accuracy_present_when_key_set():
    section = _minimal_grammar_section("OPTIONB_GRAMMAR")
    raw_rows = [_optionb_row("Alarm 6 AM", "ALARM")]
    results = [
        _row("target_commands", "ALARM", "ALARM", -0.1, source_dataset="optionb", index=0, slots={"ALARM_TIME": "6 AM"}),
    ]
    section["test_split"]["slot_accuracy"] = slot_accuracy_breakdown(results, raw_rows, OPTIONB_GRAMMAR, threshold=-0.5)

    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [section],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }
    md = render_markdown(report)
    assert "### Slot accuracy" in md
    assert "ALARM_TIME" in md
    assert "1/1" in md


def test_render_markdown_reports_slot_eval_skip_reason():
    report = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "grammar_sections": [_minimal_grammar_section("TOY_GRAMMAR")],
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": "out/vcm/slot_eval/manifest.csv not found",
    }
    md = render_markdown(report)
    assert "Skipped: out/vcm/slot_eval/manifest.csv not found" in md


# ---------------------------------------------------------------------------
# evaluate_slot_eval_set: the graceful-skip contract from Task 05's own
# Execution Log ("present but malformed" path was unexercised by any
# automated test) -- exercised for real here, CPU-only, stub model.
# ---------------------------------------------------------------------------


def _write_slot_eval_manifest(tmp_path, wav_factory, *, malformed=False):
    slot_dir = tmp_path / "slot_eval"
    slot_dir.mkdir()
    manifest_path = slot_dir / "manifest.csv"
    if malformed:
        # References a wav file that was never written -- forces the real
        # torchaudio.load() call inside evaluate_slot_eval_set to raise,
        # which is exactly the "present but evaluation failed" path
        # main()'s broad except is there to catch.
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "path", "label", "sample_rate"])
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "missing.wav",
                    "path": "audio/missing.wav",
                    "label": "CALL",
                    "sample_rate": "16000",
                }
            )
        return manifest_path

    wav_factory(slot_dir / "audio" / "call_mom.wav", duration_s=0.2)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "path", "label", "sample_rate"])
        writer.writeheader()
        writer.writerow(
            {
                "filename": "call_mom.wav",
                "path": "audio/call_mom.wav",
                "label": "CALL",
                "sample_rate": "16000",
            }
        )
    return manifest_path


def test_evaluate_slot_eval_set_scores_accept_and_intent_correct(
    tmp_path, vcm_wav_factory, vcm_stub_model_factory
):
    from me2_voicegen.vcm.evaluate import evaluate_slot_eval_set
    from me2_voicegen.common.features import LogMelFeatureExtractor

    manifest_path = _write_slot_eval_manifest(tmp_path, vcm_wav_factory)
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    fe = LogMelFeatureExtractor()

    section = evaluate_slot_eval_set(
        model, fe, manifest_path, TOY_GRAMMAR, "TOY_GRAMMAR", threshold=-50.0, beam_width=50, device="cpu"
    )

    assert section["n_clips"] == 1
    assert section["n_accepted"] == 1
    assert section["n_intent_correct"] == 1
    assert "KNOWN slot-vocabulary training gap" in section["framing"]
    assert section["rows"][0]["predicted_intent"] == "CALL"


# ---------------------------------------------------------------------------
# main(): the two slot-eval-set control-flow paths ticket 05 flagged as
# untested -- graceful skip when absent, non-fatal catch when malformed.
# Monkeypatches load_checkpoint so no real checkpoint.pt is ever touched.
# ---------------------------------------------------------------------------


def _build_main_manifest(vcm_fake_manifest_factory):
    specs = []
    for split in ("val", "test"):
        specs.append(
            {
                "bucket": "target_commands",
                "source_dataset": "sanitized_clean",
                "label": "CALL",
                "split": split,
            }
        )
        specs.append(
            {
                "bucket": "babble",
                "source_dataset": "common_voice_negative",
                "label": "unknown",
                "split": split,
                "transcript": "some other speech",
            }
        )
        specs.append(
            {
                "bucket": "silence",
                "source_dataset": "background_noise",
                "label": "unknown",
                "split": split,
            }
        )
    return vcm_fake_manifest_factory(specs)


def _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory):
    import me2_voicegen.vcm.evaluate as evaluate_mod

    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    checkpoint_meta = {"preset": "default", "epoch": 1, "val_loss": 1.0, "license": "CC-BY-NC-SA-4.0"}
    monkeypatch.setattr(evaluate_mod, "load_checkpoint", lambda path, device: (model, checkpoint_meta))


def test_main_gracefully_skips_absent_slot_eval_manifest(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    out_dir = tmp_path / "eval_out"
    missing_slot_manifest = tmp_path / "does_not_exist" / "manifest.csv"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(missing_slot_manifest),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert report["slot_eval_sections"] is None
    assert "not found" in report["slot_eval_skipped_reason"]
    assert (out_dir / "metadata" / "eval_report.md").exists()
    assert "Skipped:" in (out_dir / "metadata" / "eval_report.md").read_text()


def test_main_catches_and_continues_on_malformed_slot_eval_manifest(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_wav_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)
    slot_manifest = _write_slot_eval_manifest(tmp_path, vcm_wav_factory, malformed=True)

    out_dir = tmp_path / "eval_out"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(slot_manifest),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert report["slot_eval_sections"] is None
    assert "present but evaluation failed" in report["slot_eval_skipped_reason"]


def test_main_runs_slot_eval_section_when_manifest_present_and_valid(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_wav_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)
    slot_manifest = _write_slot_eval_manifest(tmp_path, vcm_wav_factory, malformed=False)

    out_dir = tmp_path / "eval_out"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(slot_manifest),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert report["slot_eval_skipped_reason"] is None
    assert report["slot_eval_sections"] is not None
    assert len(report["slot_eval_sections"]) == 2
    for section in report["slot_eval_sections"]:
        assert section["n_clips"] == 1


def test_main_default_grammar_is_spec_toy(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    out_dir = tmp_path / "eval_out"
    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(tmp_path / "does_not_exist" / "manifest.csv"),
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    labels = [s["grammar"] for s in report["grammar_sections"]]
    assert labels == ["SPEC_GRAMMAR", "TOY_GRAMMAR"]


def _build_optionb_main_manifest(vcm_fake_manifest_factory):
    specs = []
    for split in ("val", "test"):
        specs.append(
            {
                "bucket": "target_commands",
                "source_dataset": "optionb",
                "label": "CALL",
                "split": split,
                "transcript": "Call",
            }
        )
        specs.append(
            {
                "bucket": "babble",
                "source_dataset": "common_voice_negative",
                "label": "unknown",
                "split": split,
                "transcript": "some other speech",
            }
        )
        specs.append(
            {
                "bucket": "silence",
                "source_dataset": "background_noise",
                "label": "unknown",
                "split": split,
            }
        )
    return vcm_fake_manifest_factory(specs)


def test_main_grammar_optionb_selects_optionb_grammar_and_its_own_intent_labels(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_optionb_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    out_dir = tmp_path / "eval_out"
    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            str(out_dir),
            "--slot-eval-manifest",
            str(tmp_path / "does_not_exist" / "manifest.csv"),
            "--grammar",
            "optionb",
        ]
    )

    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    assert len(report["grammar_sections"]) == 1
    section = report["grammar_sections"][0]
    assert section["grammar"] == "OPTIONB_GRAMMAR"
    assert "DIM_UP" not in section["intent_labels"]
    assert "BRIGHTNESS" in section["intent_labels"]
    # Non-degenerate sweep (D5/D11): babble+silence reject probes present.
    ts = section["test_split"]
    assert ts["n_target_commands"] == 1
    assert ts["false_accept_rate_babble"]["n"] == 1
    assert ts["false_accept_rate_silence"]["n"] == 1


def test_main_rejects_unknown_grammar_key(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    with pytest.raises(SystemExit):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--checkpoint",
                "unused.pt",
                "--out-dir",
                str(tmp_path / "eval_out"),
                "--grammar",
                "bogus",
            ]
        )


# ---------------------------------------------------------------------------
# required_command_margin plumbing (ticket 03 of
# .scratch/incomplete-grammar-rejection/tickets, docs/
# INCOMPLETE-GRAMMAR-REJECTION.md Step 3): evaluate CLI flag, decode_split
# threading, the pure rejection-count helper, and the additive report keys.
# ---------------------------------------------------------------------------


def test_evaluate_cli_exposes_required_command_margin():
    from me2_voicegen.vcm.evaluate import build_arg_parser

    assert build_arg_parser().parse_args([]).required_command_margin is None
    assert (
        build_arg_parser().parse_args(["--required-command-margin", "0.5"]).required_command_margin
        == 0.5
    )
    # negative margins are legal (a stricter gate); no range constraint here
    assert (
        build_arg_parser().parse_args(["--required-command-margin", "-0.5"]).required_command_margin
        == -0.5
    )


def test_decode_split_threads_required_command_margin(monkeypatch, vcm_fake_manifest_factory):
    import me2_voicegen.vcm.dataset as dataset_mod
    import me2_voicegen.vcm.evaluate as evaluate_mod

    from me2_voicegen.vcm.decoder import DecodeResult

    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    val_dataset = dataset_mod.VCMDataset(manifest_path, split="val", augmenter=None)

    calls: list = []

    def _recording_infer_waveform(
        model,
        feature_extractor,
        waveform,
        grammar,
        threshold,
        beam_width=50,
        device="cpu",
        required_command_margin=None,
    ):
        calls.append(required_command_margin)
        return DecodeResult(
            intent=None,
            slots={},
            text="",
            confidence=float("-inf"),
            no_match=True,
            out_of_grammar_gap=float("inf"),
        )

    monkeypatch.setattr(evaluate_mod, "infer_waveform", _recording_infer_waveform)

    for margin in (None, 0.0):
        calls.clear()
        results = evaluate_mod.decode_split(
            None,
            None,
            val_dataset,
            TOY_GRAMMAR,
            beam_width=50,
            device="cpu",
            required_command_margin=margin,
        )
        # one decode per row, margin threaded per row, no cross-row state
        assert len(results) == len(val_dataset)
        assert len(calls) == len(val_dataset)
        assert all(c == margin for c in calls)
        assert all(r.rejection_reason is None for r in results)


def test_decode_split_threads_incomplete_gate_diagnostic_fields(monkeypatch, vcm_fake_manifest_factory):
    import me2_voicegen.vcm.dataset as dataset_mod
    import me2_voicegen.vcm.evaluate as evaluate_mod

    from me2_voicegen.vcm.decoder import DecodeResult

    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    val_dataset = dataset_mod.VCMDataset(manifest_path, split="val", augmenter=None)

    def _fake_infer_waveform(
        model,
        feature_extractor,
        waveform,
        grammar,
        threshold,
        beam_width=50,
        device="cpu",
        required_command_margin=None,
    ):
        return DecodeResult(
            intent="CALL",
            slots={},
            text="call",
            confidence=-0.1,
            no_match=False,
            out_of_grammar_gap=0.0,
            rejection_reason=None,
            incomplete_prefix="cal",
            incomplete_gap=0.5,
            command_raw_score=-1.0,
            incomplete_raw_score=-1.5,
        )

    monkeypatch.setattr(evaluate_mod, "infer_waveform", _fake_infer_waveform)

    results = evaluate_mod.decode_split(
        None,
        None,
        val_dataset,
        TOY_GRAMMAR,
        beam_width=50,
        device="cpu",
        required_command_margin=None,
    )
    assert len(results) == len(val_dataset)
    for row in results:
        assert row.incomplete_prefix == "cal"
        assert row.incomplete_gap == 0.5
        assert row.command_raw_score == -1.0
        assert row.incomplete_raw_score == -1.5


def test_incomplete_prefix_rejection_counts_helper():
    from me2_voicegen.vcm.evaluate import _incomplete_prefix_rejection_counts

    def _row_with_reason(index, reason):
        return RowResult(
            index=index,
            bucket="target_commands",
            label="TIME",
            text="color",
            intent=None,
            confidence=None,
            rejection_reason=reason,
        )

    val = [
        _row("target_commands", "CALL", "CALL", -0.1),
        _row_with_reason(1, "incomplete_prefix"),
        _row_with_reason(2, None),  # rejected by the confidence gate, not the margin gate
    ]
    test = [
        _row_with_reason(0, "incomplete_prefix"),
        _row_with_reason(1, "incomplete_prefix"),
        _row("target_commands", "CALL", "CALL", -0.2),
    ]
    assert _incomplete_prefix_rejection_counts(val, test) == {"val": 1, "test": 2}


def test_render_markdown_incomplete_prefix_rejections_line_only_when_margin_set():
    section = _minimal_grammar_section("TOY_GRAMMAR")
    report_base = {
        "license_note": "CC-BY-NC-SA-4.0",
        "checkpoint_path": "out/vcm/checkpoint.pt",
        "checkpoint_meta": {"preset": "default", "epoch": 1, "val_loss": 1.0},
        "manifest_path": "manifest.csv",
        "device": "cpu",
        "beam_width": 50,
        "slot_eval_sections": None,
        "slot_eval_skipped_reason": None,
    }

    # Key absent (old report / partial section dict): no line, renders as before.
    md_absent = render_markdown({**report_base, "grammar_sections": [section]})
    assert "incomplete-prefix gate rejections" not in md_absent

    # Key present but margin disabled (None): no line.
    section_off = {
        **section,
        "required_command_margin": None,
        "incomplete_prefix_rejections": {"val": 0, "test": 0},
    }
    md_off = render_markdown({**report_base, "grammar_sections": [section_off]})
    assert "incomplete-prefix gate rejections" not in md_off

    # Key present and margin set: one bullet with both split counts.
    section_on = {
        **section,
        "required_command_margin": 0.0,
        "incomplete_prefix_rejections": {"val": 1, "test": 3},
    }
    md_on = render_markdown({**report_base, "grammar_sections": [section_on]})
    assert "- incomplete-prefix gate rejections: val=1, test=3" in md_on


def test_same_margin_across_evaluate_cli_streaming_config_and_decode(
    tmp_path, monkeypatch, vcm_fake_manifest_factory
):
    """Spec matrix row 'Configuration' (audit E5): one explicit margin value
    must reach `decode()` through every surface -- the evaluate CLI flag,
    the streaming JSON config, the streaming runner, and the evaluate
    decode path (recorder per ticket 03 P6)."""
    import me2_voicegen.vcm.dataset as dataset_mod
    import me2_voicegen.vcm.evaluate as evaluate_mod

    from me2_voicegen.vcm.decoder import DecodeResult
    from me2_voicegen.vcm.evaluate import build_arg_parser
    from me2_voicegen.vcm.streaming.config import StreamingConfig
    from me2_voicegen.vcm.streaming.policy import ThresholdPolicy
    from me2_voicegen.vcm.streaming.runner import StreamingRunner

    MARGIN = 0.5

    # (1) evaluate CLI flag
    args = build_arg_parser().parse_args(["--required-command-margin", str(MARGIN)])
    assert args.required_command_margin == MARGIN

    # (2) streaming JSON config
    cfg_path = tmp_path / "streaming.json"
    cfg_path.write_text(json.dumps({"required_command_margin": MARGIN}))
    cfg = StreamingConfig.from_json(cfg_path)
    assert cfg.required_command_margin == MARGIN

    # (3) streaming runner surface: the config value reaches the stored
    # runner field (ticket 03 P4 proves the stored field reaches `decode`
    # on every window). Construction only -- source/backend are never
    # touched by `__init__`.
    runner = StreamingRunner(
        source=object(),
        backend=object(),
        grammar=TOY_GRAMMAR,
        policy=ThresholdPolicy(threshold=-0.1),
        window_s=1.0,
        stride_s=0.25,
        refractory_s=0.0,
        beam_width=25,
        required_command_margin=cfg.required_command_margin,
    )
    assert runner.required_command_margin == MARGIN

    # (4) evaluate surface: decode_split records the CLI-parsed value on
    # every row it decodes.
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    val_dataset = dataset_mod.VCMDataset(manifest_path, split="val", augmenter=None)
    seen: list = []

    def _recorder(
        model,
        feature_extractor,
        waveform,
        grammar,
        threshold,
        beam_width=50,
        device="cpu",
        required_command_margin=None,
    ):
        seen.append(required_command_margin)
        return DecodeResult(
            None, {}, "", float("-inf"), True, float("inf")
        )

    monkeypatch.setattr(evaluate_mod, "infer_waveform", _recorder)
    evaluate_mod.decode_split(
        None,
        None,
        val_dataset,
        TOY_GRAMMAR,
        beam_width=50,
        device="cpu",
        required_command_margin=args.required_command_margin,
    )
    assert seen and all(m == MARGIN for m in seen)
