"""Fast, checkpoint-free tests for `vcm.incomplete_calibration` (ticket 06,
docs/INCOMPLETE-GRAMMAR-REJECTION.md Step 5). Every metric function is
exercised purely over in-memory `RowResult`s and probe dicts; the only real
computation is `decoder.decode_utterance` over synthetic posteriors
(duplicated from `tests/test_vcm_decoder_incomplete_prefix.py`'s idiom,
since `tests/` is not a package) for the arithmetic-vs-live parity check and
the micro-benchmark. No checkpoint, GPU, or audio file anywhere here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from me2_voicegen.vcm import alphabet as vcm_alphabet
from me2_voicegen.vcm import decoder as dec
from me2_voicegen.common.features import HOP_LENGTH, LogMelFeatureExtractor
from me2_voicegen.vcm.evaluate import RowResult, TARGET_BUCKET
from me2_voicegen.vcm.optiona.grammar import TOY_GRAMMAR
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb import incomplete_probes as ip
from me2_voicegen.vcm.optionb.text import normalize_text as optionb_normalize_text
from me2_voicegen.vcm.optionb.transcript import prepare_ctc_transcript

import me2_voicegen.vcm.incomplete_calibration as ic

NEGINF = float("-inf")


class _TrailingSilenceStub(torch.nn.Module):
    """Scripted near-one-hot model for `decode_dataset_rows` (quartznet-
    promotion ticket 01): emits the CTC token path of `text` on frames
    `[0, len(path))` and BLANK peaks on every later frame, so `force_align`
    anchors the characters at frames `0..len(path)-1` and every later frame
    is natural trailing silence. It *advertises* `total_stride` (like
    `QuartzNetCTC`) but never subsamples time -- this is a conversion probe,
    not a strided network. `total_stride=None` leaves the attribute unset,
    mirroring `MatchboxNetCTC`."""

    def __init__(self, text: str, total_stride: int | None = None, peak: float = 20.0) -> None:
        super().__init__()
        ids: list[int] = []
        prev: int | None = None
        for ch in text:
            cid = vcm_alphabet.CHAR_TO_ID[ch]
            if cid == prev:
                ids.append(vcm_alphabet.BLANK_ID)
                prev = None
            ids.append(cid)
            prev = cid
        self.forced_ids = ids
        self.peak = peak
        if total_stride is not None:
            self.total_stride = total_stride

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, frames = features.shape
        logits = torch.full((batch, frames, vcm_alphabet.ALPHABET_SIZE), -self.peak)
        for t, cid in enumerate(self.forced_ids):
            if t >= frames:
                break
            logits[:, t, cid] = self.peak
        for t in range(len(self.forced_ids), frames):
            logits[:, t, vcm_alphabet.BLANK_ID] = self.peak
        return logits


class _MiniDataset:
    """Minimal stand-in for `VCMDataset` exposing exactly what
    `decode_dataset_rows` touches: `.rows` dicts and `__getitem__`
    returning an example with `.waveform`."""

    def __init__(self, rows: list[dict], waveforms: list[torch.Tensor]) -> None:
        self.rows = rows
        self._waveforms = waveforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        return _MiniExample(self._waveforms[i])


class _MiniExample:
    def __init__(self, waveform: torch.Tensor) -> None:
        self.waveform = waveform


# ---------------------------------------------------------------------------
# Grammar-derived command sets.
# ---------------------------------------------------------------------------


def test_single_word_commands_derived_from_grammar():
    # Live-upstream refresh: STOP's v1 is now the bare word "stop" (was
    # "stop song"), so it joins the single-word set.
    words = ic.single_word_commands(OPTIONB_GRAMMAR)
    assert words == frozenset(
        {"call", "message", "pause", "reminders", "stop", "time", "weather"}
    )


def test_strict_prefix_commands_derived_from_grammar():
    # Live-upstream refresh: "stop" is now also a strict-prefix command
    # ("stop" is itself accepted and a whole-word prefix of "stop playing").
    assert ic.strict_prefix_commands(OPTIONB_GRAMMAR) == frozenset({"pause", "stop"})


def test_strict_prefix_commands_excludes_phrases_with_no_longer_sibling():
    # "time" is a single-word command but not a strict prefix of any other
    # accepted phrase (no accepted phrase starts with the word "time").
    assert "time" not in ic.strict_prefix_commands(OPTIONB_GRAMMAR)


def test_character_overlap_prefixes_include_documented_examples():
    overlap = ic.character_overlap_prefixes(OPTIONB_GRAMMAR)
    assert {"remind", "reminder", "what"} <= overlap
    # every flagged prefix must actually be a designated incomplete prefix.
    assert overlap <= OPTIONB_GRAMMAR.incomplete_prefixes


def test_character_overlap_prefixes_empty_for_grammar_without_incomplete_prefixes():
    assert ic.character_overlap_prefixes(TOY_GRAMMAR) == frozenset()


# ---------------------------------------------------------------------------
# Trailing-silence bucketing.
# ---------------------------------------------------------------------------


def test_trailing_silence_bucket_label_boundaries():
    edges = (0.2, 0.5, 1.0, 2.0)
    assert ic.trailing_silence_bucket_label(None, edges) == "unknown"
    assert ic.trailing_silence_bucket_label(0.0, edges) == "[0, 0.2)s"
    assert ic.trailing_silence_bucket_label(0.19, edges) == "[0, 0.2)s"
    assert ic.trailing_silence_bucket_label(0.2, edges) == "[0.2, 0.5)s"
    assert ic.trailing_silence_bucket_label(1.999, edges) == "[1, 2)s"
    assert ic.trailing_silence_bucket_label(2.0, edges) == ">=2s"
    assert ic.trailing_silence_bucket_label(50.0, edges) == ">=2s"


# ---------------------------------------------------------------------------
# RowResult builders for pure-metric tests.
# ---------------------------------------------------------------------------


def _row(
    bucket,
    label,
    intent,
    confidence,
    index=0,
    incomplete_prefix=None,
    incomplete_gap=None,
):
    return RowResult(
        index=index,
        bucket=bucket,
        label=label,
        text="",
        intent=intent,
        confidence=confidence,
        incomplete_prefix=incomplete_prefix,
        incomplete_gap=incomplete_gap,
    )


THRESHOLD = -0.1


def test_accept_with_margin_none_is_confidence_only():
    accepted = _row("target_commands", "TIME", "TIME", -0.05)
    assert ic._accept_with_margin(accepted, THRESHOLD, None)
    rejected = _row("target_commands", "TIME", "TIME", -0.5)
    assert not ic._accept_with_margin(rejected, THRESHOLD, None)


def test_accept_with_margin_gates_on_incomplete_gap():
    row = _row("target_commands", "TIME", "TIME", -0.05, incomplete_gap=-1.41)
    assert ic._accept_with_margin(row, THRESHOLD, -5.0)
    assert not ic._accept_with_margin(row, THRESHOLD, 0.0)
    # gap == margin passes (decoder's own gate is a strict `<` reject).
    assert ic._accept_with_margin(row, THRESHOLD, -1.41)


def test_accept_with_margin_no_competitor_always_passes_margin_gate():
    row = _row("target_commands", "TIME", "TIME", -0.05, incomplete_gap=None)
    assert ic._accept_with_margin(row, THRESHOLD, 5.0)
    assert ic._accept_with_margin(row, THRESHOLD, -5.0)


# ---------------------------------------------------------------------------
# incomplete_prefix_far
# ---------------------------------------------------------------------------


def test_incomplete_prefix_far_overall_and_per_prefix():
    probe_results = [
        _row("incomplete_prefix", "", "TIME", -0.05, index=0, incomplete_gap=None),
        _row("incomplete_prefix", "", None, None, index=1, incomplete_gap=None),
        _row("incomplete_prefix", "", "WEATHER", -0.02, index=2, incomplete_gap=None),
    ]
    probe_rows = [
        {"prefix": "what"},
        {"prefix": "color"},
        {"prefix": "what"},
    ]
    result = ic.incomplete_prefix_far(probe_results, probe_rows, THRESHOLD, None)
    assert result["n"] == 3
    assert result["n_false_accept"] == 2
    assert result["far"] == pytest.approx(2 / 3)

    what_stats = result["per_prefix"]["what"]
    assert what_stats["n"] == 2
    assert what_stats["n_false_accept"] == 2
    assert what_stats["false_accept_intent_distribution"] == {"TIME": 1, "WEATHER": 1}

    color_stats = result["per_prefix"]["color"]
    assert color_stats["n"] == 1
    assert color_stats["n_false_accept"] == 0
    assert color_stats["false_accept_intent_distribution"] == {}


def test_incomplete_prefix_far_never_folds_prefixes_into_aggregate():
    probe_results = [_row("incomplete_prefix", "", "TIME", -0.05, index=i) for i in range(5)]
    probe_rows = [{"prefix": p} for p in ["a", "b", "c", "d", "e"]]
    result = ic.incomplete_prefix_far(probe_results, probe_rows, THRESHOLD, None)
    assert set(result["per_prefix"]) == {"a", "b", "c", "d", "e"}


def test_incomplete_prefix_far_respects_margin_gate():
    probe_results = [
        _row("incomplete_prefix", "", "TIME", -0.05, index=0, incomplete_gap=-1.41),
    ]
    probe_rows = [{"prefix": "color"}]
    at_zero_margin = ic.incomplete_prefix_far(probe_results, probe_rows, THRESHOLD, 0.0)
    assert at_zero_margin["n_false_accept"] == 0
    at_permissive_margin = ic.incomplete_prefix_far(probe_results, probe_rows, THRESHOLD, -5.0)
    assert at_permissive_margin["n_false_accept"] == 1


# ---------------------------------------------------------------------------
# frr_for_phrase_set
# ---------------------------------------------------------------------------


def test_frr_for_phrase_set_counts_only_matching_phrases():
    results = [
        _row("target_commands", "PAUSE", "PAUSE", -0.05, index=0),
        _row("target_commands", "PAUSE", None, None, index=1),  # rejected
        _row("target_commands", "TIME", "TIME", -0.05, index=2),  # not in phrase set
    ]
    raw_rows = [
        {"transcript": "pause"},
        {"transcript": "pause"},
        {"transcript": "time"},
    ]
    result = ic.frr_for_phrase_set(results, raw_rows, frozenset({"pause"}), THRESHOLD, None)
    assert result["n"] == 2
    assert result["n_rejected"] == 1
    assert result["frr"] == pytest.approx(0.5)


def test_frr_for_phrase_set_empty_set_yields_none_not_zerodiv():
    result = ic.frr_for_phrase_set([], [], frozenset({"pause"}), THRESHOLD, None)
    assert result["n"] == 0
    assert result["frr"] is None


# ---------------------------------------------------------------------------
# gate_caused_frr_by_competitor
# ---------------------------------------------------------------------------


def test_gate_caused_frr_by_competitor_groups_rejections():
    results = [
        _row("target_commands", "COLOR", "COLOR", -0.02, index=0, incomplete_prefix="color", incomplete_gap=-2.0),
        _row("target_commands", "TIME", "TIME", -0.02, index=1, incomplete_prefix="what", incomplete_gap=-1.0),
        _row("target_commands", "PAUSE", "PAUSE", -0.02, index=2, incomplete_gap=5.0),  # survives the gate
        _row("target_commands", "STOP", None, None, index=3),  # never confidence-accepted at all
    ]
    result = ic.gate_caused_frr_by_competitor(results, THRESHOLD, margin=0.0)
    assert result["n_confidence_accepted"] == 3
    assert result["n_gate_rejected"] == 2
    assert result["gate_caused_frr"] == pytest.approx(2 / 3)
    assert result["by_competitor"] == {"color": 1, "what": 1}


def test_gate_caused_frr_by_competitor_margin_none_is_always_zero():
    """R1-2/R2-2 fix: the gate-disabled baseline must not error or report
    spurious gate rejections just because `incomplete_gap` happens to be
    very negative -- there is no gate at margin=None."""
    results = [
        _row("target_commands", "COLOR", "COLOR", -0.02, index=0, incomplete_prefix="color", incomplete_gap=-99.0),
    ]
    result = ic.gate_caused_frr_by_competitor(results, THRESHOLD, margin=None)
    assert result["n_confidence_accepted"] == 1
    assert result["n_gate_rejected"] == 0
    assert result["gate_caused_frr"] == 0.0
    assert result["by_competitor"] == {}


def test_gate_caused_frr_by_competitor_detailed_adds_overlap_flag_and_label_distribution():
    results = [
        _row("target_commands", "LIST_REMINDERS", "LIST_REMINDERS", -0.02, index=0, incomplete_prefix="reminder", incomplete_gap=-2.0),
        _row("target_commands", "CREATE_REMINDER", "CREATE_REMINDER", -0.02, index=1, incomplete_prefix="reminder", incomplete_gap=-1.5),
        _row("target_commands", "TIME", "TIME", -0.02, index=2, incomplete_prefix="color", incomplete_gap=-3.0),
    ]
    overlap_prefixes = frozenset({"reminder"})
    result = ic.gate_caused_frr_by_competitor_detailed(results, THRESHOLD, margin=0.0, overlap_prefixes=overlap_prefixes)
    reminder_entry = result["by_competitor"]["reminder"]
    assert reminder_entry["n"] == 2
    assert reminder_entry["char_overlap"] is True
    assert reminder_entry["rejected_true_label_distribution"] == {"CREATE_REMINDER": 1, "LIST_REMINDERS": 1}

    color_entry = result["by_competitor"]["color"]
    assert color_entry["char_overlap"] is False
    assert color_entry["n"] == 1


def test_gate_caused_frr_by_competitor_detailed_margin_none_empty_by_competitor():
    results = [_row("target_commands", "TIME", "TIME", -0.02, index=0, incomplete_prefix="color", incomplete_gap=-99.0)]
    result = ic.gate_caused_frr_by_competitor_detailed(results, THRESHOLD, margin=None, overlap_prefixes=frozenset())
    assert result["by_competitor"] == {}


# ---------------------------------------------------------------------------
# false_accept_stats_with_margin
# ---------------------------------------------------------------------------


def test_false_accept_stats_with_margin_matches_confidence_only_when_margin_none():
    results = [
        _row("babble", "n/a", "CALL", -0.05, index=0),
        _row("babble", "n/a", None, None, index=1),
    ]
    stats = ic.false_accept_stats_with_margin(results, THRESHOLD, None, "babble")
    assert stats == {"n": 2, "false_accepts": 1, "rate": 0.5}


def test_false_accept_stats_with_margin_gate_reduces_false_accepts():
    results = [_row("babble", "n/a", "CALL", -0.05, index=0, incomplete_gap=-2.0)]
    stats = ic.false_accept_stats_with_margin(results, THRESHOLD, 0.0, "babble")
    assert stats["false_accepts"] == 0


def test_false_accept_stats_with_margin_empty_bucket_no_zerodiv():
    stats = ic.false_accept_stats_with_margin([], THRESHOLD, None, "silence")
    assert stats == {"n": 0, "false_accepts": 0, "rate": None}


# ---------------------------------------------------------------------------
# results_by_trailing_silence_bucket / probe_silences_by_index
# ---------------------------------------------------------------------------


def test_results_by_trailing_silence_bucket():
    results = [
        _row("target_commands", "TIME", "TIME", -0.05, index=0),
        _row("target_commands", "TIME", None, None, index=1),
        _row("target_commands", "TIME", "TIME", -0.05, index=2),
    ]
    silences = {0: 0.1, 1: 0.1, 2: 3.0}
    buckets = ic.results_by_trailing_silence_bucket(results, silences, THRESHOLD, None)
    assert buckets["[0, 0.2)s"]["n"] == 2
    assert buckets["[0, 0.2)s"]["n_accepted"] == 1
    assert buckets[">=2s"]["n"] == 1
    assert buckets[">=2s"]["n_accepted"] == 1


def test_probe_silences_by_index_handles_missing_and_blank():
    rows = [{"trailing_silence_s": "0.5"}, {"trailing_silence_s": ""}, {}]
    silences = ic.probe_silences_by_index(rows)
    assert silences[0] == pytest.approx(0.5)
    assert silences[1] is None
    assert silences[2] is None


# ---------------------------------------------------------------------------
# count_alignment_failures / load_probe_generation_failures
# ---------------------------------------------------------------------------


def test_count_alignment_failures_only_counts_optionb_target_rows():
    rows = [
        {"bucket": "target_commands", "source_dataset": "optionb"},
        {"bucket": "target_commands", "source_dataset": "optionb"},
        {"bucket": "babble", "source_dataset": "optionb"},
        {"bucket": "target_commands", "source_dataset": "filipino_speech_corpus"},
    ]
    silences = [0.1, None, 0.2, None]
    result = ic.count_alignment_failures(rows, silences)
    assert result == {"n_attempted": 2, "n_failed": 1}


def test_load_probe_generation_failures_missing_file(tmp_path):
    probe_manifest = tmp_path / "manifest.csv"
    probe_manifest.write_text("filename,path\n")
    result = ic.load_probe_generation_failures(probe_manifest)
    assert result["available"] is False
    assert result["n_failures"] == 0


def test_load_probe_generation_failures_present_file(tmp_path):
    """Matches ticket 05's actual `incomplete_probes.build_generation_report`
    output shape (R1-1): `generation_report.json` with a `failure_counts`
    dict, not a `probe_generation_failures.csv`."""
    probe_manifest = tmp_path / "manifest.csv"
    probe_manifest.write_text("filename,path\n")
    (tmp_path / "generation_report.json").write_text(
        json.dumps(
            {
                "split": "val",
                "source_rows_selected": 10,
                "probe_rows_written": 40,
                "failure_counts": {
                    "force_align_error": 2,
                    "no_gap": 1,
                    "crop_too_short": 0,
                    "missing_audio": 0,
                },
                "failures": [],
            }
        )
    )
    result = ic.load_probe_generation_failures(probe_manifest)
    assert result["available"] is True
    assert result["n_failures"] == 3
    assert result["by_reason"] == {
        "crop_too_short": 0,
        "force_align_error": 2,
        "missing_audio": 0,
        "no_gap": 1,
    }
    assert result["source_rows_selected"] == 10
    assert result["probe_rows_written"] == 40


# ---------------------------------------------------------------------------
# evaluate_margin_grid / select_margin
# ---------------------------------------------------------------------------


def _target_row(index, intent, label, confidence, incomplete_gap=None):
    return RowResult(
        index=index, bucket="target_commands", label=label, text="", intent=intent,
        confidence=confidence, incomplete_gap=incomplete_gap,
    )


def test_evaluate_margin_grid_includes_baseline_row_first():
    targets = [_target_row(0, "TIME", "TIME", -0.05, incomplete_gap=-1.41)]
    probes = [_row("incomplete_prefix", "", "TIME", -0.05, index=0, incomplete_gap=None)]
    probe_rows = [{"prefix": "time"}]
    grid = ic.evaluate_margin_grid(targets, probes, probe_rows, THRESHOLD, margins=(0.0, -5.0))
    assert grid[0]["is_baseline"] and grid[0]["margin"] is None
    assert [row["margin"] for row in grid[1:]] == [0.0, -5.0]
    # R2-2(a): every grid row carries per-prefix FAR + gate-caused FRR, not
    # just the aggregate scalar `select_margin` consumes.
    for row in grid:
        assert "per_prefix_far" in row
        assert "gate_caused_frr" in row


def test_evaluate_margin_grid_splits_far_by_digital_zero_silence_source():
    """R2-6: a grid row's `incomplete_prefix_far` (all probes) can differ from
    `incomplete_prefix_far_excl_digital_zero` (room-tone/none only) when a
    margin's accept/reject decision differs between a digital-zero-padded
    probe and a room-tone-padded one -- constructed here so the two rates are
    provably different, not just present."""
    # Row 0: digital-zero padding, gap sits just above margin -5.0 -> accepted
    #        (false accept) at margin=-5.0 but rejected at margin=0.0.
    probes = [
        _row("incomplete_prefix", "", "REMIND", -0.05, index=0, incomplete_gap=-3.0),
        # Row 1: room-tone padding, gap always below -5.0 -> never accepted.
        _row("incomplete_prefix", "", "REMIND", -0.05, index=1, incomplete_gap=-9.0),
    ]
    probe_rows = [
        {"prefix": "remind", "silence_source": ic.SILENCE_SOURCE_DIGITAL_ZERO},
        {"prefix": "remind", "silence_source": ic.SILENCE_SOURCE_TAIL},
    ]
    grid = ic.evaluate_margin_grid([], probes, probe_rows, THRESHOLD, margins=(-5.0,))
    row = next(r for r in grid if r["margin"] == -5.0)
    assert row["n_probe"] == 2
    assert row["n_probe_excl_digital_zero"] == 1
    assert row["n_probe_digital_zero_only"] == 1
    # All-probe FAR: 1/2 false-accepted (the digital-zero one). Excl-digital-zero
    # FAR: 0/1 (the room-tone probe alone). Digital-zero-only FAR: 1/1.
    assert row["incomplete_prefix_far"] == pytest.approx(0.5)
    assert row["incomplete_prefix_far_excl_digital_zero"] == pytest.approx(0.0)
    assert row["incomplete_prefix_far_digital_zero_only"] == pytest.approx(1.0)


def test_select_margin_picks_lowest_far_excl_digital_zero_within_budget():
    grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 0.90,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 0.895,
            "incomplete_prefix_far": 0.10, "incomplete_prefix_far_excl_digital_zero": 0.10,
        },
        {
            "margin": -1.0, "is_baseline": False, "target_exact_accuracy": 0.85,
            "incomplete_prefix_far": 0.02, "incomplete_prefix_far_excl_digital_zero": 0.02,
        },
        {
            "margin": -5.0, "is_baseline": False, "target_exact_accuracy": 0.899,
            "incomplete_prefix_far": 0.30, "incomplete_prefix_far_excl_digital_zero": 0.30,
        },
    ]
    # -1.0's accuracy drop is 5pp > 1.0pp budget -> ineligible despite best FAR.
    # Among the two eligible margins (0.0, -5.0), 0.0 has the lower FAR.
    chosen = ic.select_margin(grid, regression_budget_pp=1.0)
    assert chosen["margin"] == 0.0


def test_select_margin_uses_excl_digital_zero_not_all_probe_far():
    """R2-6's actual bug: selection must ignore `incomplete_prefix_far` (all
    probes, digital-zero-diluted) and use `incomplete_prefix_far_excl_digital_zero`
    instead. Constructed so the two fields disagree on which margin is best --
    a pre-fix implementation reading the wrong field would pick -5.0 here."""
    grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 0.90,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 0.895,
            "incomplete_prefix_far": 0.20, "incomplete_prefix_far_excl_digital_zero": 0.05,
        },
        {
            "margin": -5.0, "is_baseline": False, "target_exact_accuracy": 0.899,
            "incomplete_prefix_far": 0.10, "incomplete_prefix_far_excl_digital_zero": 0.30,
        },
    ]
    chosen = ic.select_margin(grid, regression_budget_pp=1.0)
    assert chosen["margin"] == 0.0


def test_select_margin_ties_break_toward_smallest_margin():
    grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 0.90,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
        },
        {
            "margin": 1.0, "is_baseline": False, "target_exact_accuracy": 0.895,
            "incomplete_prefix_far": 0.05, "incomplete_prefix_far_excl_digital_zero": 0.05,
        },
        {
            "margin": -2.0, "is_baseline": False, "target_exact_accuracy": 0.895,
            "incomplete_prefix_far": 0.05, "incomplete_prefix_far_excl_digital_zero": 0.05,
        },
    ]
    chosen = ic.select_margin(grid, regression_budget_pp=1.0)
    assert chosen["margin"] == -2.0


def test_select_margin_raises_when_no_margin_eligible():
    grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 0.90,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 0.50,
            "incomplete_prefix_far": 0.01, "incomplete_prefix_far_excl_digital_zero": 0.01,
        },
    ]
    with pytest.raises(ic.NoEligibleMarginError):
        ic.select_margin(grid, regression_budget_pp=1.0)


def test_select_margin_raises_when_excl_digital_zero_is_none_for_all_candidates():
    """Edge case: every candidate's non-digital-zero probe set is empty (all
    padding happened to be digital-zero). `incomplete_prefix_far_excl_digital_zero`
    is `None` for each, so none are eligible -- must raise, not crash with
    TypeError from comparing `None` in `min()`."""
    grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 0.90,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 0.895,
            "incomplete_prefix_far": 0.10, "incomplete_prefix_far_excl_digital_zero": None,
        },
    ]
    with pytest.raises(ic.NoEligibleMarginError):
        ic.select_margin(grid, regression_budget_pp=1.0)


# ---------------------------------------------------------------------------
# sha256_of_file / hash_checkpoint_and_onnx / check_holdout_can_run
# ---------------------------------------------------------------------------


def test_sha256_of_file_is_deterministic_and_content_sensitive(tmp_path):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"hello world")
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"hello world")
    f3 = tmp_path / "c.bin"
    f3.write_bytes(b"different")
    assert ic.sha256_of_file(f1) == ic.sha256_of_file(f2)
    assert ic.sha256_of_file(f1) != ic.sha256_of_file(f3)


def test_hash_checkpoint_and_onnx_includes_every_onnx_file(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"ckpt")
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.fp32.onnx").write_bytes(b"fp32")
    (export_dir / "model.int8.onnx").write_bytes(b"int8")
    (export_dir / "notes.txt").write_bytes(b"ignored")

    hashes = ic.hash_checkpoint_and_onnx(checkpoint, export_dir)
    assert set(hashes) == {"checkpoint", "model.fp32.onnx", "model.int8.onnx"}
    assert hashes["checkpoint"] == ic.sha256_of_file(checkpoint)


def test_hash_checkpoint_and_onnx_missing_onnx_dir_still_hashes_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"ckpt")
    hashes = ic.hash_checkpoint_and_onnx(checkpoint, tmp_path / "does_not_exist")
    assert set(hashes) == {"checkpoint"}


def test_check_holdout_can_run_refuses_when_report_already_exists(tmp_path):
    report_path = tmp_path / "test_report.json"
    report_path.write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        ic.check_holdout_can_run(report_path, {"checkpoint_sha256": "abc"}, {"checkpoint": "abc"})


def test_check_holdout_can_run_refuses_on_hash_mismatch(tmp_path):
    report_path = tmp_path / "test_report.json"
    with pytest.raises(SystemExit, match="mismatch"):
        ic.check_holdout_can_run(report_path, {"checkpoint_sha256": "abc"}, {"checkpoint": "different"})


def test_check_holdout_can_run_passes_when_no_report_and_hash_matches(tmp_path):
    report_path = tmp_path / "test_report.json"
    ic.check_holdout_can_run(report_path, {"checkpoint_sha256": "abc"}, {"checkpoint": "abc"})  # no raise


# ---------------------------------------------------------------------------
# R2-3 guards: split validation, aligner-checkpoint validation, select-vs-
# holdout ordering, holdout beam-width reconciliation.
# ---------------------------------------------------------------------------


def test_validate_probe_split_passes_when_all_rows_match():
    rows = [{"split": "val"}, {"split": "val"}]
    ic.validate_probe_split(rows, "val")  # no raise


def test_validate_probe_split_refuses_on_wrong_split():
    rows = [{"split": "val"}, {"split": "test"}]
    with pytest.raises(SystemExit, match="split"):
        ic.validate_probe_split(rows, "val")


def test_validate_probe_split_refuses_holdout_on_val_probes():
    rows = [{"split": "val"}]
    with pytest.raises(SystemExit):
        ic.validate_probe_split(rows, "test")


def test_validate_aligner_checkpoint_passes_when_matching():
    rows = [{"aligner_checkpoint_sha256": "abc"}, {"aligner_checkpoint_sha256": "abc"}]
    ic.validate_aligner_checkpoint(rows, "abc")  # no raise


def test_validate_aligner_checkpoint_refuses_on_mismatch():
    rows = [{"aligner_checkpoint_sha256": "different"}]
    with pytest.raises(SystemExit, match="aligner_checkpoint_sha256"):
        ic.validate_aligner_checkpoint(rows, "abc")


def test_check_select_can_run_passes_when_no_holdout_report(tmp_path):
    ic.check_select_can_run(tmp_path)  # no raise


def test_check_select_can_run_refuses_when_holdout_report_exists(tmp_path):
    (tmp_path / "test_report.json").write_text("{}")
    with pytest.raises(SystemExit, match="holdout report already exists"):
        ic.check_select_can_run(tmp_path)


def test_resolve_holdout_beam_width_inherits_from_selection_when_not_passed():
    assert ic.resolve_holdout_beam_width(None, 25) == 25


def test_resolve_holdout_beam_width_accepts_matching_explicit_value():
    assert ic.resolve_holdout_beam_width(25, 25) == 25


def test_resolve_holdout_beam_width_refuses_mismatched_explicit_value():
    with pytest.raises(SystemExit, match="beam"):
        ic.resolve_holdout_beam_width(50, 25)


# ---------------------------------------------------------------------------
# build_metrics: single entry point bundling every margin-dependent metric
# (R1-2/R2-2), called identically at baseline and at a candidate margin.
# ---------------------------------------------------------------------------


def _build_metrics_kwargs(margin: float | None) -> dict:
    target_results = [
        _target_row(0, "PAUSE", "PAUSE", -0.05, incomplete_gap=None),
        _target_row(1, "TIME", "TIME", -0.05, incomplete_gap=-1.41),
    ]
    probe_results = [
        _row("incomplete_prefix", "", "TIME", -0.05, index=0, incomplete_gap=None),
    ]
    probe_rows = [{"prefix": "what", "trailing_silence_s": "0.5"}]
    raw_target_rows = [{"transcript": "pause"}, {"transcript": "time"}]
    reject_results_all = target_results + [_row("babble", "n/a", None, None, index=2)]
    return dict(
        target_results=target_results,
        probe_results=probe_results,
        probe_rows=probe_rows,
        raw_target_rows=raw_target_rows,
        reject_results_all=reject_results_all,
        target_silences_by_index={0: 0.1, 1: 3.0},
        probe_silences={0: 0.5},
        threshold=THRESHOLD,
        margin=margin,
        grammar=OPTIONB_GRAMMAR,
    )


def test_build_metrics_baseline_and_gated_have_identical_shape():
    baseline = ic.build_metrics(**_build_metrics_kwargs(None))
    gated = ic.build_metrics(**_build_metrics_kwargs(0.0))
    assert set(baseline) == set(gated)
    assert baseline["is_baseline"] is True
    assert gated["is_baseline"] is False
    # R1-2: slot exact match / confusion counts are present at both configs.
    assert "slot_accuracy" in baseline and "slot_accuracy" in gated
    assert "confusion_counts" in baseline and "confusion_counts" in gated


def test_build_metrics_gated_reflects_margin_gate_in_confusion_counts():
    gated = ic.build_metrics(**_build_metrics_kwargs(0.0))
    # TIME's row has incomplete_gap=-1.41 < margin 0.0 -> margin-gate rejects
    # it, so confusion_counts (fed the margin-gated RowResults) must show
    # REJECTED for TIME, not TIME.
    assert gated["confusion_counts"]["TIME"] == {"REJECTED": 1}
    baseline = ic.build_metrics(**_build_metrics_kwargs(None))
    assert baseline["confusion_counts"]["TIME"] == {"TIME": 1}


# ---------------------------------------------------------------------------
# benchmark_decode
# ---------------------------------------------------------------------------


def _make_posterior(text: str, peak: float = 20.0) -> np.ndarray:
    frame_ids: list[int] = []
    prev = None
    for ch in text:
        cid = vcm_alphabet.CHAR_TO_ID[ch]
        if cid == prev:
            frame_ids.append(vcm_alphabet.BLANK_ID)
            prev = None
        frame_ids.append(cid)
        prev = cid
    T = len(frame_ids)
    logits = np.full((T, vcm_alphabet.ALPHABET_SIZE), -peak, dtype=np.float64)
    for t, cid in enumerate(frame_ids):
        logits[t, cid] = peak
    m = logits.max(axis=-1, keepdims=True)
    return logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))


def test_benchmark_decode_reports_median_p95_and_peak_memory():
    logp_arrays = [_make_posterior("time") for _ in range(5)]
    result = ic.benchmark_decode(logp_arrays, OPTIONB_GRAMMAR, threshold=-1.0, beam_width=25, required_command_margin=None)
    assert result["n"] == 5
    assert result["median_latency_ms"] >= 0.0
    assert result["p95_latency_ms"] >= result["median_latency_ms"]
    assert result["peak_memory_bytes"] > 0


def test_benchmark_decode_empty_input_no_crash():
    result = ic.benchmark_decode([], OPTIONB_GRAMMAR, threshold=-1.0, beam_width=25, required_command_margin=None)
    assert result["n"] == 0
    assert result["median_latency_ms"] is None


# ---------------------------------------------------------------------------
# Must Verify (a): arithmetic margin decision matches a live
# decode_utterance(required_command_margin=m) on synthetic posteriors,
# including the no-terminal case.
# ---------------------------------------------------------------------------


Q_COLOR_CHAR = 0.56
Q_COLOR_BLANK = 0.44
TIME_RAW_TARGET = -6.98
Q_TIME_CHAR = math.exp((TIME_RAW_TARGET - 5.0 * math.log(Q_COLOR_BLANK)) / 4.0)
Q_TIME_BLANK = 1.0 - Q_TIME_CHAR


def make_color_broken_posterior(total_frames: int) -> np.ndarray:
    size = vcm_alphabet.ALPHABET_SIZE
    logp = np.full((total_frames, size), NEGINF, dtype=np.float64)
    for t, ch in enumerate("color"):
        logp[t, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_COLOR_CHAR)
        logp[t, vcm_alphabet.BLANK_ID] = math.log(Q_COLOR_BLANK)
    for i, ch in enumerate("time"):
        logp[5 + i, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_TIME_CHAR)
        logp[5 + i, vcm_alphabet.BLANK_ID] = math.log(Q_TIME_BLANK)
    logp[9:, vcm_alphabet.BLANK_ID] = 0.0
    return logp


def make_color_no_completion_posterior(total_frames: int) -> np.ndarray:
    size = vcm_alphabet.ALPHABET_SIZE
    logp = np.full((total_frames, size), NEGINF, dtype=np.float64)
    for t, ch in enumerate("color"):
        logp[t, vcm_alphabet.CHAR_TO_ID[ch]] = math.log(Q_COLOR_CHAR)
        logp[t, vcm_alphabet.BLANK_ID] = math.log(Q_COLOR_BLANK)
    logp[5:, vcm_alphabet.BLANK_ID] = 0.0
    return logp


def _baseline_row(logp, threshold, beam_width=50):
    decoded = dec.decode_utterance(logp, OPTIONB_GRAMMAR, threshold=threshold, beam_width=beam_width, required_command_margin=None)
    return ic._row_result_from_decode(0, "target_commands", "TIME", None, "optionb", decoded), decoded


@pytest.mark.parametrize("margin", [0.0, -1.41, -10.0, 5.0])
def test_arithmetic_margin_decision_matches_live_decode_completion_case(margin):
    threshold = -0.1
    logp = make_color_broken_posterior(151)
    baseline_row, baseline_decoded = _baseline_row(logp, threshold)
    predicted_accept = ic._accept_with_margin(baseline_row, threshold, margin)

    live = dec.decode_utterance(logp, OPTIONB_GRAMMAR, threshold=threshold, beam_width=50, required_command_margin=margin)
    assert (not live.no_match) == predicted_accept


@pytest.mark.parametrize("margin", [0.0, -10.0, 5.0])
def test_arithmetic_margin_decision_matches_live_decode_no_terminal_case(margin):
    threshold = -1.0
    logp = make_color_no_completion_posterior(70)
    baseline_row, baseline_decoded = _baseline_row(logp, threshold)
    assert baseline_decoded.intent is None  # no completed terminal exists at all

    predicted_accept = ic._accept_with_margin(baseline_row, threshold, margin)
    assert predicted_accept is False  # no intent -> never accepted regardless of margin

    live = dec.decode_utterance(logp, OPTIONB_GRAMMAR, threshold=threshold, beam_width=50, required_command_margin=margin)
    assert live.no_match is True
    assert (not live.no_match) == predicted_accept


# ---------------------------------------------------------------------------
# build_val_report / render_val_markdown (ticket 09 gap audit: neither was
# exercised by ticket 06's own suite, despite being the report-assembly
# glue every metric function above ultimately feeds).
# ---------------------------------------------------------------------------


def _empty_benchmark() -> dict:
    return ic.benchmark_decode([], OPTIONB_GRAMMAR, threshold=-1.0, beam_width=25, required_command_margin=None)


def _build_val_report_kwargs(margin_grid: list[dict]) -> dict:
    target_results = [
        _target_row(0, "PAUSE", "PAUSE", -0.05, incomplete_gap=None),
        _target_row(1, "TIME", "TIME", -0.05, incomplete_gap=None),
    ]
    probe_results = [
        _row("incomplete_prefix", "", "TIME", -0.05, index=0, incomplete_gap=None),
        _row("incomplete_prefix", "", None, None, index=1, incomplete_gap=None),
    ]
    probe_rows = [{"prefix": "what", "trailing_silence_s": "0.5"}, {"prefix": "color", "trailing_silence_s": ""}]
    raw_target_rows = [{"transcript": "pause"}, {"transcript": "time"}]
    reject_results_all = target_results + [
        _row("babble", "n/a", None, None, index=2),
        _row("silence", "n/a", None, None, index=3),
    ]
    return dict(
        checkpoint_path=Path("ckpt.pt"),
        beam_width=25,
        threshold=THRESHOLD,
        margin_grid=margin_grid,
        grammar=OPTIONB_GRAMMAR,
        target_results=target_results,
        probe_results=probe_results,
        probe_rows=probe_rows,
        probe_manifest_path=Path("probes/val/manifest.csv"),
        raw_target_rows=raw_target_rows,
        target_silences=[0.1, 3.0],
        reject_results_all=reject_results_all,
        benchmark_gate_off=_empty_benchmark(),
        benchmark_gate_chosen_margin=_empty_benchmark(),
        alignment_failures={"n_attempted": 2, "n_failed": 0},
        probe_generation_failures={"available": False, "path": "n/a", "n_failures": 0, "by_reason": {}},
        hashes_before={"checkpoint": "abc"},
        hashes_after={"checkpoint": "abc"},
    )


def test_build_val_report_happy_path_shape_and_renders_markdown():
    margin_grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 1.0,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
            "incomplete_prefix_far_digital_zero_only": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 1.0,
            "incomplete_prefix_far": 0.5, "incomplete_prefix_far_excl_digital_zero": 0.5,
            "incomplete_prefix_far_digital_zero_only": None,
        },
    ]
    report = ic.build_val_report(**_build_val_report_kwargs(margin_grid))

    assert "no_eligible_margin" not in report
    assert report["chosen_margin"]["margin"] == 0.0
    assert report["checkpoint_sha256"] == "abc"
    assert report["probe_manifest_path"] == "probes/val/manifest.csv"
    gated = report["gated_metrics"]
    baseline = report["baseline_metrics"]
    assert set(gated["character_overlap_prefixes"]) <= OPTIONB_GRAMMAR.incomplete_prefixes
    assert gated["incomplete_prefix_far"]["n"] == 2
    assert gated["target_by_silence_bucket"]
    assert gated["probe_by_silence_bucket"]
    assert baseline["is_baseline"] is True
    assert gated["is_baseline"] is False
    # R1-2: slot exact match and confusion counts must be present.
    assert "slot_accuracy" in gated
    assert "confusion_counts" in gated

    md = ic.render_val_markdown(report)
    assert "Chosen margin: **0.0**" in md
    assert "## Margin grid" in md
    assert "## Baseline vs. chosen-margin metrics" in md
    assert "### baseline (gate off)" in md
    assert "### margin=0.0" in md
    assert "## Checkpoint/ONNX hashes" in md
    # "what" is a documented character-overlap prefix (see
    # test_character_overlap_prefixes_include_documented_examples); its
    # per-prefix FAR row must carry the YES flag and its false-accept
    # intent ("TIME") must show up in the rendered intent-confusion column,
    # for BOTH the baseline and gated blocks (R2-2).
    what_lines = [line for line in md.splitlines() if line.startswith("| what ")]
    assert len(what_lines) == 2
    for line in what_lines:
        assert "| YES |" in line
        assert "TIME=1" in line
    color_lines = [line for line in md.splitlines() if line.startswith("| color ")]
    for line in color_lines:
        assert "|  |" in line  # no overlap flag for "color"


def test_build_val_report_no_eligible_margin_short_circuits():
    margin_grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 1.0,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
            "incomplete_prefix_far_digital_zero_only": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 0.0,
            "incomplete_prefix_far": 0.5, "incomplete_prefix_far_excl_digital_zero": 0.5,
            "incomplete_prefix_far_digital_zero_only": None,
        },
    ]
    report = ic.build_val_report(**_build_val_report_kwargs(margin_grid))

    assert report["no_eligible_margin"]
    assert "chosen_margin" not in report
    assert "gated_metrics" not in report

    md = ic.render_val_markdown(report)
    assert "No eligible margin" in md
    assert "## Margin grid" in md


def test_build_val_report_regression_budget_pp_defaults_and_is_recorded():
    margin_grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 1.0,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
            "incomplete_prefix_far_digital_zero_only": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 1.0,
            "incomplete_prefix_far": 0.5, "incomplete_prefix_far_excl_digital_zero": 0.5,
            "incomplete_prefix_far_digital_zero_only": None,
        },
    ]
    report = ic.build_val_report(**_build_val_report_kwargs(margin_grid))
    assert report["regression_budget_pp"] == ic.REGRESSION_BUDGET_PP
    md = ic.render_val_markdown(report)
    assert f"regression_budget_pp={ic.REGRESSION_BUDGET_PP}" in md


def test_build_val_report_custom_regression_budget_pp_changes_selection_and_is_recorded():
    """A tighter budget than the default (0.2pp) makes margin=0.0 ineligible
    here (its 0.5pp accuracy drop from baseline exceeds 0.2pp but is within
    the default 1.0pp budget) -- proves the CLI-facing knob actually reaches
    `select_margin`, not just that the field exists."""
    margin_grid = [
        {
            "margin": None, "is_baseline": True, "target_exact_accuracy": 1.0,
            "incomplete_prefix_far": None, "incomplete_prefix_far_excl_digital_zero": None,
            "incomplete_prefix_far_digital_zero_only": None,
        },
        {
            "margin": 0.0, "is_baseline": False, "target_exact_accuracy": 0.995,
            "incomplete_prefix_far": 0.5, "incomplete_prefix_far_excl_digital_zero": 0.5,
            "incomplete_prefix_far_digital_zero_only": None,
        },
    ]
    kwargs = _build_val_report_kwargs(margin_grid)

    default_report = ic.build_val_report(**kwargs)
    assert default_report["regression_budget_pp"] == ic.REGRESSION_BUDGET_PP
    assert default_report["chosen_margin"]["margin"] == 0.0

    strict_report = ic.build_val_report(**kwargs, regression_budget_pp=0.2)
    assert strict_report["regression_budget_pp"] == 0.2
    assert strict_report["no_eligible_margin"]
    assert "chosen_margin" not in strict_report


# ---------------------------------------------------------------------------
# render_holdout_markdown: R2-2(c) parity with the val report -- overlap
# flags, FRR by competitor, single-word/strict-prefix FRR, and silence
# buckets must all still be present, not stripped down to the aggregate FAR.
# ---------------------------------------------------------------------------


def test_render_holdout_markdown_has_full_parity_with_val_metrics_blocks():
    baseline = ic.build_metrics(**_build_metrics_kwargs(None))
    gated = ic.build_metrics(**_build_metrics_kwargs(0.0))
    report = {
        "checkpoint_path": "ckpt.pt",
        "beam_width": 25,
        "margin": 0.0,
        "threshold": THRESHOLD,
        "val_selection_path": "out/val_selection.json",
        "baseline_metrics": baseline,
        "gated_metrics": gated,
        "hashes_before": {"checkpoint": "abc"},
        "hashes_after": {"checkpoint": "abc"},
    }
    md = ic.render_holdout_markdown(report)
    assert "## baseline (gate off)" in md
    assert "## margin=0.0" in md
    # overlap flag, per-competitor FRR, single-word/strict-prefix FRR, and
    # silence buckets all present -- not just the aggregate FAR.
    assert "char_overlap" in md
    assert "FRR (single-word commands)" in md
    assert "FRR (strict-prefix commands)" in md
    assert "trailing-silence bucket" in md
    assert "silence_source" in md  # R2-5: holdout gets the same breakdown as val.
    assert "## Checkpoint/ONNX hashes" in md


# ---------------------------------------------------------------------------
# R2-5 fix: per-silence_source breakdown (ticket 05's incomplete_probes.py
# SILENCE_SOURCE_* constants). Operating Principle 10: the fixture below
# calls 05's real `generate_probes` (with synthetic waveforms/logp -- no
# checkpoint or WAV file, same idiom as tests/test_vcm_incomplete_probes.py)
# rather than hand-fabricating `silence_source` string values, so this
# module is verified against what 05 actually emits, not a description of
# it. All four real `silence_source` values it can produce -- `none`,
# `lead_in_room_tone`, `tail_room_tone`, `digital_zero` -- are exercised.
# ---------------------------------------------------------------------------


def _make_synthetic_posterior(text: str, peak: float = 20.0) -> np.ndarray:
    frame_ids: list[int] = []
    prev = None
    for ch in text:
        cid = vcm_alphabet.CHAR_TO_ID[ch]
        if cid == prev:
            frame_ids.append(vcm_alphabet.BLANK_ID)
            prev = None
        frame_ids.append(cid)
        prev = cid
    T = len(frame_ids)
    logits = np.full((T, vcm_alphabet.ALPHABET_SIZE), -peak, dtype=np.float64)
    for t, cid in enumerate(frame_ids):
        logits[t, cid] = peak
    m = logits.max(axis=-1, keepdims=True)
    return logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))


def _real_probe_manifest_rows_with_all_silence_sources() -> list[dict]:
    """Builds one probe manifest via a real call to
    `incomplete_probes.generate_probes`, engineered (per that module's own
    `find_quiet_window`/`build_trailing_silence` rules, reverse-derived by
    running `force_align` first to get real frame boundaries, not guessed)
    so its three non-`digital_zero` waveforms each land on a different real
    `silence_source`, plus the trivial `silence_buckets=(0.0, 0.3)` `none`
    bucket every spec also produces."""
    text = "set the lights to red"
    prefix = "set the lights to"
    rng = np.random.default_rng(0)

    def row(filename: str) -> dict:
        return {"filename": filename, "transcript": text, "path": f"audio/{filename}", "group_id": "g1"}

    # lead_in_room_tone: prepend confident-blank frames so force_align's
    # start_frame lands far enough in for a full lead-in window; end the
    # waveform right at speech end so no tail region exists at all.
    K = 50
    blank_row = np.full(vcm_alphabet.ALPHABET_SIZE, -1e9)
    blank_row[vcm_alphabet.BLANK_ID] = 0.0
    logp_leadin = np.vstack([np.tile(blank_row, (K, 1)), _make_synthetic_posterior(text)])
    speech_start, speech_end = 8000, 11360  # = start_frame*160, (end_frame+1)*160, verified below
    wav_leadin = np.zeros(speech_end, dtype=np.float32)
    wav_leadin[:4800] = rng.normal(0, 0.001, 4800)
    wav_leadin[speech_start:speech_end] = rng.normal(0, 1.0, speech_end - speech_start)

    # tail_room_tone: no leading blanks (start_frame=0 -> no lead-in region
    # at all); quiet tail window well past the backoff zone.
    logp_tail = _make_synthetic_posterior(text)
    wav_tail = np.zeros(10000, dtype=np.float32)
    wav_tail[0:3360] = rng.normal(0, 1.0, 3360)
    wav_tail[6560:10000] = rng.normal(0, 0.001, 10000 - 6560)

    # digital_zero: an all-silent waveform has speech_rms == 0, which
    # find_quiet_window treats as "cannot validate" -> always digital_zero.
    logp_zero = _make_synthetic_posterior(text)
    wav_zero = np.zeros(3360, dtype=np.float32)

    waveforms = {"leadin.wav": wav_leadin, "tail.wav": wav_tail, "zero.wav": wav_zero}
    logps = {"leadin.wav": logp_leadin, "tail.wav": logp_tail, "zero.wav": logp_zero}

    specs = [ip.ProbeSpec(source_row=row(fn), prefix=prefix) for fn in waveforms]

    def load_waveform(source_row: dict) -> np.ndarray:
        return waveforms[source_row["filename"]]

    def compute_logp(waveform: np.ndarray) -> np.ndarray:
        for filename, candidate in waveforms.items():
            if candidate is waveform:
                return logps[filename]
        raise AssertionError("unmatched waveform in test fixture")

    result = ip.generate_probes(
        specs,
        split="val",
        grace_frames=2,
        silence_buckets=(0.0, 0.3),
        aligner_checkpoint_path="ckpt.pt",
        aligner_checkpoint_sha256="deadbeef",
        load_waveform=load_waveform,
        compute_logp=compute_logp,
        resolve_and_normalize=lambda source_row: source_row["transcript"],
    )
    assert not result.failure_counts, f"fixture construction hit a real failure: {result.failure_counts}"
    return result.manifest_rows


def test_real_generate_probes_fixture_covers_all_four_silence_sources():
    """Sanity-checks the fixture itself actually exercises 05's real
    constants (not a hand-typed guess at what they'd be)."""
    rows = _real_probe_manifest_rows_with_all_silence_sources()
    sources = {row["silence_source"] for row in rows}
    assert sources == {
        ip.SILENCE_SOURCE_NONE,
        ip.SILENCE_SOURCE_LEAD_IN,
        ip.SILENCE_SOURCE_TAIL,
        ip.SILENCE_SOURCE_DIGITAL_ZERO,
    }


def test_probe_far_by_silence_source_groups_by_real_values_present():
    probe_rows = _real_probe_manifest_rows_with_all_silence_sources()
    probe_results = [
        _row("incomplete_prefix", "", "COLOR" if i % 2 == 0 else None, -0.05 if i % 2 == 0 else None, index=i)
        for i in range(len(probe_rows))
    ]
    result = ic.probe_far_by_silence_source(probe_results, probe_rows, THRESHOLD, None)

    present_sources = {row["silence_source"] for row in probe_rows}
    assert set(result) == present_sources
    for key, stats in result.items():
        expected_n = sum(1 for row in probe_rows if row["silence_source"] == key)
        assert stats["n"] == expected_n
        assert stats["far"] == stats["n_false_accept"] / stats["n"]


def test_probe_far_by_silence_source_missing_key_uses_sentinel_not_a_real_constant():
    probe_results = [_row("incomplete_prefix", "", "TIME", -0.05, index=0)]
    probe_rows = [{"prefix": "what"}]  # no silence_source column at all
    result = ic.probe_far_by_silence_source(probe_results, probe_rows, THRESHOLD, None)
    assert set(result) == {ic._MISSING_SILENCE_SOURCE}
    assert ic._MISSING_SILENCE_SOURCE not in {
        ip.SILENCE_SOURCE_NONE, ip.SILENCE_SOURCE_LEAD_IN, ip.SILENCE_SOURCE_TAIL, ip.SILENCE_SOURCE_DIGITAL_ZERO,
    }


def test_ordered_silence_source_keys_uses_display_order_then_leftovers_sorted():
    present = {
        ip.SILENCE_SOURCE_DIGITAL_ZERO: {},
        ip.SILENCE_SOURCE_NONE: {},
        "some_future_source": {},
        ip.SILENCE_SOURCE_LEAD_IN: {},
    }
    ordered = ic._ordered_silence_source_keys(present)
    assert ordered == [
        ip.SILENCE_SOURCE_NONE,
        ip.SILENCE_SOURCE_LEAD_IN,
        ip.SILENCE_SOURCE_DIGITAL_ZERO,
        "some_future_source",
    ]


def test_build_metrics_includes_probe_far_by_silence_source_from_real_probe_rows():
    probe_rows = _real_probe_manifest_rows_with_all_silence_sources()
    probe_results = [_row("incomplete_prefix", "", "COLOR", -0.05, index=i) for i in range(len(probe_rows))]
    kwargs = _build_metrics_kwargs(0.0)
    kwargs["probe_results"] = probe_results
    kwargs["probe_rows"] = probe_rows
    kwargs["probe_silences"] = {}
    metrics = ic.build_metrics(**kwargs)
    assert "probe_far_by_silence_source" in metrics
    assert set(metrics["probe_far_by_silence_source"]) == {row["silence_source"] for row in probe_rows}


def test_render_metrics_markdown_shows_silence_source_breakdown_with_real_values():
    probe_rows = _real_probe_manifest_rows_with_all_silence_sources()
    probe_results = [_row("incomplete_prefix", "", "COLOR", -0.05, index=i) for i in range(len(probe_rows))]
    kwargs = _build_metrics_kwargs(0.0)
    kwargs["probe_results"] = probe_results
    kwargs["probe_rows"] = probe_rows
    kwargs["probe_silences"] = {}
    metrics = ic.build_metrics(**kwargs)
    lines = ic.render_metrics_markdown(metrics, "###")
    md = "\n".join(lines)
    assert "silence_source" in md
    assert ip.SILENCE_SOURCE_DIGITAL_ZERO in md
    assert ip.SILENCE_SOURCE_LEAD_IN in md
    assert ip.SILENCE_SOURCE_TAIL in md
    assert ip.SILENCE_SOURCE_NONE in md


# ---------------------------------------------------------------------------
# CLI parser: --regression-budget
# ---------------------------------------------------------------------------


def test_select_cli_regression_budget_defaults_to_module_constant():
    parser = ic.build_arg_parser()
    args = parser.parse_args(["select", "--probe-manifest", "probes/val/manifest.csv"])
    assert args.regression_budget == ic.REGRESSION_BUDGET_PP


def test_select_cli_regression_budget_accepts_override():
    parser = ic.build_arg_parser()
    args = parser.parse_args(
        ["select", "--probe-manifest", "probes/val/manifest.csv", "--regression-budget", "2.5"]
    )
    assert args.regression_budget == 2.5


# ---------------------------------------------------------------------------
# decode_dataset_rows: stride-aware trailing silence (quartznet-promotion
# ticket 01). The forced-alignment frame indices are posterior frames, so the
# seconds conversion must scale by the model's total stride: 2x for a
# stride-2 model at the same frame index, and bit-identical to the old
# stride-1 arithmetic when the model has no total_stride attribute.
# ---------------------------------------------------------------------------


def _decode_trailing_silence(total_stride: int | None):
    transcript_raw = "call"
    text = optionb_normalize_text(prepare_ctc_transcript(transcript_raw))
    fe = LogMelFeatureExtractor()
    waveform = torch.zeros(16000)
    model = _TrailingSilenceStub(text, total_stride=total_stride)
    row = {
        "bucket": TARGET_BUCKET,
        "label": "CALL",
        "source_dataset": "optionb",
        "group_id": "",
        "transcript": transcript_raw,
    }
    dataset = _MiniDataset([row], [waveform])
    results, silences, _benchmark_logp = ic.decode_dataset_rows(
        model, fe, dataset, OPTIONB_GRAMMAR, beam_width=25, device="cpu"
    )
    return results, silences


def test_decode_dataset_rows_trailing_silence_stride1_is_unchanged_golden():
    results, silences = _decode_trailing_silence(total_stride=None)
    assert len(results) == 1
    fe = LogMelFeatureExtractor()
    t_frames = fe(torch.zeros(16000)).shape[-1]
    # "call" -> CTC path [c, a, l, blank, l] occupies frames 0..4, so the
    # natural trailing silence is the remaining t_frames - 5 posterior frames.
    # The division is innermost, matching the pre-ticket-01
    # `trailing_frames * FRAME_DURATION_S` arithmetic bit-for-bit.
    expected = (t_frames - 5) * (HOP_LENGTH / 16000)
    assert silences[0] == expected


def test_decode_dataset_rows_trailing_silence_stride2_is_2x_stride1():
    results_s1, silences_s1 = _decode_trailing_silence(total_stride=None)
    results_s2, silences_s2 = _decode_trailing_silence(total_stride=2)
    assert len(results_s2) == 1
    # Same frame index (the stub never subsamples), stride-2 hop: exactly 2x
    # the stride-1 seconds, bit-for-bit.
    assert silences_s2[0] == 2 * silences_s1[0]
    assert silences_s2[0] > 0.0
