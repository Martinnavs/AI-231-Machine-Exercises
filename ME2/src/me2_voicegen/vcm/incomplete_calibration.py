"""Step 5 (docs/INCOMPLETE-GRAMMAR-REJECTION.md) calibration/reporting tool
for the `required_command_margin` incomplete-prefix rejection gate.

Two CLI phases, run against `out/vcm/optionb-optiond/checkpoints/checkpoint.pt`
(preset optiond, epoch 71) at beam width 25:

    select  --split val   sweeps a margin grid on validation data (plus the
                           probe-manifest rows ticket 05 produces) and picks
                           one operating margin by a documented rule.
    holdout --split test  reads that choice back and reports it once, exactly,
                           against the disjoint held-out split.

Every metric function below is pure over already-decoded `evaluate.RowResult`
lists (plus, where noted, the raw manifest/probe-manifest dict rows joined by
list index) so the whole calibration policy is unit-testable without a
checkpoint, a GPU, or any audio file. Only the `run_select`/`run_holdout`
orchestration functions and `main()` touch a real model/dataset.

Probe-manifest contract (ticket 05 / docs/INCOMPLETE-GRAMMAR-REJECTION.md):
columns `filename, path, bucket="incomplete_prefix", label="", split,
group_id, source_dataset="optionb_incomplete_probe", source_filename,
source_path, source_transcript, prefix, prefix_word_count, char_overlap
(bool), grace_frames, crop_end_frame, crop_end_sample, trailing_silence_s,
silence_source, alignment_log_prob, aligner_checkpoint,
aligner_checkpoint_sha256`. `path` (and `source_path`) are relative to the
probe-manifest's own directory, not the val/test manifest's audio root --
`resolve_transcript` raises on `source_dataset ==
"optionb_incomplete_probe"`, so probe rows are loaded directly here, never
via `VCMDataset`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import statistics
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.common.grammar_core import Grammar
from me2_voicegen.vcm.decoder import DecodeResult, decode_utterance
from me2_voicegen.vcm.evaluate import (
    REJECT_PROBE_BUCKETS,
    TARGET_BUCKET,
    RowResult,
    _accepted,
    choose_operating_threshold,
    confusion_counts,
    slot_accuracy_breakdown,
    sweep_thresholds,
)
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb.incomplete_probes import (
    SILENCE_SOURCE_DIGITAL_ZERO,
    SILENCE_SOURCE_LEAD_IN,
    SILENCE_SOURCE_NONE,
    SILENCE_SOURCE_TAIL,
)
from me2_voicegen.vcm.optionb.text import normalize_text as optionb_normalize_text
from me2_voicegen.vcm.optionb.transcript import prepare_ctc_transcript
from me2_voicegen.vcm.pipeline import frame_hop_samples, load_checkpoint, logp_for_waveform
from me2_voicegen.vcm.segment_scorer import force_align

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST = PROJECT_ROOT / "out" / "conversions" / "v2" / "optionb" / "manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "out" / "vcm" / "optionb-optiond" / "checkpoints" / "checkpoint.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "vcm" / "optionb-optiond" / "metadata" / "incomplete_calibration"

BEAM_WIDTH = 25
REGRESSION_BUDGET_PP = 1.0
NEG_INF_THRESHOLD = float("-inf")

DEFAULT_MARGIN_GRID: tuple[float, ...] = (-10.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_TRAILING_SILENCE_BUCKET_EDGES: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0)
DEFAULT_BENCHMARK_SUBSET_SIZE = 50

HARDWARE_LABEL = "ai-n002.hpc.coe.upd.edu.ph (measured on this node)"

_MISSING_SILENCE_SOURCE = "<missing>"
"""Fallback key for `probe_far_by_silence_source` when a probe row has no
`silence_source` value at all -- distinct from any of ticket 05's real
`SILENCE_SOURCE_*` constants, so a genuinely missing column is never
silently folded into one of them."""

_SILENCE_SOURCE_DISPLAY_ORDER: tuple[str, ...] = (
    SILENCE_SOURCE_NONE,
    SILENCE_SOURCE_LEAD_IN,
    SILENCE_SOURCE_TAIL,
    SILENCE_SOURCE_DIGITAL_ZERO,
)
"""Preferred Markdown row order (imported from `incomplete_probes`, never
hardcoded literals) -- `probe_far_by_silence_source` itself groups by
whichever values actually appear in the input rows, so a value ticket 05
adds in the future still renders (appended, sorted) even before this order
tuple is updated."""


class NoEligibleMarginError(RuntimeError):
    """Raised by `select_margin` when no candidate margin satisfies the
    regression budget -- the CLI catches this, prints "no eligible margin",
    and exits non-zero rather than silently picking an out-of-budget value."""


# ---------------------------------------------------------------------------
# Grammar-derived command sets (never hardcoded phrase lists).
# ---------------------------------------------------------------------------


def single_word_commands(grammar: Grammar) -> frozenset[str]:
    """Accepted phrases that are exactly one word (e.g. "call", "pause")."""
    return frozenset(text for text, _, _ in grammar.all_phrases() if len(text.split()) == 1)


def strict_prefix_commands(grammar: Grammar) -> frozenset[str]:
    """Accepted phrases that are themselves a proper whole-word prefix of a
    DIFFERENT accepted phrase (e.g. "pause" prefixing "pause audio").
    Distinct from `Grammar.incomplete_prefixes`: these are commands, not
    rejection competitors -- `docs/INCOMPLETE-GRAMMAR-REJECTION.md`'s
    completion criterion 4 requires they keep decoding normally."""
    accepted = sorted({text for text, _, _ in grammar.all_phrases()})
    strict: set[str] = set()
    for short in accepted:
        short_words = short.split()
        for long in accepted:
            if long != short and long.split()[: len(short_words)] == short_words:
                strict.add(short)
                break
    return frozenset(strict)


def character_overlap_prefixes(grammar: Grammar) -> frozenset[str]:
    """Designated `incomplete_prefixes` that are also a character-level
    (non-word-boundary) prefix of some accepted phrase -- e.g. "remind" is a
    char-prefix of "reminder study" (continuation char "e", not a space),
    "reminder" is a char-prefix of "reminders" (continuation "s"), and
    "what" is a char-prefix of "what's the weather" (continuation "'").
    These are acoustically confusable with a DIFFERENT accepted phrase (and
    sometimes a different intent) beyond the ordinary word-boundary
    incomplete-prefix case, so they get their own always-visible report
    rows rather than being folded into the aggregate FAR."""
    accepted = {text for text, _, _ in grammar.all_phrases()}
    overlapping: set[str] = set()
    for prefix in grammar.incomplete_prefixes:
        for phrase in accepted:
            if len(phrase) > len(prefix) and phrase.startswith(prefix) and phrase[len(prefix)] != " ":
                overlapping.add(prefix)
                break
    return frozenset(overlapping)


# ---------------------------------------------------------------------------
# Trailing-silence bucketing.
# ---------------------------------------------------------------------------


def trailing_silence_bucket_label(
    seconds: float | None, edges: tuple[float, ...] = DEFAULT_TRAILING_SILENCE_BUCKET_EDGES
) -> str:
    if seconds is None:
        return "unknown"
    prev = 0.0
    for edge in edges:
        if seconds < edge:
            return f"[{prev:g}, {edge:g})s"
        prev = edge
    return f">={prev:g}s"


# ---------------------------------------------------------------------------
# RowResult construction shared by target/reject-probe and incomplete-prefix
# probe decoding (Established: `decode_split` returns no logp, so this
# module builds `RowResult` directly from its own logp -> decode loop).
# ---------------------------------------------------------------------------


def _row_result_from_decode(
    index: int, bucket: str, label: str, group_id: str | None, source_dataset: str | None, decoded: DecodeResult
) -> RowResult:
    confidence = None if decoded.intent is None else decoded.confidence
    return RowResult(
        index=index,
        bucket=bucket,
        label=label,
        text=decoded.text,
        intent=decoded.intent,
        confidence=confidence,
        group_id=group_id,
        source_dataset=source_dataset,
        slots=decoded.slots,
        rejection_reason=decoded.rejection_reason,
        incomplete_prefix=decoded.incomplete_prefix,
        incomplete_gap=decoded.incomplete_gap,
        command_raw_score=decoded.command_raw_score,
        incomplete_raw_score=decoded.incomplete_raw_score,
    )


@torch.no_grad()
def decode_dataset_rows(
    model,
    feature_extractor: LogMelFeatureExtractor,
    dataset,
    grammar: Grammar,
    beam_width: int,
    device: str | torch.device,
    benchmark_subset_size: int = DEFAULT_BENCHMARK_SUBSET_SIZE,
) -> tuple[list[RowResult], list[float | None], list[np.ndarray]]:
    """Decode every row of `dataset` once at `required_command_margin=None`,
    `threshold=-inf`. Returns `(results, target_trailing_silence_s,
    benchmark_logp_subset)`:

    - `target_trailing_silence_s[i]` is the force-aligned natural trailing
      silence (seconds) for `TARGET_BUCKET` Option B rows, `None` for every
      other row or on an alignment failure (Established:
      `segment_scorer.force_align` raises on an impossible alignment; that
      is caught and counted, not propagated). The seconds value scales by
      the model's total stride via `pipeline.frame_hop_samples`, so it is
      correct for strided (QuartzNet) checkpoints, not just stride-1.
    - `benchmark_logp_subset` is the first `benchmark_subset_size` target
      rows' raw `logp` arrays, kept for the micro-benchmark so it never
      re-runs the model forward pass.
    """
    results: list[RowResult] = []
    silences: list[float | None] = []
    benchmark_logp: list[np.ndarray] = []

    for i in range(len(dataset)):
        row = dataset.rows[i]
        example = dataset[i]
        logp = logp_for_waveform(model, feature_extractor, example.waveform, device=device)
        decoded = decode_utterance(
            logp, grammar, threshold=NEG_INF_THRESHOLD, beam_width=beam_width, required_command_margin=None
        )
        results.append(
            _row_result_from_decode(i, row["bucket"], row["label"], row.get("group_id"), row.get("source_dataset"), decoded)
        )

        trailing_s: float | None = None
        if row["bucket"] == TARGET_BUCKET and row.get("source_dataset") == "optionb":
            if len(benchmark_logp) < benchmark_subset_size:
                benchmark_logp.append(logp)
            try:
                transcript = optionb_normalize_text(prepare_ctc_transcript(row.get("transcript", "")))
                alignment = force_align(logp, transcript)
                # `alignment.end_frame` and `logp.shape[0]` are posterior
                # frames, so one frame spans `frame_hop_samples(model)`
                # waveform samples (10 ms for stride-1 MatchboxNet, 20 ms for
                # stride-2 QuartzNet). The `(hop / SAMPLE_RATE)` division is
                # kept innermost so the stride-1 value stays bit-identical to
                # the old `FRAME_DURATION_S` arithmetic.
                trailing_frames = logp.shape[0] - 1 - alignment.end_frame
                trailing_s = trailing_frames * (frame_hop_samples(model) / SAMPLE_RATE)
            except (ValueError, AssertionError):
                trailing_s = None
        silences.append(trailing_s)

    return results, silences, benchmark_logp


def load_probe_rows(probe_manifest_path: Path) -> list[dict]:
    with Path(probe_manifest_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def decode_probe_rows(
    model,
    feature_extractor: LogMelFeatureExtractor,
    probe_rows: list[dict],
    probe_manifest_path: Path,
    grammar: Grammar,
    beam_width: int,
    device: str | torch.device,
) -> list[RowResult]:
    """Decode every probe-manifest row once, same decode-time parameters as
    `decode_dataset_rows`. Loaded directly (never via `VCMDataset`) per this
    module's docstring. `path` is relative to `probe_manifest_path`'s own
    directory (the probe-manifest contract), not the val/test manifest's
    audio root."""
    audio_root = Path(probe_manifest_path).parent
    results: list[RowResult] = []
    for i, row in enumerate(probe_rows):
        waveform, sample_rate = torchaudio.load(str(audio_root / row["path"]))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        logp = logp_for_waveform(model, feature_extractor, waveform, device=device)
        decoded = decode_utterance(
            logp, grammar, threshold=NEG_INF_THRESHOLD, beam_width=beam_width, required_command_margin=None
        )
        results.append(
            _row_result_from_decode(
                i, "incomplete_prefix", "", row.get("group_id"), row.get("source_dataset"), decoded
            )
        )
    return results


# ---------------------------------------------------------------------------
# Pure metric functions.
# ---------------------------------------------------------------------------


def _accept_with_margin(r: RowResult, threshold: float, margin: float | None) -> bool:
    """`sweep_margins`'s own formula (`incomplete_gap is None or
    incomplete_gap >= margin`), ANDed with the confidence gate -- the same
    arithmetic decision `decoder.decode_utterance` makes live, verified by
    this ticket's own parity test against synthetic posteriors."""
    if not _accepted(r, threshold):
        return False
    if margin is None:
        return True
    return r.incomplete_gap is None or r.incomplete_gap >= margin


def _split_probes_by_digital_zero(
    probe_results: list[RowResult], probe_rows: list[dict]
) -> tuple[list[RowResult], list[dict], list[RowResult], list[dict]]:
    """Splits probes into (non-digital-zero, digital-zero-only) pairs, by
    `silence_source`. R2-6: digital-zero padding is not acoustically neutral
    (`LogMelFeatureExtractor` normalizes per-utterance), so its false-accept
    rate is not a fair measurement of the same thing room-tone-padded probes
    measure -- margin selection must not average the two together."""
    excl_results: list[RowResult] = []
    excl_rows: list[dict] = []
    dz_results: list[RowResult] = []
    dz_rows: list[dict] = []
    for r, row in zip(probe_results, probe_rows):
        if row.get("silence_source") == SILENCE_SOURCE_DIGITAL_ZERO:
            dz_results.append(r)
            dz_rows.append(row)
        else:
            excl_results.append(r)
            excl_rows.append(row)
    return excl_results, excl_rows, dz_results, dz_rows


def incomplete_prefix_far(
    probe_results: list[RowResult], probe_rows: list[dict], threshold: float, margin: float | None
) -> dict:
    """Overall + per-designated-prefix false-accept rate over probe rows,
    joined to `probe_rows` by list index. Every prefix gets its own row --
    none are folded into an "other" aggregate."""
    accepted_flags = [_accept_with_margin(r, threshold, margin) for r in probe_results]
    n = len(probe_results)
    n_false_accept = sum(accepted_flags)

    per_prefix: dict[str, dict] = {}
    for r, row, accepted in zip(probe_results, probe_rows, accepted_flags):
        prefix = row.get("prefix", "")
        entry = per_prefix.setdefault(prefix, {"n": 0, "n_false_accept": 0, "false_accept_intents": Counter()})
        entry["n"] += 1
        if accepted:
            entry["n_false_accept"] += 1
            entry["false_accept_intents"][r.intent or "UNKNOWN"] += 1

    per_prefix_out = {
        prefix: {
            "n": e["n"],
            "n_false_accept": e["n_false_accept"],
            "far": e["n_false_accept"] / e["n"] if e["n"] else None,
            "false_accept_intent_distribution": dict(sorted(e["false_accept_intents"].items())),
        }
        for prefix, e in sorted(per_prefix.items())
    }

    return {
        "n": n,
        "n_false_accept": n_false_accept,
        "far": n_false_accept / n if n else None,
        "per_prefix": per_prefix_out,
    }


def probe_far_by_silence_source(
    probe_results: list[RowResult], probe_rows: list[dict], threshold: float, margin: float | None
) -> dict[str, dict]:
    """False-accept rate over probe rows grouped by `silence_source`
    (ticket 05's `incomplete_probes.SILENCE_SOURCE_*` values: whichever the
    real probe manifest carries, not a hardcoded fixed set) -- R2-5's
    residual `digital_zero` share must be visible here on its own, not
    averaged into the aggregate FAR."""
    buckets: dict[str, dict] = {}
    for r, row in zip(probe_results, probe_rows):
        key = row.get("silence_source") or _MISSING_SILENCE_SOURCE
        entry = buckets.setdefault(key, {"n": 0, "n_false_accept": 0})
        entry["n"] += 1
        if _accept_with_margin(r, threshold, margin):
            entry["n_false_accept"] += 1
    return {
        key: {**e, "far": e["n_false_accept"] / e["n"] if e["n"] else None} for key, e in sorted(buckets.items())
    }


def _ordered_silence_source_keys(present: dict) -> list[str]:
    """`_SILENCE_SOURCE_DISPLAY_ORDER` first (for the values it names),
    then any other key (e.g. `_MISSING_SILENCE_SOURCE`, or a future
    `SILENCE_SOURCE_*` this module hasn't been updated for yet) sorted
    alphabetically -- so rendering never drops a value present in the
    data."""
    ordered = [k for k in _SILENCE_SOURCE_DISPLAY_ORDER if k in present]
    ordered += sorted(k for k in present if k not in _SILENCE_SOURCE_DISPLAY_ORDER)
    return ordered


def _true_phrase_for_row(raw_row: dict) -> str:
    return optionb_normalize_text(prepare_ctc_transcript(raw_row.get("transcript", "")))


def frr_for_phrase_set(
    target_results: list[RowResult], raw_rows: list[dict], phrase_set: frozenset[str], threshold: float, margin: float | None
) -> dict:
    """False-reject rate among `TARGET_BUCKET` rows whose true (resolved,
    normalized) transcript is exactly one of `phrase_set`'s phrases."""
    matching = [
        r
        for r in target_results
        if r.bucket == TARGET_BUCKET and _true_phrase_for_row(raw_rows[r.index]) in phrase_set
    ]
    n = len(matching)
    n_rejected = sum(1 for r in matching if not _accept_with_margin(r, threshold, margin))
    return {"n": n, "n_rejected": n_rejected, "frr": n_rejected / n if n else None}


def gate_caused_frr_by_competitor(target_results: list[RowResult], threshold: float, margin: float | None) -> dict:
    """Among `TARGET_BUCKET` rows the confidence gate alone would accept,
    the subset the margin gate additionally rejects, grouped by the winning
    competitor `incomplete_prefix`. `margin=None` (the gate-disabled
    baseline) always yields zero gate-caused rejections by definition --
    there is no gate to cause them."""
    confidence_accepted = [r for r in target_results if r.bucket == TARGET_BUCKET and _accepted(r, threshold)]
    by_competitor: dict[str, int] = defaultdict(int)
    n_gate_rejected = 0
    if margin is not None:
        for r in confidence_accepted:
            if r.incomplete_gap is not None and r.incomplete_gap < margin:
                n_gate_rejected += 1
                by_competitor[r.incomplete_prefix or "<none>"] += 1
    n = len(confidence_accepted)
    return {
        "n_confidence_accepted": n,
        "n_gate_rejected": n_gate_rejected,
        "gate_caused_frr": n_gate_rejected / n if n else None,
        "by_competitor": dict(sorted(by_competitor.items())),
    }


def gate_caused_frr_by_competitor_detailed(
    target_results: list[RowResult], threshold: float, margin: float | None, overlap_prefixes: frozenset[str]
) -> dict:
    """`gate_caused_frr_by_competitor`, with each `by_competitor` entry
    additionally carrying `char_overlap` (whether that competitor is one of
    `character_overlap_prefixes`) and `rejected_true_label_distribution`
    (the true `label`s of the target rows it caused to be rejected) -- the
    exact overlap-behavior visibility the user's R2-1 policy decision on the
    original incomplete-grammar-rejection review asked for."""
    summary = gate_caused_frr_by_competitor(target_results, threshold, margin)
    detail: dict[str, dict] = {}
    if margin is not None:
        confidence_accepted = [r for r in target_results if r.bucket == TARGET_BUCKET and _accepted(r, threshold)]
        for r in confidence_accepted:
            if r.incomplete_gap is not None and r.incomplete_gap < margin:
                key = r.incomplete_prefix or "<none>"
                entry = detail.setdefault(
                    key, {"n": 0, "char_overlap": key in overlap_prefixes, "rejected_true_label_distribution": Counter()}
                )
                entry["n"] += 1
                entry["rejected_true_label_distribution"][r.label] += 1
    summary["by_competitor"] = {
        k: {
            "n": v["n"],
            "char_overlap": v["char_overlap"],
            "rejected_true_label_distribution": dict(sorted(v["rejected_true_label_distribution"].items())),
        }
        for k, v in sorted(detail.items())
    }
    return summary


def false_accept_stats_with_margin(results: list[RowResult], threshold: float, margin: float | None, bucket: str) -> dict:
    rows = [r for r in results if r.bucket == bucket]
    n = len(rows)
    n_false_accept = sum(1 for r in rows if _accept_with_margin(r, threshold, margin))
    return {"n": n, "false_accepts": n_false_accept, "rate": n_false_accept / n if n else None}


def results_by_trailing_silence_bucket(
    results: list[RowResult],
    silences_by_index: dict[int, float | None],
    threshold: float,
    margin: float | None,
    edges: tuple[float, ...] = DEFAULT_TRAILING_SILENCE_BUCKET_EDGES,
) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for r in results:
        label = trailing_silence_bucket_label(silences_by_index.get(r.index), edges)
        entry = buckets.setdefault(label, {"n": 0, "n_accepted": 0})
        entry["n"] += 1
        if _accept_with_margin(r, threshold, margin):
            entry["n_accepted"] += 1
    return {
        label: {**e, "accept_rate": e["n_accepted"] / e["n"] if e["n"] else None} for label, e in sorted(buckets.items())
    }


def probe_silences_by_index(probe_rows: list[dict]) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for i, row in enumerate(probe_rows):
        raw = row.get("trailing_silence_s")
        out[i] = float(raw) if raw not in (None, "") else None
    return out


def count_alignment_failures(target_dataset_rows: list[dict], target_silences: list[float | None]) -> dict:
    attempted = [
        (row, s)
        for row, s in zip(target_dataset_rows, target_silences)
        if row.get("bucket") == TARGET_BUCKET and row.get("source_dataset") == "optionb"
    ]
    n_attempted = len(attempted)
    n_failed = sum(1 for _, s in attempted if s is None)
    return {"n_attempted": n_attempted, "n_failed": n_failed}


def load_probe_generation_failures(probe_manifest_path: Path) -> dict:
    """Ticket 05's `generation_report.json` (`incomplete_probes.
    build_generation_report`/`write_generation_report`), written next to its
    manifest with a `failure_counts` dict (`force_align_error`, `no_gap`,
    `crop_too_short`, `missing_audio`). Absent gracefully rather than
    hard-failing this tool -- ticket 05 is a separate parallel task and may
    not have landed its report in the exact same run."""
    report_path = Path(probe_manifest_path).parent / "generation_report.json"
    if not report_path.exists():
        return {"available": False, "path": str(report_path), "n_failures": 0, "by_reason": {}}
    report = json.loads(report_path.read_text())
    failure_counts: dict = report.get("failure_counts", {})
    return {
        "available": True,
        "path": str(report_path),
        "n_failures": sum(failure_counts.values()),
        "by_reason": dict(sorted(failure_counts.items())),
        "source_rows_selected": report.get("source_rows_selected"),
        "probe_rows_written": report.get("probe_rows_written"),
    }


# ---------------------------------------------------------------------------
# Margin-dependent metrics bundle -- one call per margin config (baseline
# `None` or a candidate/chosen numeric margin), so val and holdout render
# IDENTICAL structure at both configurations (R2-2: no metric is ever
# computed only at the chosen margin, with the aggregate FAR as the sole
# survivor elsewhere).
# ---------------------------------------------------------------------------


def _apply_margin_gate(results: list[RowResult], threshold: float, margin: float | None) -> list[RowResult]:
    """Rows the combined confidence+margin gate would reject get `intent`/
    `confidence`/`slots` cleared, so `evaluate.confusion_counts`/
    `slot_accuracy_breakdown` (which only know about the plain confidence
    threshold) report exactly the margin-gated outcome without needing their
    own margin awareness."""
    return [
        r if _accept_with_margin(r, threshold, margin) else dataclasses.replace(r, intent=None, confidence=None, slots={})
        for r in results
    ]


def build_metrics(
    *,
    target_results: list[RowResult],
    probe_results: list[RowResult],
    probe_rows: list[dict],
    raw_target_rows: list[dict],
    reject_results_all: list[RowResult],
    target_silences_by_index: dict[int, float | None],
    probe_silences: dict[int, float | None],
    threshold: float,
    margin: float | None,
    grammar: Grammar,
) -> dict:
    single_word = single_word_commands(grammar)
    strict_prefix = strict_prefix_commands(grammar)
    overlap_prefixes = character_overlap_prefixes(grammar)
    gated_target_results = _apply_margin_gate(target_results, threshold, margin)

    n_target = len(target_results)
    n_target_exact = sum(1 for r in target_results if _accept_with_margin(r, threshold, margin) and r.intent == r.label)

    return {
        "margin": margin,
        "is_baseline": margin is None,
        "target_exact_accuracy": {
            "n": n_target,
            "n_exact_correct": n_target_exact,
            "exact_accuracy": n_target_exact / n_target if n_target else None,
        },
        "confusion_counts": confusion_counts(gated_target_results, threshold),
        "slot_accuracy": slot_accuracy_breakdown(gated_target_results, raw_target_rows, grammar, threshold),
        "incomplete_prefix_far": incomplete_prefix_far(probe_results, probe_rows, threshold, margin),
        "probe_far_by_silence_source": probe_far_by_silence_source(probe_results, probe_rows, threshold, margin),
        "character_overlap_prefixes": sorted(overlap_prefixes),
        "frr_single_word_commands": frr_for_phrase_set(target_results, raw_target_rows, single_word, threshold, margin),
        "frr_strict_prefix_commands": frr_for_phrase_set(target_results, raw_target_rows, strict_prefix, threshold, margin),
        "gate_caused_frr_by_competitor": gate_caused_frr_by_competitor_detailed(
            target_results, threshold, margin, overlap_prefixes
        ),
        "reject_probe_far": {
            bucket: false_accept_stats_with_margin(reject_results_all, threshold, margin, bucket)
            for bucket in REJECT_PROBE_BUCKETS
        },
        "target_by_silence_bucket": results_by_trailing_silence_bucket(
            target_results, target_silences_by_index, threshold, margin
        ),
        "probe_by_silence_bucket": results_by_trailing_silence_bucket(probe_results, probe_silences, threshold, margin),
    }


# ---------------------------------------------------------------------------
# Margin grid + selection.
# ---------------------------------------------------------------------------


def evaluate_margin_grid(
    target_results: list[RowResult],
    probe_results: list[RowResult],
    probe_rows: list[dict],
    threshold: float,
    margins: tuple[float, ...] = DEFAULT_MARGIN_GRID,
) -> list[dict]:
    """One row per candidate margin plus a `margin=None` (gate-disabled)
    baseline row, in that order. Pure arithmetic over already-decoded
    `required_command_margin=None` results -- no re-decoding. Every row
    also carries the full per-prefix FAR breakdown and the gate-caused FRR
    total, not just the aggregate FAR + accuracy `select_margin` needs --
    per docs/INCOMPLETE-GRAMMAR-REJECTION.md's Step 5 report requirement
    ("for each margin candidate")."""
    excl_results, excl_rows, dz_results, dz_rows = _split_probes_by_digital_zero(probe_results, probe_rows)

    rows: list[dict] = []
    for margin in (None, *margins):
        n_target = len(target_results)
        n_target_exact = sum(
            1 for r in target_results if _accept_with_margin(r, threshold, margin) and r.intent == r.label
        )
        far = incomplete_prefix_far(probe_results, probe_rows, threshold, margin)
        far_excl_dz = incomplete_prefix_far(excl_results, excl_rows, threshold, margin)
        far_dz_only = incomplete_prefix_far(dz_results, dz_rows, threshold, margin)
        gate_frr = gate_caused_frr_by_competitor(target_results, threshold, margin)
        rows.append(
            {
                "margin": margin,
                "is_baseline": margin is None,
                "n_target": n_target,
                "target_exact_accuracy": n_target_exact / n_target if n_target else None,
                "n_probe": far["n"],
                "incomplete_prefix_far": far["far"],
                "per_prefix_far": far["per_prefix"],
                # R2-6: selection uses this field, not "incomplete_prefix_far" above --
                # digital-zero padding is excluded because it is not acoustically
                # neutral and was found to understate the false-accept rate.
                "n_probe_excl_digital_zero": far_excl_dz["n"],
                "incomplete_prefix_far_excl_digital_zero": far_excl_dz["far"],
                "n_probe_digital_zero_only": far_dz_only["n"],
                "incomplete_prefix_far_digital_zero_only": far_dz_only["far"],
                "gate_caused_frr": gate_frr["gate_caused_frr"],
                "n_gate_rejected": gate_frr["n_gate_rejected"],
            }
        )
    return rows


MARGIN_SELECTION_RULE = (
    "Among margins whose val target exact-accuracy drop from the gate-disabled baseline is at "
    "most the regression budget, pick the lowest incomplete_prefix_far_excl_digital_zero (NOT "
    "incomplete_prefix_far, which includes digital-zero-padded probes); ties break toward the "
    "smallest margin. R2-6: digital-zero padding is not acoustically neutral and was found to "
    "understate the false-accept rate, biasing selection toward too small a margin -- excluded "
    "from the selection criterion. incomplete_prefix_far (all probes) and "
    "incomplete_prefix_far_digital_zero_only are still reported on every grid row for visibility."
)


def select_margin(grid_rows: list[dict], regression_budget_pp: float = REGRESSION_BUDGET_PP) -> dict:
    """Selection rule (ticket, updated by R2-6 -- see `MARGIN_SELECTION_RULE`):
    among margins whose val target exact-accuracy drop from the gate-disabled
    baseline is at most `regression_budget_pp` percentage points, pick the
    lowest incomplete-prefix FAR excluding digital-zero-padded probes; ties
    break toward the smallest margin. Raises `NoEligibleMarginError` if no
    margin qualifies."""
    baseline = next(row for row in grid_rows if row["is_baseline"])
    baseline_acc = baseline["target_exact_accuracy"]
    candidates = [row for row in grid_rows if not row["is_baseline"]]

    eligible = [
        row
        for row in candidates
        if baseline_acc is not None
        and row["target_exact_accuracy"] is not None
        and (baseline_acc - row["target_exact_accuracy"]) * 100.0 <= regression_budget_pp
        and row["incomplete_prefix_far_excl_digital_zero"] is not None
    ]
    if not eligible:
        raise NoEligibleMarginError(
            "no eligible margin: every candidate margin's val target exact-accuracy "
            f"drop from the gate-disabled baseline ({baseline_acc!r}) exceeds the "
            f"{regression_budget_pp}pp regression budget, or has no non-digital-zero probes "
            "to compute incomplete_prefix_far_excl_digital_zero from"
        )

    best_far = min(row["incomplete_prefix_far_excl_digital_zero"] for row in eligible)
    tied = [row for row in eligible if row["incomplete_prefix_far_excl_digital_zero"] == best_far]
    return min(tied, key=lambda row: row["margin"])


# ---------------------------------------------------------------------------
# Checkpoint/ONNX hashing + holdout guard.
# ---------------------------------------------------------------------------


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_checkpoint_and_onnx(checkpoint_path: str | Path, onnx_dir: str | Path) -> dict[str, str]:
    hashes = {"checkpoint": sha256_of_file(checkpoint_path)}
    onnx_dir = Path(onnx_dir)
    if onnx_dir.is_dir():
        for onnx_path in sorted(onnx_dir.glob("*.onnx")):
            hashes[onnx_path.name] = sha256_of_file(onnx_path)
    return hashes


def check_holdout_can_run(report_json_path: str | Path, selection: dict, computed_hashes: dict[str, str]) -> None:
    """Raises `SystemExit` (non-zero exit for CLI callers) rather than
    returning a bool: both refusal reasons are meant to hard-stop the
    holdout run, never be silently swallowed by a caller that forgets to
    check a return value."""
    report_json_path = Path(report_json_path)
    if report_json_path.exists():
        raise SystemExit(f"holdout report already exists at {report_json_path}; refusing to re-run")
    expected = selection.get("checkpoint_sha256")
    actual = computed_hashes.get("checkpoint")
    if expected is None or actual is None or expected != actual:
        raise SystemExit(
            f"checkpoint sha256 mismatch: selection JSON records {expected!r}, "
            f"but the checkpoint at hand hashes to {actual!r}"
        )


# ---------------------------------------------------------------------------
# Contamination / operating-point-drift guards (R2-3).
# ---------------------------------------------------------------------------


def validate_probe_split(probe_rows: list[dict], expected_split: str) -> None:
    """Refuses if any probe row's own `split` column disagrees with the
    phase being run -- prevents silently tuning the margin on held-out
    (`test`) probe rows, or reporting `holdout` against `val` probes."""
    bad = sorted({row.get("split") for row in probe_rows} - {expected_split})
    if bad:
        raise SystemExit(
            f"probe manifest contains split(s) {bad}, expected only {expected_split!r} for this phase"
        )


def validate_aligner_checkpoint(probe_rows: list[dict], checkpoint_sha256: str) -> None:
    """Refuses if any probe row's `aligner_checkpoint_sha256` disagrees with
    the checkpoint being calibrated/reported -- a probe manifest generated
    against a different checkpoint's forced alignments is not a valid input
    here."""
    mismatched = sorted({row.get("aligner_checkpoint_sha256") for row in probe_rows} - {checkpoint_sha256})
    if mismatched:
        raise SystemExit(
            f"probe manifest aligner_checkpoint_sha256 {mismatched} does not match "
            f"the checkpoint being calibrated ({checkpoint_sha256!r})"
        )


def check_select_can_run(out_dir: str | Path) -> None:
    """Refuses to run/overwrite `val_selection.json` if a holdout report
    already exists in `out_dir` -- re-running `select` after `holdout` has
    reported would be re-tuning the margin after seeing held-out results."""
    test_report_path = Path(out_dir) / "test_report.json"
    if test_report_path.exists():
        raise SystemExit(
            f"a holdout report already exists at {test_report_path}; refusing to re-run "
            "select and overwrite val_selection.json after held-out results are visible"
        )


def resolve_holdout_beam_width(cli_beam_width: int | None, selection_beam_width: int) -> int:
    """`holdout` has no independent beam-width choice: `None` (not passed on
    the CLI) inherits the val selection's beam width; an explicitly passed
    value that disagrees with it is refused rather than silently reporting
    at a different operating point than the one calibrated."""
    if cli_beam_width is None:
        return selection_beam_width
    if cli_beam_width != selection_beam_width:
        raise SystemExit(
            f"--beam-width {cli_beam_width} does not match the val selection's "
            f"beam_width {selection_beam_width}"
        )
    return cli_beam_width


# ---------------------------------------------------------------------------
# Micro-benchmark.
# ---------------------------------------------------------------------------


def benchmark_decode(
    logp_arrays: list[np.ndarray],
    grammar: Grammar,
    threshold: float,
    beam_width: int,
    required_command_margin: float | None,
    hardware_label: str = HARDWARE_LABEL,
) -> dict:
    """Median/p95 wall time (`perf_counter`) and peak memory (`tracemalloc`)
    of `decode_utterance` over `logp_arrays`, at one gate configuration.
    Labeled per `benchmark.py`'s convention: every number here is measured
    on this node, never presented as a claim about any other hardware."""
    if not logp_arrays:
        return {
            "hardware_label": hardware_label,
            "n": 0,
            "median_latency_ms": None,
            "p95_latency_ms": None,
            "peak_memory_bytes": None,
        }

    tracemalloc.start()
    latencies_s: list[float] = []
    for logp in logp_arrays:
        t0 = time.perf_counter()
        decode_utterance(
            logp, grammar, threshold, beam_width=beam_width, required_command_margin=required_command_margin
        )
        t1 = time.perf_counter()
        latencies_s.append(t1 - t0)
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies_s.sort()
    p95_index = min(len(latencies_s) - 1, round(0.95 * (len(latencies_s) - 1)))
    return {
        "hardware_label": hardware_label,
        "n": len(logp_arrays),
        "median_latency_ms": statistics.median(latencies_s) * 1000.0,
        "p95_latency_ms": latencies_s[p95_index] * 1000.0,
        "peak_memory_bytes": peak_bytes,
    }


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------


def render_metrics_markdown(metrics: dict, heading_prefix: str) -> list[str]:
    """One full metrics block (all of R2-2's stratified breakdowns) for a
    single margin config (`metrics["margin"]`). Shared verbatim between
    `render_val_markdown` (baseline + chosen-margin blocks) and
    `render_holdout_markdown` (baseline + gated blocks), so held-out output
    is never a stripped-down aggregate-only version of the val report."""
    lines: list[str] = []
    label = "baseline (gate off)" if metrics["is_baseline"] else f"margin={metrics['margin']}"
    lines.append(f"{heading_prefix} {label}")
    lines.append("")

    acc = metrics["target_exact_accuracy"]
    lines.append(f"Target exact accuracy: {acc['n_exact_correct']}/{acc['n']} ({acc['exact_accuracy']})")
    lines.append("")

    slot_acc = metrics["slot_accuracy"]
    if slot_acc is not None:
        rate = slot_acc["slot_exact_match_rate_given_intent_correct"]
        lines.append(
            f"Slot exact match (given intent correct): {slot_acc['n_intent_and_slots_correct']}/"
            f"{slot_acc['n_intent_correct']} ({rate}), over {slot_acc['n_slot_bearing_target_rows']} "
            f"slot-bearing target rows ({slot_acc['n_unparseable_ground_truth']} unparseable ground truth)"
        )
    else:
        lines.append("Slot exact match: n/a (no classifiable slot-bearing Option B target row)")
    lines.append("")

    far = metrics["incomplete_prefix_far"]
    lines.append(f"Incomplete-prefix FAR: {far['n_false_accept']}/{far['n']} ({far['far']})")
    lines.append("")
    lines.append("| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |")
    lines.append("|---|---|---|---|---|---|")
    overlap = set(metrics["character_overlap_prefixes"])
    for prefix, stats in far["per_prefix"].items():
        flag = "YES" if prefix in overlap else ""
        intents = ", ".join(f"{k}={v}" for k, v in stats["false_accept_intent_distribution"].items()) or "-"
        lines.append(f"| {prefix} | {flag} | {stats['n']} | {stats['n_false_accept']} | {stats['far']} | {intents} |")
    lines.append("")

    by_source = metrics["probe_far_by_silence_source"]
    lines.append(
        "Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a "
        "dataset-limitation fallback, not a neutral padding choice -- its "
        "share and FAR must stay visible on their own, never averaged away):"
    )
    lines.append("")
    lines.append("| silence_source | n | n_false_accept | far |")
    lines.append("|---|---|---|---|")
    for key in _ordered_silence_source_keys(by_source):
        stats = by_source[key]
        lines.append(f"| {key} | {stats['n']} | {stats['n_false_accept']} | {stats['far']} |")
    lines.append("")

    swc = metrics["frr_single_word_commands"]
    spc = metrics["frr_strict_prefix_commands"]
    lines.append(f"FRR (single-word commands): {swc['n_rejected']}/{swc['n']} ({swc['frr']})")
    lines.append(f"FRR (strict-prefix commands): {spc['n_rejected']}/{spc['n']} ({spc['frr']})")
    lines.append("")

    gate_frr = metrics["gate_caused_frr_by_competitor"]
    lines.append(
        f"Gate-caused FRR: {gate_frr['n_gate_rejected']}/{gate_frr['n_confidence_accepted']} "
        f"confidence-accepted target rows additionally rejected ({gate_frr['gate_caused_frr']})"
    )
    lines.append("")
    lines.append("| competitor | char_overlap | n | rejected true-label distribution |")
    lines.append("|---|---|---|---|")
    for competitor, stats in gate_frr["by_competitor"].items():
        labels = ", ".join(f"{k}={v}" for k, v in stats["rejected_true_label_distribution"].items()) or "-"
        lines.append(f"| {competitor} | {'YES' if stats['char_overlap'] else ''} | {stats['n']} | {labels} |")
    lines.append("")

    lines.append("Babble/silence FAR:")
    for bucket in REJECT_PROBE_BUCKETS:
        stats = metrics["reject_probe_far"][bucket]
        lines.append(f"- {bucket}: {stats['false_accepts']}/{stats['n']} ({stats['rate']})")
    lines.append("")

    lines.append("Target rows by trailing-silence bucket (natural, force-aligned):")
    lines.append("")
    lines.append("| bucket | n | n_accepted | accept_rate |")
    lines.append("|---|---|---|---|")
    for bucket_label, stats in metrics["target_by_silence_bucket"].items():
        lines.append(f"| {bucket_label} | {stats['n']} | {stats['n_accepted']} | {stats['accept_rate']} |")
    lines.append("")
    lines.append("Probe rows by trailing-silence bucket (crafted):")
    lines.append("")
    lines.append("| bucket | n | n_accepted | accept_rate |")
    lines.append("|---|---|---|---|")
    for bucket_label, stats in metrics["probe_by_silence_bucket"].items():
        lines.append(f"| {bucket_label} | {stats['n']} | {stats['n_accepted']} | {stats['accept_rate']} |")
    lines.append("")

    return lines


def render_val_markdown(report: dict) -> str:
    lines: list[str] = [
        "# Incomplete-prefix rejection gate -- val calibration report",
        "",
        f"Checkpoint: `{report['checkpoint_path']}`  Beam width: {report['beam_width']}",
        f"Chosen threshold (tau): {report['threshold']}",
        "",
    ]

    def _fmt(v: float | None) -> str:
        return ("%.4f" % v) if v is not None else "n/a"

    grid_header = (
        "| margin | target_exact_accuracy | far_excl_digital_zero (selection) | far_all_probes | "
        "far_digital_zero_only | gate_caused_frr |"
    )
    grid_sep = "|---|---|---|---|---|---|"

    if report.get("no_eligible_margin"):
        lines.append(f"**No eligible margin: {report['no_eligible_margin']}**")
        lines.append("")
        lines.append("## Margin grid")
        lines.append("")
        lines.append(
            f"Selection rule: {report.get('margin_selection_rule', MARGIN_SELECTION_RULE)} "
            f"(regression_budget_pp={report.get('regression_budget_pp', REGRESSION_BUDGET_PP)})"
        )
        lines.append("")
        lines.append(grid_header)
        lines.append(grid_sep)
        for row in report["margin_grid"]:
            margin_str = "baseline (gate off)" if row["is_baseline"] else str(row["margin"])
            lines.append(
                f"| {margin_str} | {_fmt(row['target_exact_accuracy'])} | "
                f"{_fmt(row['incomplete_prefix_far_excl_digital_zero'])} | {_fmt(row['incomplete_prefix_far'])} | "
                f"{_fmt(row['incomplete_prefix_far_digital_zero_only'])} | {row.get('gate_caused_frr')} |"
            )
        lines.append("")
        return "\n".join(lines) + "\n"

    chosen = report["chosen_margin"]
    lines.append(
        f"Chosen margin: **{chosen['margin']}** "
        f"(val target_exact_accuracy={chosen['target_exact_accuracy']:.4f}, "
        f"val incomplete_prefix_far_excl_digital_zero={_fmt(chosen['incomplete_prefix_far_excl_digital_zero'])} "
        f"[selection criterion], val incomplete_prefix_far_all_probes={_fmt(chosen['incomplete_prefix_far'])})"
    )
    lines.append("")
    lines.append("## Margin grid")
    lines.append("")
    lines.append(
        f"Selection rule: {report.get('margin_selection_rule', MARGIN_SELECTION_RULE)} "
        f"(regression_budget_pp={report.get('regression_budget_pp', REGRESSION_BUDGET_PP)})"
    )
    lines.append("")
    lines.append(grid_header)
    lines.append(grid_sep)
    for row in report["margin_grid"]:
        margin_str = "baseline (gate off)" if row["is_baseline"] else str(row["margin"])
        lines.append(
            f"| {margin_str} | {_fmt(row['target_exact_accuracy'])} | "
            f"{_fmt(row['incomplete_prefix_far_excl_digital_zero'])} | {_fmt(row['incomplete_prefix_far'])} | "
            f"{_fmt(row['incomplete_prefix_far_digital_zero_only'])} | {row.get('gate_caused_frr')} |"
        )
    lines.append("")

    lines.append("## Baseline vs. chosen-margin metrics")
    lines.append("")
    lines.extend(render_metrics_markdown(report["baseline_metrics"], "###"))
    lines.extend(render_metrics_markdown(report["gated_metrics"], "###"))

    lines.append("## Decode latency/peak memory: gate off vs. chosen margin")
    lines.append("")
    off = report["benchmark_gate_off"]
    on = report["benchmark_gate_chosen_margin"]
    lines.append(f"Hardware: {off['hardware_label']}")
    lines.append("")
    lines.append("| config | n | median latency (ms) | p95 latency (ms) | peak memory (bytes) |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| gate off | {off['n']} | {off['median_latency_ms']} | {off['p95_latency_ms']} | {off['peak_memory_bytes']} |")
    lines.append(f"| margin={chosen['margin']} | {on['n']} | {on['median_latency_ms']} | {on['p95_latency_ms']} | {on['peak_memory_bytes']} |")
    lines.append("")

    lines.append("## Alignment / probe-generation failures")
    lines.append("")
    align = report["target_alignment_failures"]
    probegen = report["probe_generation_failures"]
    lines.append(f"- target forced-alignment: {align['n_failed']}/{align['n_attempted']} failed")
    if probegen["available"]:
        lines.append(f"- probe generation: {probegen['n_failures']} failures: {probegen['by_reason']}")
    else:
        lines.append(f"- probe generation: no failures log found at `{probegen['path']}`")
    lines.append("")

    lines.append("## Checkpoint/ONNX hashes")
    lines.append("")
    lines.append(f"Before: {report['hashes_before']}")
    lines.append(f"After:  {report['hashes_after']}")
    lines.append(f"Unchanged: {report['hashes_before'] == report['hashes_after']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def build_val_report(
    *,
    checkpoint_path: Path,
    beam_width: int,
    threshold: float,
    margin_grid: list[dict],
    grammar: Grammar,
    target_results: list[RowResult],
    probe_results: list[RowResult],
    probe_rows: list[dict],
    probe_manifest_path: Path,
    raw_target_rows: list[dict],
    target_silences: list[float | None],
    reject_results_all: list[RowResult],
    benchmark_gate_off: dict,
    benchmark_gate_chosen_margin: dict,
    alignment_failures: dict,
    probe_generation_failures: dict,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    regression_budget_pp: float = REGRESSION_BUDGET_PP,
) -> dict:
    base: dict = {
        "checkpoint_path": str(checkpoint_path),
        "beam_width": beam_width,
        "threshold": threshold,
        "regression_budget_pp": regression_budget_pp,
        "margin_selection_rule": MARGIN_SELECTION_RULE,
        "margin_grid": margin_grid,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "checkpoint_sha256": hashes_before.get("checkpoint"),
        "probe_manifest_path": str(probe_manifest_path),
        "probe_manifest_aligner_checkpoint_sha256": (
            probe_rows[0].get("aligner_checkpoint_sha256") if probe_rows else None
        ),
    }

    try:
        chosen = select_margin(margin_grid, regression_budget_pp)
    except NoEligibleMarginError as exc:
        base["no_eligible_margin"] = str(exc)
        return base

    margin = chosen["margin"]
    target_silences_by_index = {i: s for i, s in enumerate(target_silences)}
    probe_silences = probe_silences_by_index(probe_rows)

    metrics_kwargs = dict(
        target_results=target_results,
        probe_results=probe_results,
        probe_rows=probe_rows,
        raw_target_rows=raw_target_rows,
        reject_results_all=reject_results_all,
        target_silences_by_index=target_silences_by_index,
        probe_silences=probe_silences,
        threshold=threshold,
        grammar=grammar,
    )

    base.update(
        {
            "chosen_margin": chosen,
            "baseline_metrics": build_metrics(margin=None, **metrics_kwargs),
            "gated_metrics": build_metrics(margin=margin, **metrics_kwargs),
            "benchmark_gate_off": benchmark_gate_off,
            "benchmark_gate_chosen_margin": benchmark_gate_chosen_margin,
            "target_alignment_failures": alignment_failures,
            "probe_generation_failures": probe_generation_failures,
        }
    )
    return base


# ---------------------------------------------------------------------------
# CLI orchestration (touches a real model/dataset; not unit-tested directly).
# ---------------------------------------------------------------------------


def _load_val_datasets(manifest_path: Path):
    from me2_voicegen.vcm.dataset import VCMDataset

    return VCMDataset(manifest_path, split="val", augmenter=None)


def render_holdout_markdown(report: dict) -> str:
    """Full parity with `render_val_markdown`'s per-margin metrics blocks
    (R2-2(c)): the held-out report is never reduced to the aggregate FAR
    alone."""
    lines: list[str] = [
        "# Incomplete-prefix rejection gate -- held-out test report",
        "",
        f"Checkpoint: `{report['checkpoint_path']}`  Beam width: {report['beam_width']}",
        f"Margin (from val selection): {report['margin']}  Threshold: {report['threshold']}",
        f"Val selection: `{report['val_selection_path']}`",
        "",
    ]
    lines.extend(render_metrics_markdown(report["baseline_metrics"], "##"))
    lines.extend(render_metrics_markdown(report["gated_metrics"], "##"))

    lines.append("## Checkpoint/ONNX hashes")
    lines.append("")
    lines.append(f"Before: {report['hashes_before']}")
    lines.append(f"After:  {report['hashes_after']}")
    lines.append(f"Unchanged: {report['hashes_before'] == report['hashes_after']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_select(args: argparse.Namespace) -> int:
    check_select_can_run(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = args.checkpoint.parent.parent / "export"
    hashes_before = hash_checkpoint_and_onnx(args.checkpoint, onnx_dir)

    model, _ckpt = load_checkpoint(args.checkpoint, device=args.device)
    feature_extractor = LogMelFeatureExtractor()
    grammar = OPTIONB_GRAMMAR

    val_dataset = _load_val_datasets(args.manifest)
    all_results, target_silences, benchmark_logp = decode_dataset_rows(
        model, feature_extractor, val_dataset, grammar, args.beam_width, args.device
    )
    target_results = [r for r in all_results if r.bucket == TARGET_BUCKET]

    probe_rows = load_probe_rows(args.probe_manifest)
    validate_probe_split(probe_rows, "val")
    validate_aligner_checkpoint(probe_rows, hashes_before["checkpoint"])
    probe_results = decode_probe_rows(
        model, feature_extractor, probe_rows, args.probe_manifest, grammar, args.beam_width, args.device
    )

    sweep = sweep_thresholds(all_results)
    chosen_threshold_row = choose_operating_threshold(sweep)
    threshold = chosen_threshold_row["threshold"]

    margins = args.margins if args.margins else DEFAULT_MARGIN_GRID
    margin_grid = evaluate_margin_grid(target_results, probe_results, probe_rows, threshold, margins)

    alignment_failures = count_alignment_failures(val_dataset.rows, target_silences)
    probe_generation_failures = load_probe_generation_failures(args.probe_manifest)

    try:
        chosen = select_margin(margin_grid, args.regression_budget)
    except NoEligibleMarginError:
        chosen = None

    if chosen is not None:
        margin = chosen["margin"]
        benchmark_gate_off = benchmark_decode(benchmark_logp, grammar, threshold, args.beam_width, None)
        benchmark_gate_chosen_margin = benchmark_decode(benchmark_logp, grammar, threshold, args.beam_width, margin)
    else:
        benchmark_gate_off = benchmark_decode(benchmark_logp, grammar, threshold, args.beam_width, None)
        benchmark_gate_chosen_margin = benchmark_gate_off

    hashes_after = hash_checkpoint_and_onnx(args.checkpoint, onnx_dir)

    report = build_val_report(
        checkpoint_path=args.checkpoint,
        beam_width=args.beam_width,
        threshold=threshold,
        margin_grid=margin_grid,
        grammar=grammar,
        target_results=target_results,
        probe_results=probe_results,
        probe_rows=probe_rows,
        probe_manifest_path=args.probe_manifest,
        raw_target_rows=val_dataset.rows,
        target_silences=target_silences,
        reject_results_all=all_results,
        benchmark_gate_off=benchmark_gate_off,
        benchmark_gate_chosen_margin=benchmark_gate_chosen_margin,
        alignment_failures=alignment_failures,
        probe_generation_failures=probe_generation_failures,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        regression_budget_pp=args.regression_budget,
    )

    json_path = args.out_dir / "val_selection.json"
    md_path = args.out_dir / "val_report.md"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)
    md_path.write_text(render_val_markdown(report))
    print(f"wrote {json_path} and {md_path}")

    if chosen is None:
        print(f"no eligible margin: {report['no_eligible_margin']}")
        return 1
    print(f"chosen margin: {chosen['margin']}")
    return 0


def run_holdout(args: argparse.Namespace) -> int:
    selection = json.loads(Path(args.selection).read_text())
    if selection.get("no_eligible_margin"):
        raise SystemExit(f"val selection recorded no eligible margin: {selection['no_eligible_margin']}")
    margin = selection["chosen_margin"]["margin"]
    threshold = selection["threshold"]
    beam_width = resolve_holdout_beam_width(args.beam_width, selection["beam_width"])

    report_json_path = args.out_dir / "test_report.json"
    onnx_dir = args.checkpoint.parent.parent / "export"
    computed_hashes = hash_checkpoint_and_onnx(args.checkpoint, onnx_dir)
    check_holdout_can_run(report_json_path, selection, computed_hashes)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, _ckpt = load_checkpoint(args.checkpoint, device=args.device)
    feature_extractor = LogMelFeatureExtractor()
    grammar = OPTIONB_GRAMMAR

    from me2_voicegen.vcm.dataset import VCMDataset

    test_dataset = VCMDataset(args.manifest, split="test", augmenter=None)
    all_results, target_silences, _benchmark_logp = decode_dataset_rows(
        model, feature_extractor, test_dataset, grammar, beam_width, args.device
    )
    target_results = [r for r in all_results if r.bucket == TARGET_BUCKET]

    probe_rows = load_probe_rows(args.probe_manifest)
    validate_probe_split(probe_rows, "test")
    validate_aligner_checkpoint(probe_rows, computed_hashes["checkpoint"])
    probe_results = decode_probe_rows(
        model, feature_extractor, probe_rows, args.probe_manifest, grammar, beam_width, args.device
    )

    target_silences_by_index = {i: s for i, s in enumerate(target_silences)}
    probe_silences = probe_silences_by_index(probe_rows)
    metrics_kwargs = dict(
        target_results=target_results,
        probe_results=probe_results,
        probe_rows=probe_rows,
        raw_target_rows=test_dataset.rows,
        reject_results_all=all_results,
        target_silences_by_index=target_silences_by_index,
        probe_silences=probe_silences,
        threshold=threshold,
        grammar=grammar,
    )

    report = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": computed_hashes.get("checkpoint"),
        "beam_width": beam_width,
        "threshold": threshold,
        "margin": margin,
        "baseline_metrics": build_metrics(margin=None, **metrics_kwargs),
        "gated_metrics": build_metrics(margin=margin, **metrics_kwargs),
        "hashes_before": computed_hashes,
        "hashes_after": hash_checkpoint_and_onnx(args.checkpoint, onnx_dir),
        "val_selection_path": str(args.selection),
    }

    md_path = args.out_dir / "test_report.md"
    with report_json_path.open("w") as f:
        json.dump(report, f, indent=2)
    md_path.write_text(render_holdout_markdown(report))
    print(f"wrote {report_json_path} and {md_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select_p = sub.add_parser("select", help="calibrate the margin on --split val")
    select_p.add_argument("--split", choices=["val"], default="val")
    select_p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    select_p.add_argument("--probe-manifest", type=Path, required=True)
    select_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    select_p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    select_p.add_argument("--device", default="cpu")
    select_p.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    select_p.add_argument(
        "--margins", type=str, default=None, help="comma-separated float overrides for the margin grid"
    )
    select_p.add_argument(
        "--regression-budget",
        type=float,
        default=REGRESSION_BUDGET_PP,
        help="max acceptable val target exact-accuracy drop (pp) from the gate-disabled baseline "
        f"when selecting a margin (default: {REGRESSION_BUDGET_PP})",
    )
    select_p.set_defaults(func=run_select)

    holdout_p = sub.add_parser("holdout", help="report once against --split test")
    holdout_p.add_argument("--split", choices=["test"], default="test")
    holdout_p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    holdout_p.add_argument("--probe-manifest", type=Path, required=True)
    holdout_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    holdout_p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    holdout_p.add_argument("--device", default="cpu")
    holdout_p.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="must match the val selection's beam_width if passed; default: inherit from the selection",
    )
    holdout_p.add_argument("--selection", type=Path, default=None, help="default: <out-dir>/val_selection.json")
    holdout_p.set_defaults(func=run_holdout)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "select" and args.margins:
        args.margins = tuple(float(x) for x in args.margins.split(","))
    if args.command == "holdout" and args.selection is None:
        args.selection = args.out_dir / "val_selection.json"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
