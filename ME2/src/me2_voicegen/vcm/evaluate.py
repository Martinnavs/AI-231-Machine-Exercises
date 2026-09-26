"""Full evaluation CLI for the trained toy VCM CTC model: real checkpoint +
`vcm.pipeline`'s whole-clip mode over `out/conversions/v2/test_set/
manifest.csv`'s real `test` split.

Per decision (A) (docs/VCM-CONTRACT.md, ticket .scratch/vcm-toy/tickets/
05-pipeline-evaluate.md): `SPEC_GRAMMAR` and `TOY_GRAMMAR` are evaluated
and reported as two entirely separate sections, never merged. SPEC_GRAMMAR
is the verbatim BNF grammar -- 12 of the 20 `INTENT_PHRASES` intents'
canonical dataset phrases are not accepted by it. Only 2 of those (MESSAGE,
SET_REMINDER) have no SPEC_GRAMMAR rule at all; the other 10 (ALARM,
TIMER, CALL, TIME, WEATHER, LIST_REMINDERS, DIM_UP, DIM_DOWN, and the two
temperature-rule alternatives) DO have a `$CMD_*` BNF rule, but that rule
requires a slot or different phrasing than the dataset's bare canonical
phrase provides -- see `vcm.optiona.grammar`'s module docstring's "KNOWN GRAMMAR
LIMITATION" note and `tests/test_vcm_grammar.py::
test_message_and_set_reminder_have_no_spec_grammar_rule` (which asserts
the no-rule-at-all case for exactly those 2 intents, not all 12). So
SPEC_GRAMMAR is *expected* to report `no_match`/`REJECTED` for
essentially all of those 12 intents' clips by grammar construction --
that is a grammar-coverage fact, not a model failure, and both this
module's Markdown/JSON output say so explicitly (and correctly
distinguish the "no rule" vs. "rule needs a slot/different phrasing"
cases).

License (docs/VCM-CONTRACT.md section 8): `out/conversions/v2/
background_noise/` (feeding the `silence` bucket) is ESC-50,
CC-BY-NC-SA-4.0. Any report referencing the trained checkpoint's results
carries that provenance note in its header (this module reuses
`vcm.train.LICENSE_NOTE` verbatim rather than re-deriving separate
wording).

Threshold: `decode()`'s accept/reject cutoff is a mean per-frame
log-probability. This module never hardcodes an operating value -- it
sweeps a fixed candidate grid on the manifest's `val` split (never `test`)
per grammar, picks the value maximizing (target-accept-rate minus
false-accept-rate) on `babble`+`silence`, and reports the full sweep table
alongside the chosen point so the choice is auditable, not asserted.

`filipino_speech_corpus` rows resolve to a `None` transcript (excluded
from CTC training loss, per `vcm.text.resolve_transcript`) but are part of
the `babble` bucket in the manifest and are evaluated here like any other
`babble` row -- no special-casing needed, they are exactly the
rejection/false-accept probes decision (B) intended them to be.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.common.grammar_core import Grammar
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.optiona.grammar import SPEC_GRAMMAR, TOY_GRAMMAR
from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb.text import normalize_text as optionb_normalize_text
from me2_voicegen.vcm.optionb.transcript import prepare_ctc_transcript
from me2_voicegen.vcm.pipeline import infer_waveform, load_checkpoint
from me2_voicegen.vcm.train import LICENSE_NOTE

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "out" / "vcm" / "checkpoints" / "checkpoint.pt"
DEFAULT_SLOT_EVAL_MANIFEST = PROJECT_ROOT / "out" / "vcm" / "metadata" / "slot_eval" / "manifest.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "vcm"

NEG_INF_THRESHOLD = float("-inf")

# Candidate operating points for the val-split sweep, spanning the
# confidence range actually observed for the Task 04 checkpoint (real
# target-command accepts and real babble false-accepts both cluster
# between 0 and about -1.3 mean-log-prob-per-frame; the tail out to -5
# is included so the sweep table isn't silently truncated if a future
# retrained checkpoint's distribution shifts wider).
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
    0.0,
    -0.01,
    -0.02,
    -0.03,
    -0.05,
    -0.075,
    -0.1,
    -0.15,
    -0.2,
    -0.25,
    -0.3,
    -0.35,
    -0.4,
    -0.5,
    -0.75,
    -1.0,
    -1.5,
    -2.0,
    -5.0,
)

TARGET_BUCKET = "target_commands"
REJECT_PROBE_BUCKETS = ("babble", "silence")

# Filipino reference speakers in the Option B dataset, per
# docs/raw_requirements/optionb-dataset-readme.md ("Speakers" table, line
# 44): s68-s80, plus s89, s90, s100 (16 speakers total). Test-split target
# commands are all `s100` (180 clips); train carries the other 13 IDs
# (2,146 clips) and val carries s89/s90 (350 clips) -- so this checkpoint
# is not accent-naive, and the test-split Filipino group is a single
# held-out speaker, not a held-out accent.
FILIPINO_REFERENCE_SPEAKER_IDS: frozenset[str] = frozenset(
    {f"s{n}" for n in range(68, 81)} | {"s89", "s90", "s100"}
)

_OPTIONB_SPEAKER_ID_RE = re.compile(r"^s\d+$")

# Grammar-selection registry for `--grammar` (default "spec,toy" reproduces
# the original hardcoded SPEC_GRAMMAR/TOY_GRAMMAR pair bit-for-bit; "optionb"
# adds OPTIONB_GRAMMAR (D11, .scratch/optionb-dataset/tickets/
# 04-pipeline-wiring.md) so Option B rows can be decoded against their own
# grammar instead of only ever producing VCM intent labels that can never
# equal an Option B `label`.
GRAMMAR_REGISTRY: dict[str, tuple[Grammar, str]] = {
    "spec": (SPEC_GRAMMAR, "SPEC_GRAMMAR"),
    "toy": (TOY_GRAMMAR, "TOY_GRAMMAR"),
    "optionb": (OPTIONB_GRAMMAR, "OPTIONB_GRAMMAR"),
}


def _intent_labels_for(grammar_key: str) -> list[str]:
    """Per-intent table row labels (render_markdown). SPEC/TOY keep the
    full 20 dataset `INTENT_PHRASES` labels (including the ones SPEC_GRAMMAR
    has no rule for at all -- that's the point of its coverage note, not a
    gap to fix). OPTIONB_GRAMMAR's own label vocabulary is disjoint from
    INTENT_PHRASES (e.g. BRIGHTNESS/COLOR/CREATE_REMINDER/TEMPERATURE vs.
    DIM_UP/DIM_DOWN/SET_REMINDER/TEMP_UP/TEMP_DOWN), so it is derived from
    the grammar itself rather than reusing INTENT_PHRASES."""
    if grammar_key == "optionb":
        return sorted({intent for _, intent, _ in OPTIONB_GRAMMAR.all_phrases()})
    return sorted(INTENT_PHRASES)


@dataclasses.dataclass
class RowResult:
    index: int
    bucket: str
    label: str
    text: str
    intent: str | None
    confidence: float | None
    """`None` iff no grammar terminal was ever reached (equivalent to
    `-inf`, but JSON-serializable)."""
    group_id: str | None = None
    source_dataset: str | None = None
    slots: dict = dataclasses.field(default_factory=dict)
    """The decoder's extracted slot values (e.g. `{"AMPM": "am"}` for an
    ALARM clip) -- empty dict when `no_match` (per `vcm.decoder.decode`),
    never `None`, so callers can always safely do dict comparisons/lookups
    without a None-check."""
    rejection_reason: str | None = None
    """The decoder's `rejection_reason` (e.g. `"incomplete_prefix"` when the
    required-command-margin gate rejected the row) -- `None` whenever the
    gate was disabled or the row was not rejected by the gate, so the value
    is JSON-safe either way."""
    incomplete_prefix: str | None = None
    """The winning designated incomplete-prefix text from the beam search,
    or `None` if no designated prefix was present in the final beam --
    populated regardless of whether the required-command-margin gate was
    enabled for this decode."""
    incomplete_gap: float | None = None
    """`command_raw_score - incomplete_raw_score` (raw, unnormalized beam
    log mass), or `None` if either side was absent. Computed before the
    gate branch, so a decode at `required_command_margin=None` still
    yields the gap a live gated decode at any other margin would have used
    to decide accept/reject."""
    command_raw_score: float | None = None
    """The best completed-command terminal's raw beam log mass, or `None`
    if no terminal was reached."""
    incomplete_raw_score: float | None = None
    """The strongest designated incomplete-prefix beam's raw log mass, or
    `None` if none was present."""


@torch.no_grad()
def decode_split(
    model,
    feature_extractor: LogMelFeatureExtractor,
    dataset: VCMDataset,
    grammar: Grammar,
    beam_width: int,
    device: str | torch.device,
    required_command_margin: float | None = None,
) -> list[RowResult]:
    """Decode every row of `dataset` against `grammar` once, at the most
    permissive possible threshold (`-inf`) -- this captures each row's
    best-reachable intent/confidence independent of any operating
    threshold, so a threshold sweep afterwards is pure arithmetic over
    these cached results rather than re-running the model/decoder per
    candidate threshold. `required_command_margin`
    (docs/INCOMPLETE-GRAMMAR-REJECTION.md, Step 3) is a decode-time
    parameter threaded into every row independently (no cross-row state)."""
    results: list[RowResult] = []
    for i in range(len(dataset)):
        row = dataset.rows[i]
        example = dataset[i]
        decoded = infer_waveform(
            model,
            feature_extractor,
            example.waveform,
            grammar,
            threshold=NEG_INF_THRESHOLD,
            beam_width=beam_width,
            device=device,
            required_command_margin=required_command_margin,
        )
        confidence = None if decoded.intent is None else decoded.confidence
        results.append(
            RowResult(
                index=i,
                bucket=row["bucket"],
                label=row["label"],
                text=decoded.text,
                intent=decoded.intent,
                confidence=confidence,
                group_id=row.get("group_id"),
                source_dataset=row.get("source_dataset"),
                slots=decoded.slots,
                rejection_reason=decoded.rejection_reason,
                incomplete_prefix=decoded.incomplete_prefix,
                incomplete_gap=decoded.incomplete_gap,
                command_raw_score=decoded.command_raw_score,
                incomplete_raw_score=decoded.incomplete_raw_score,
            )
        )
    return results


def _incomplete_prefix_rejection_counts(
    val_results: list[RowResult], test_results: list[RowResult]
) -> dict:
    """Count rows rejected by the incomplete-prefix gate, per split
    (docs/INCOMPLETE-GRAMMAR-REJECTION.md, Step 3). Pure over cached
    `RowResult`s so it is unit-testable without a model; yields zero counts
    whenever the gate was disabled (`rejection_reason` stays `None`)."""
    return {
        "val": sum(1 for r in val_results if r.rejection_reason == "incomplete_prefix"),
        "test": sum(1 for r in test_results if r.rejection_reason == "incomplete_prefix"),
    }


def _accepted(r: RowResult, threshold: float) -> bool:
    return r.intent is not None and r.confidence is not None and r.confidence >= threshold


def sweep_thresholds(
    results: list[RowResult], thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID
) -> list[dict]:
    """Sweep candidate thresholds against already-decoded `val`-split
    results. `target_accept_rate` is the fraction of `target_commands`
    rows accepted (regardless of whether the accepted intent is correct --
    this sweep is about the accept/reject boundary, not per-intent
    accuracy, which `confusion_counts` reports separately on `test`).
    `false_accept_rate` is the fraction of `babble`+`silence` rows
    accepted (a false trigger)."""
    target_idxs = [i for i, r in enumerate(results) if r.bucket == TARGET_BUCKET]
    reject_idxs = [i for i, r in enumerate(results) if r.bucket in REJECT_PROBE_BUCKETS]

    sweep: list[dict] = []
    for t in thresholds:
        target_accepts = sum(1 for i in target_idxs if _accepted(results[i], t))
        reject_accepts = sum(1 for i in reject_idxs if _accepted(results[i], t))
        target_rate = target_accepts / len(target_idxs) if target_idxs else 0.0
        false_rate = reject_accepts / len(reject_idxs) if reject_idxs else 0.0
        sweep.append(
            {
                "threshold": t,
                "n_target_commands": len(target_idxs),
                "target_accept_rate": target_rate,
                "n_reject_probes": len(reject_idxs),
                "false_accept_rate": false_rate,
                "youden_j": target_rate - false_rate,
            }
        )
    return sweep


def sweep_margins(
    results: list[RowResult], margins: tuple[float, ...], threshold: float
) -> list[dict]:
    """Sweep candidate `required_command_margin` values against
    already-decoded results. Precondition: `results` must come from a
    decode run with `required_command_margin=None` (the gate disabled), so
    `incomplete_gap` reflects the true baseline gap for every candidate
    margin -- pure arithmetic, no re-decoding. Violating this precondition
    raises `ValueError` (see below). Mirrors `sweep_thresholds`'s shape:
    `target_accept_rate` over `TARGET_BUCKET` rows and `false_accept_rate`
    over `REJECT_PROBE_BUCKETS` rows, both gated by `_accepted(row,
    threshold)` AND the margin gate (`incomplete_gap is None or
    incomplete_gap >= margin`). `incomplete_prefix_reject_rate` additionally
    isolates, among `TARGET_BUCKET` rows that would be accepted on the
    confidence gate alone, the fraction the margin gate specifically
    rejects.

    Raises:
        ValueError: if any row has `rejection_reason is not None` --
            `rejection_reason` is only ever set when the gate was actually
            enabled at decode time (decoder.py never sets it on the
            `required_command_margin=None` path), so its presence on any
            row proves this batch was not decoded with the gate disabled.
    """
    if any(r.rejection_reason is not None for r in results):
        raise ValueError(
            "sweep_margins requires results decoded with required_command_margin=None "
            "(the incomplete-prefix gate disabled) -- found a row with rejection_reason set, "
            "meaning the gate was already active at decode time."
        )

    target_idxs = [i for i, r in enumerate(results) if r.bucket == TARGET_BUCKET]
    reject_idxs = [i for i, r in enumerate(results) if r.bucket in REJECT_PROBE_BUCKETS]

    def _margin_ok(r: RowResult, margin: float) -> bool:
        return r.incomplete_gap is None or r.incomplete_gap >= margin

    sweep: list[dict] = []
    for margin in margins:
        target_accepts = sum(
            1 for i in target_idxs if _accepted(results[i], threshold) and _margin_ok(results[i], margin)
        )
        reject_accepts = sum(
            1 for i in reject_idxs if _accepted(results[i], threshold) and _margin_ok(results[i], margin)
        )
        target_rate = target_accepts / len(target_idxs) if target_idxs else 0.0
        false_rate = reject_accepts / len(reject_idxs) if reject_idxs else 0.0

        confidence_accepted_idxs = [i for i in target_idxs if _accepted(results[i], threshold)]
        margin_rejected = sum(
            1
            for i in confidence_accepted_idxs
            if results[i].incomplete_gap is not None and results[i].incomplete_gap < margin
        )
        incomplete_prefix_reject_rate = (
            margin_rejected / len(confidence_accepted_idxs) if confidence_accepted_idxs else 0.0
        )

        sweep.append(
            {
                "margin": margin,
                "target_accept_rate": target_rate,
                "false_accept_rate": false_rate,
                "incomplete_prefix_reject_rate": incomplete_prefix_reject_rate,
            }
        )
    return sweep


def choose_operating_threshold(sweep: list[dict]) -> dict:
    """Documented selection rule: the candidate maximizing Youden's J
    (target_accept_rate - false_accept_rate) on the val-split sweep. Ties
    broken toward the least permissive (highest/least-negative) threshold
    among the tied candidates, since a stricter cutoff is the safer
    default at equal J."""
    best_j = max(row["youden_j"] for row in sweep)
    tied = [row for row in sweep if row["youden_j"] == best_j]
    return max(tied, key=lambda row: row["threshold"])


def confusion_counts(results: list[RowResult], threshold: float) -> dict[str, dict[str, int]]:
    """`target_commands` rows only: true INTENT_PHRASES label ->
    {predicted-intent-or-"REJECTED": count}, at the given threshold."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r.bucket != TARGET_BUCKET:
            continue
        predicted = r.intent if _accepted(r, threshold) else "REJECTED"
        counts[r.label][predicted] += 1
    return {label: dict(preds) for label, preds in counts.items()}


def false_accept_stats(results: list[RowResult], threshold: float, bucket: str) -> dict:
    rows = [r for r in results if r.bucket == bucket]
    if not rows:
        return {"n": 0, "false_accepts": 0, "rate": None}
    false_accepts = sum(1 for r in rows if _accepted(r, threshold))
    return {"n": len(rows), "false_accepts": false_accepts, "rate": false_accepts / len(rows)}


def classify_speaker_group(source_dataset: str | None, group_id: str | None) -> str | None:
    """`filipino_reference` / `foreign_reference` / `None` (unclassified or
    not applicable). Only `optionb` and `fil50_persona` rows are classified
    -- the other four `source_dataset` values (`background_noise`,
    `youtube_institutional`, `filipino_speech_corpus`,
    `common_voice_negative`) use unrelated `group_id` shapes (ESC-50
    filenames, video IDs, zero-padded numerics, empty) that must never be
    misread as a foreign `s<N>` speaker ID.

    `fil50_persona` (feature `accent-balance-fil50`,
    .scratch/accent-balance-fil50/tickets/00-RECAP.md T6): every row is a
    persona-synthesized Filipino voice by construction (`collate.py` only
    ever emits it for rows drawn from the Filipino voice pool), so it's
    unconditionally `filipino_reference` -- it never needs `group_id`
    disambiguation the way `optionb`'s pooled `s<N>` IDs do."""
    if source_dataset == "fil50_persona":
        return "filipino_reference"
    if source_dataset != "optionb":
        return None
    if not group_id:
        return None
    if group_id in FILIPINO_REFERENCE_SPEAKER_IDS:
        return "filipino_reference"
    if _OPTIONB_SPEAKER_ID_RE.match(group_id):
        return "foreign_reference"
    return None


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def speaker_group_breakdown(results: list[RowResult], threshold: float) -> dict | None:
    """Filipino-reference vs. foreign-reference exact-accuracy comparison on
    `TARGET_BUCKET` rows only, mirroring `false_accept_stats`/
    `confusion_counts`'s shape. `None` when no classifiable Option B target
    row exists at all (e.g. a `spec`/`toy` run against the vcm-toy
    manifest, which has no `optionb` rows)."""
    target_rows = [r for r in results if r.bucket == TARGET_BUCKET]
    groups: dict[str, list[RowResult]] = {"filipino_reference": [], "foreign_reference": []}
    n_unclassified = 0
    for r in target_rows:
        group = classify_speaker_group(r.source_dataset, r.group_id)
        if group is None:
            n_unclassified += 1
            continue
        groups[group].append(r)

    if not groups["filipino_reference"] and not groups["foreign_reference"]:
        return None

    per_group: dict[str, dict] = {}
    for group_name, rows in groups.items():
        n = len(rows)
        n_accepted = sum(1 for r in rows if _accepted(r, threshold))
        n_exact_correct = sum(1 for r in rows if _accepted(r, threshold) and r.intent == r.label)
        per_group[group_name] = {
            "n": n,
            "n_accepted": n_accepted,
            "n_exact_correct": n_exact_correct,
            "accept_rate": n_accepted / n if n else None,
            "exact_accuracy": n_exact_correct / n if n else None,
            "exact_accuracy_ci95": _wilson_interval(n_exact_correct, n),
        }

    filipino_acc = per_group["filipino_reference"]["exact_accuracy"]
    foreign_acc = per_group["foreign_reference"]["exact_accuracy"]
    gap = foreign_acc - filipino_acc if filipino_acc is not None and foreign_acc is not None else None

    return {
        "filipino_reference": per_group["filipino_reference"],
        "foreign_reference": per_group["foreign_reference"],
        "n_unclassified": n_unclassified,
        "exact_accuracy_gap_foreign_minus_filipino": gap,
    }


def _true_slots_for_row(raw_row: dict, grammar: Grammar) -> dict | None:
    """Ground-truth slot values for one Option B manifest row, derived by
    parsing its OWN resolved transcript through `grammar` -- the same
    digit-spelling (`prepare_ctc_transcript`) and normalization
    (`optionb.text.normalize_text`) applied before training (see
    `vcm.text.resolve_transcript`'s `optionb` branch) -- rather than a
    hand-maintained lookup table that could drift from what the model was
    actually trained to predict. `None` if the resolved transcript isn't
    accepted by `grammar` at all, or accepted only under a different
    intent than `raw_row["label"]` -- both should be impossible for a
    genuine Option B canonical phrase; reported via
    `n_unparseable_ground_truth` rather than raising, since a QA-flagged
    manifest row surviving into `target_commands` is a data problem, not a
    reason to crash the eval run."""
    resolved = prepare_ctc_transcript(raw_row.get("transcript", ""))
    normalized = optionb_normalize_text(resolved)
    matches = grammar.accepts(normalized)
    if not matches:
        return None
    for intent, slots in matches:
        if intent == raw_row.get("label"):
            return slots
    return None


def slot_accuracy_breakdown(
    results: list[RowResult], raw_rows: list[dict], grammar: Grammar, threshold: float
) -> dict | None:
    """Among `target_commands` rows whose true intent carries at least one
    grammar slot (derived from `grammar.all_phrases()`, not hardcoded --
    stays correct if the grammar changes), and whose predicted intent is
    already correct: what fraction ALSO got every slot value right (e.g.
    predicted `AMPM=am` when the clip actually said "9 PM")? Answers "does
    the model get intent right but the slot value wrong" -- a question the
    existing `exact_accuracy`/`per_intent_confusion` metrics cannot answer,
    since they only compare `intent == label`, never `decoded.slots`.

    Restricted to `source_dataset == "optionb"` rows: Option A's
    (`vcm.optiona`) canonical `INTENT_PHRASES` dataset clips carry no slot
    values at all (bare phrases like "set alarm"), so this metric is not
    meaningful for `spec`/`toy` runs. Returns `None` when no classifiable,
    slot-bearing Option B target row exists (e.g. `--grammar spec,toy`, or
    an `--grammar optionb` run whose current grammar happens to have no
    slotted intents)."""
    slotted_intents = {intent for _, intent, slots in grammar.all_phrases() if slots}
    if not slotted_intents:
        return None

    n_checked = 0
    n_unparseable = 0
    n_intent_correct = 0
    n_intent_and_slots_correct = 0
    per_slot_name: dict[str, dict[str, int]] = {}

    for r in results:
        if r.bucket != TARGET_BUCKET or r.source_dataset != "optionb" or r.label not in slotted_intents:
            continue
        n_checked += 1
        true_slots = _true_slots_for_row(raw_rows[r.index], grammar)
        if true_slots is None:
            n_unparseable += 1
            continue
        if not (_accepted(r, threshold) and r.intent == r.label):
            continue
        n_intent_correct += 1
        if r.slots == true_slots:
            n_intent_and_slots_correct += 1
        for slot_name, true_value in true_slots.items():
            stat = per_slot_name.setdefault(slot_name, {"n": 0, "n_correct": 0})
            stat["n"] += 1
            if r.slots.get(slot_name) == true_value:
                stat["n_correct"] += 1

    if n_checked == 0:
        return None

    return {
        "n_slot_bearing_target_rows": n_checked,
        "n_unparseable_ground_truth": n_unparseable,
        "n_intent_correct": n_intent_correct,
        "n_intent_and_slots_correct": n_intent_and_slots_correct,
        "slot_exact_match_rate_given_intent_correct": (
            n_intent_and_slots_correct / n_intent_correct if n_intent_correct else None
        ),
        "per_slot_name_accuracy": {
            name: {
                "n": stat["n"],
                "n_correct": stat["n_correct"],
                "accuracy": stat["n_correct"] / stat["n"] if stat["n"] else None,
            }
            for name, stat in sorted(per_slot_name.items())
        },
    }


def evaluate_grammar(
    model,
    feature_extractor: LogMelFeatureExtractor,
    val_dataset: VCMDataset,
    test_dataset: VCMDataset,
    grammar: Grammar,
    grammar_label: str,
    beam_width: int,
    device: str | torch.device,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    intent_labels: list[str] | None = None,
    required_command_margin: float | None = None,
) -> dict:
    val_results = decode_split(
        model, feature_extractor, val_dataset, grammar, beam_width, device,
        required_command_margin=required_command_margin,
    )
    sweep = sweep_thresholds(val_results, threshold_grid)
    chosen = choose_operating_threshold(sweep)
    threshold = chosen["threshold"]

    test_results = decode_split(
        model, feature_extractor, test_dataset, grammar, beam_width, device,
        required_command_margin=required_command_margin,
    )
    confusion = confusion_counts(test_results, threshold)

    target_rows = [r for r in test_results if r.bucket == TARGET_BUCKET]
    n_target = len(target_rows)
    n_accepted = sum(1 for r in target_rows if _accepted(r, threshold))
    n_exact_correct = sum(
        1 for r in target_rows if _accepted(r, threshold) and r.intent == r.label
    )

    return {
        "grammar": grammar_label,
        "intent_labels": intent_labels if intent_labels is not None else sorted(INTENT_PHRASES),
        "threshold_sweep_on_val": sweep,
        "chosen_operating_threshold": threshold,
        "chosen_operating_point_val_stats": chosen,
        "required_command_margin": required_command_margin,
        "incomplete_prefix_rejections": _incomplete_prefix_rejection_counts(
            val_results, test_results
        ),
        "test_split": {
            "n_target_commands": n_target,
            "n_accepted": n_accepted,
            "n_exact_correct": n_exact_correct,
            "accept_rate": n_accepted / n_target if n_target else None,
            "exact_accuracy": n_exact_correct / n_target if n_target else None,
            "per_intent_confusion": confusion,
            "false_accept_rate_babble": false_accept_stats(test_results, threshold, "babble"),
            "false_accept_rate_silence": false_accept_stats(test_results, threshold, "silence"),
            "speaker_group_breakdown": speaker_group_breakdown(test_results, threshold),
            "slot_accuracy": slot_accuracy_breakdown(test_results, test_dataset.rows, grammar, threshold),
        },
    }


# ---------------------------------------------------------------------------
# Optional slot-eval-set section (Task 07's manifest, may not exist -- must
# not hard-fail if it doesn't).
# ---------------------------------------------------------------------------


def _load_slot_eval_rows(manifest_path: Path) -> list[dict]:
    import csv

    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def evaluate_slot_eval_set(
    model,
    feature_extractor: LogMelFeatureExtractor,
    manifest_path: Path,
    grammar: Grammar,
    grammar_label: str,
    threshold: float,
    beam_width: int,
    device: str | torch.device,
    required_command_margin: float | None = None,
) -> dict:
    """Runs Task 07's slot-eval-set clips (if present) through the same
    pipeline, at the already-chosen (`test`-split-reported) operating
    threshold for `grammar_label`. Framed per this ticket's Action section:
    quantifies the KNOWN training gap (the acoustic model saw zero
    slot-word audio during training) -- NOT proof that slot extraction
    works on real audio; `vcm.optiona.grammar`'s own synthetic-posterior tests are
    the correctness proof for the grammar/decoder side of slot
    extraction."""
    import torchaudio

    audio_root = manifest_path.parent
    rows = _load_slot_eval_rows(manifest_path)

    per_row: list[dict] = []
    for row in rows:
        wav_path = audio_root / row["path"]
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != int(row.get("sample_rate", sample_rate)):
            waveform = torchaudio.functional.resample(waveform, sample_rate, int(row["sample_rate"]))

        decoded = infer_waveform(
            model, feature_extractor, waveform, grammar, threshold=threshold, beam_width=beam_width,
            device=device, required_command_margin=required_command_margin,
        )
        per_row.append(
            {
                "filename": row.get("filename"),
                "true_intent": row.get("label"),
                "predicted_intent": decoded.intent,
                "predicted_slots": decoded.slots,
                "text": decoded.text,
                "confidence": None if decoded.no_match else decoded.confidence,
                "accepted": not decoded.no_match,
                "intent_correct": (not decoded.no_match) and decoded.intent == row.get("label"),
            }
        )

    n = len(per_row)
    n_accepted = sum(1 for r in per_row if r["accepted"])
    n_intent_correct = sum(1 for r in per_row if r["intent_correct"])
    return {
        "grammar": grammar_label,
        "threshold_used": threshold,
        "framing": (
            "Quantifies the KNOWN slot-vocabulary training gap: vcm.train's "
            "acoustic model was trained exclusively on the 20 fixed "
            "INTENT_PHRASES canonical command phrases and saw zero "
            "slot-word audio (numbers, artist names, persona names, time "
            "units, am/pm) at training time. Low accept/intent-correct "
            "rates here are an EXPECTED consequence of that training-data "
            "gap, not a grammar/decoder defect -- the grammar/decoder's "
            "own correctness for slot extraction is proven separately by "
            "vcm.optiona.grammar's synthetic-posterior unit tests, not by this "
            "real-audio probe."
        ),
        "n_clips": n,
        "n_accepted": n_accepted,
        "n_intent_correct": n_intent_correct,
        "accept_rate": n_accepted / n if n else None,
        "intent_correct_rate": n_intent_correct / n if n else None,
        "rows": per_row,
    }


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# VCM toy CTC model -- evaluation report")
    lines.append("")
    lines.append(f"License: {report['license_note']}")
    lines.append("")
    lines.append(
        f"Checkpoint: `{report['checkpoint_path']}` "
        f"(preset={report['checkpoint_meta'].get('preset')}, "
        f"epoch={report['checkpoint_meta'].get('epoch')}, "
        f"val_loss={report['checkpoint_meta'].get('val_loss')})"
    )
    lines.append(f"Manifest: `{report['manifest_path']}`")
    lines.append(f"Device: `{report['device']}`  Beam width: {report['beam_width']}")
    lines.append("")
    lines.append(
        "Threshold operating points below are chosen SEPARATELY per grammar "
        "by sweeping the candidate grid on the manifest's `val` split "
        "(never `test`) and maximizing Youden's J "
        "(target_accept_rate - false_accept_rate on babble+silence); the "
        "full sweep table is included so the choice is auditable."
    )
    lines.append("")

    for section in report["grammar_sections"]:
        lines.append(f"## {section['grammar']} results")
        lines.append("")
        if section["grammar"] == "SPEC_GRAMMAR":
            lines.append(
                "> **Grammar-coverage note:** `SPEC_GRAMMAR` is the verbatim BNF "
                "grammar. 12 of the 20 `INTENT_PHRASES` intents' canonical "
                "dataset phrases are not accepted by it -- but only 2 of "
                "those (MESSAGE, SET_REMINDER) have no SPEC_GRAMMAR rule at "
                "all; the other 10 (ALARM, TIMER, CALL, TIME, WEATHER, "
                "LIST_REMINDERS, DIM_UP, DIM_DOWN, and the temperature "
                "rule's two intents) DO have a `$CMD_*` BNF rule, it just "
                "requires a slot or different phrasing than the dataset's "
                "bare canonical phrase provides. Those 12 intents are "
                "therefore EXPECTED to report `REJECTED` for essentially "
                "all their clips in this section, by grammar construction, "
                "not because the model failed to recognize them (see the "
                "TOY_GRAMMAR section below for the same clips with the "
                "toy-only bare-phrase aliases available)."
            )
            lines.append("")

        lines.append(
            f"Chosen operating threshold (val sweep): "
            f"**{section['chosen_operating_threshold']}** "
            f"(val target_accept_rate="
            f"{section['chosen_operating_point_val_stats']['target_accept_rate']:.3f}, "
            f"val false_accept_rate="
            f"{section['chosen_operating_point_val_stats']['false_accept_rate']:.3f})"
        )
        lines.append("")
        lines.append("### Val-split threshold sweep")
        lines.append("")
        lines.append("| threshold | val target_accept_rate | val false_accept_rate | Youden J |")
        lines.append("|---|---|---|---|")
        for row in section["threshold_sweep_on_val"]:
            lines.append(
                f"| {row['threshold']} | {row['target_accept_rate']:.3f} | "
                f"{row['false_accept_rate']:.3f} | {row['youden_j']:.3f} |"
            )
        lines.append("")

        ts = section["test_split"]
        lines.append("### Test-split results at the chosen threshold")
        lines.append("")
        lines.append(
            f"- `target_commands`: {ts['n_target_commands']} clips, "
            f"{ts['n_accepted']} accepted "
            f"({ts['accept_rate']:.3f} accept rate), "
            f"{ts['n_exact_correct']} exact-intent-correct "
            f"({ts['exact_accuracy']:.3f} exact accuracy)"
        )
        fab = ts["false_accept_rate_babble"]
        fas = ts["false_accept_rate_silence"]
        fab_rate_str = f"({fab['rate']:.3f})" if fab["rate"] is not None else "(n/a)"
        fas_rate_str = f"({fas['rate']:.3f})" if fas["rate"] is not None else "(n/a)"
        lines.append(
            f"- `babble` false-accept rate: {fab['false_accepts']}/{fab['n']} "
            f"{fab_rate_str} -- includes filipino_speech_corpus rows per decision (B)"
        )
        lines.append(f"- `silence` false-accept rate: {fas['false_accepts']}/{fas['n']} {fas_rate_str}")
        # `.get`-based: old reports / partial section dicts without the key
        # render exactly as before (no line).
        rejections = section.get("incomplete_prefix_rejections")
        if rejections is not None and section.get("required_command_margin") is not None:
            lines.append(
                f"- incomplete-prefix gate rejections: val={rejections['val']}, "
                f"test={rejections['test']}"
            )
        lines.append(
            "- These false-accept rates hold **at the chosen operating "
            "threshold only** -- see the val-split sweep table above for "
            "how false-accept rate degrades at looser (more permissive) "
            "thresholds; it is not an unconditional property of the model."
        )
        lines.append("")

        breakdown = ts.get("speaker_group_breakdown")
        if breakdown is not None:
            lines.append("### Speaker-group breakdown")
            lines.append("")
            lines.append(
                "> **Caveat:** the test-split Filipino group is a single "
                "held-out speaker (`s100`, 180 clips), so a gap here "
                "confounds accent with speaker identity. 13 of the 16 "
                "Filipino speakers (2,146 clips) are in the train split, "
                "so this checkpoint is not accent-naive."
            )
            lines.append("")
            lines.append("| group | n | accept_rate | exact_accuracy | 95% CI |")
            lines.append("|---|---|---|---|---|")
            for group_name in ("filipino_reference", "foreign_reference"):
                g = breakdown[group_name]
                accept_str = f"{g['accept_rate']:.3f}" if g["accept_rate"] is not None else "n/a"
                acc_str = f"{g['exact_accuracy']:.3f}" if g["exact_accuracy"] is not None else "n/a"
                ci = g["exact_accuracy_ci95"]
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else "n/a"
                lines.append(f"| {group_name} | {g['n']} | {accept_str} | {acc_str} | {ci_str} |")
            gap = breakdown["exact_accuracy_gap_foreign_minus_filipino"]
            gap_str = f"{gap:.3f}" if gap is not None else "n/a"
            lines.append("")
            lines.append(
                f"- exact_accuracy_gap_foreign_minus_filipino: {gap_str}; "
                f"n_unclassified: {breakdown['n_unclassified']}"
            )
            lines.append("")

        slot_acc = ts.get("slot_accuracy")
        if slot_acc is not None:
            lines.append("### Slot accuracy")
            lines.append("")
            lines.append(
                "> Intent-level accuracy above only checks `intent == label` -- "
                "it does not check whether a slot value (e.g. which hour an "
                "ALARM clip named) was extracted correctly. This section "
                "does: among slot-bearing Option B target clips whose "
                "predicted intent was already correct, what fraction also "
                "got every slot value right."
            )
            lines.append("")
            rate = slot_acc["slot_exact_match_rate_given_intent_correct"]
            rate_str = f"{rate:.3f}" if rate is not None else "n/a"
            lines.append(
                f"- {slot_acc['n_intent_and_slots_correct']}/"
                f"{slot_acc['n_intent_correct']} intent-correct clips also "
                f"had every slot value correct ({rate_str}), out of "
                f"{slot_acc['n_slot_bearing_target_rows']} slot-bearing "
                f"target clips ({slot_acc['n_unparseable_ground_truth']} "
                "with unparseable ground truth)"
            )
            lines.append("")
            lines.append("| slot | n (intent-correct clips) | n correct | accuracy |")
            lines.append("|---|---|---|---|")
            for slot_name, stat in slot_acc["per_slot_name_accuracy"].items():
                acc_str = f"{stat['accuracy']:.3f}" if stat["accuracy"] is not None else "n/a"
                lines.append(f"| {slot_name} | {stat['n']} | {stat['n_correct']} | {acc_str} |")
            lines.append("")

        lines.append("### Per-intent accept/confusion counts (test split)")
        lines.append("")
        lines.append("| true intent | predicted distribution |")
        lines.append("|---|---|")
        for label in section.get("intent_labels", sorted(INTENT_PHRASES)):
            preds = ts["per_intent_confusion"].get(label, {})
            if not preds:
                lines.append(f"| {label} | (no test-split clips) |")
                continue
            pred_str = ", ".join(f"{k}={v}" for k, v in sorted(preds.items()))
            lines.append(f"| {label} | {pred_str} |")
        lines.append("")

    if report.get("slot_eval_sections"):
        lines.append("## Slot-eval-set (Task 07) results")
        lines.append("")
        lines.append(
            "> Framing: quantifies the KNOWN slot-vocabulary training gap "
            "(the acoustic model saw zero slot-word audio during "
            "training) -- not proof that slot extraction works on real "
            "audio. See each section's own `framing` text below."
        )
        lines.append("")
        for section in report["slot_eval_sections"]:
            lines.append(f"### {section['grammar']}")
            lines.append("")
            lines.append(section["framing"])
            lines.append("")
            lines.append(
                f"- {section['n_clips']} clips, {section['n_accepted']} accepted "
                f"({section['accept_rate']}), {section['n_intent_correct']} "
                f"intent-correct ({section['intent_correct_rate']})"
            )
            lines.append("")
    elif report.get("slot_eval_skipped_reason"):
        lines.append("## Slot-eval-set (Task 07) results")
        lines.append("")
        lines.append(f"Skipped: {report['slot_eval_skipped_reason']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_report(
    checkpoint_path: Path,
    checkpoint_meta: dict,
    manifest_path: Path,
    device: str,
    beam_width: int,
    grammar_sections: list[dict],
    slot_eval_sections: list[dict] | None,
    slot_eval_skipped_reason: str | None,
) -> dict:
    return {
        "license_note": LICENSE_NOTE,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_meta": {
            k: v
            for k, v in checkpoint_meta.items()
            if k != "model_state_dict"
        },
        "manifest_path": str(manifest_path),
        "device": device,
        "beam_width": beam_width,
        "grammar_sections": grammar_sections,
        "slot_eval_sections": slot_eval_sections,
        "slot_eval_skipped_reason": slot_eval_skipped_reason,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument(
        "--required-command-margin",
        type=float,
        default=None,
        help=(
            "incomplete-prefix rejection gate margin (raw unnormalized beam "
            "log-mass units; docs/INCOMPLETE-GRAMMAR-REJECTION.md). Default "
            "None = gate disabled"
        ),
    )
    parser.add_argument(
        "--slot-eval-manifest",
        type=Path,
        default=DEFAULT_SLOT_EVAL_MANIFEST,
        help="Task 07's optional slot-eval-set manifest; skipped gracefully if missing",
    )
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default=None,
        help="comma-separated float overrides for the val-split sweep grid",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default="spec,toy",
        help=(
            "comma-separated grammar keys to evaluate against, chosen from "
            f"{sorted(GRAMMAR_REGISTRY)}. Default 'spec,toy' reproduces the "
            "original SPEC_GRAMMAR/TOY_GRAMMAR pair unchanged. 'optionb' "
            "selects OPTIONB_GRAMMAR and is only meaningful against a "
            "manifest that (like out/conversions/v2/optionb/manifest.csv) "
            "carries babble/silence reject-probe rows alongside its "
            "target_commands rows -- otherwise REJECT_PROBE_BUCKETS is "
            "empty and the val-split threshold sweep degenerates (see "
            "docs/VCM-CONTRACT.md)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    threshold_grid = DEFAULT_THRESHOLD_GRID
    if args.threshold_grid:
        threshold_grid = tuple(float(x) for x in args.threshold_grid.split(","))

    grammar_keys = [g.strip().lower() for g in args.grammar.split(",") if g.strip()]
    unknown_keys = [g for g in grammar_keys if g not in GRAMMAR_REGISTRY]
    if unknown_keys:
        raise SystemExit(
            f"unknown --grammar value(s) {unknown_keys}; choices are {sorted(GRAMMAR_REGISTRY)}"
        )
    if not grammar_keys:
        raise SystemExit("--grammar must name at least one grammar")
    selected_grammars = [
        (GRAMMAR_REGISTRY[key][0], GRAMMAR_REGISTRY[key][1], _intent_labels_for(key))
        for key in grammar_keys
    ]

    model, checkpoint_meta = load_checkpoint(args.checkpoint, device=args.device)
    feature_extractor = LogMelFeatureExtractor()

    val_dataset = VCMDataset(args.manifest, split="val", augmenter=None)
    test_dataset = VCMDataset(args.manifest, split="test", augmenter=None)

    grammar_sections = []
    for grammar, label, intent_labels in selected_grammars:
        print(f"evaluating {label} ...")
        section = evaluate_grammar(
            model,
            feature_extractor,
            val_dataset,
            test_dataset,
            grammar,
            label,
            args.beam_width,
            args.device,
            threshold_grid,
            intent_labels,
            args.required_command_margin,
        )
        grammar_sections.append(section)
        ts = section["test_split"]
        print(
            f"  {label}: threshold={section['chosen_operating_threshold']} "
            f"accept_rate={ts['accept_rate']:.3f} exact_accuracy={ts['exact_accuracy']:.3f} "
            f"babble_far={ts['false_accept_rate_babble']['rate']:.3f} "
            f"silence_far={ts['false_accept_rate_silence']['rate']}"
        )

    slot_eval_sections: list[dict] | None = None
    slot_eval_skipped_reason: str | None = None
    if not args.slot_eval_manifest.exists():
        slot_eval_skipped_reason = f"{args.slot_eval_manifest} not found (Task 07 clips may not exist yet)"
        print(f"slot-eval-set skipped: {slot_eval_skipped_reason}")
    else:
        try:
            slot_eval_sections = []
            for (grammar, label, _intent_labels), section in zip(selected_grammars, grammar_sections):
                slot_eval_sections.append(
                    evaluate_slot_eval_set(
                        model,
                        feature_extractor,
                        args.slot_eval_manifest,
                        grammar,
                        label,
                        section["chosen_operating_threshold"],
                        args.beam_width,
                        args.device,
                        args.required_command_margin,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - optional section, must not hard-fail
            slot_eval_sections = None
            slot_eval_skipped_reason = f"slot-eval-set present but evaluation failed: {exc!r}"
            print(f"slot-eval-set evaluation failed (non-fatal): {exc!r}")

    report = build_report(
        args.checkpoint,
        checkpoint_meta,
        args.manifest,
        args.device,
        args.beam_width,
        grammar_sections,
        slot_eval_sections,
        slot_eval_skipped_reason,
    )

    json_path = metadata_dir / "eval_report.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    md_path = metadata_dir / "eval_report.md"
    md_path.write_text(render_markdown(report))

    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
