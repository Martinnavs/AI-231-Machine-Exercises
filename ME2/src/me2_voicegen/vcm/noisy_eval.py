"""Fixed-seed noisy/reverb eval pass for the VCM model (eval gate).

The training `Augmenter` is hard-gated to `split == "train"` in
`vcm.dataset` (correctly -- eval must not re-randomize), and the fil50
manifest carries no noisy/reverberant variant rows in val/test. So the
normal eval report is 100% clean-signal. This module adds the SEPARATE
pass: every row of the val and test splits gets one deterministic
perturbation -- one RIR from a fixed in-memory pool, then one additive
noise clip at a fixed SNR -- and the perturbed decodes are scored at the
CLEAN-val-chosen operating threshold (never re-swept on noisy val; a gate
measures whether the operating point transfers to real conditions, and
re-fitting on the eval set would break checkpoint-to-checkpoint
comparability).

Determinism contract (the point of the feature): given a manifest and a
seed, the perturbation applied to each row is pure data
(`(rir_index, noise_index, snr_db)`, drawn in manifest order from
`torch.Generator().manual_seed(zlib.crc32(f"{seed}:{tag}"))`), so the same
(manifest, seed) pair yields a bit-identical perturbed signal on every
run, against any checkpoint. Scores are therefore comparable
checkpoint-to-checkpoint by construction.

Only `common.augment.apply_rir` / `apply_noise` are reused here. SpecAugment
is deliberately NOT part of this pass (a training-only feature-space
regularizer, not a real acoustic condition), and neither is time-stretch
(a prosody/duration change, not an acoustic-channel condition). Severity
ranges match what training intended to cover: RT60 U[0.1, 0.5] s rooms
(same `build_rir_pool` generation the train augmenter uses) and SNR
U[5, 25] dB (the train augmenter's `SNR_MIN_DB`/`SNR_MAX_DB`).
"""

from __future__ import annotations

import zlib

import torch

from me2_voicegen.common.augment import (
    RT60_MAX,
    RT60_MIN,
    SNR_MAX_DB,
    SNR_MIN_DB,
    apply_noise,
    apply_rir,
    build_rir_pool,
)
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.common.grammar_core import Grammar
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.evaluate import (
    NEG_INF_THRESHOLD,
    RowResult,
    _accepted,
    confusion_counts,
    false_accept_stats,
    slot_accuracy_breakdown,
    speaker_group_breakdown,
)
from me2_voicegen.vcm.pipeline import infer_waveform

NOISY_EVAL_RIR_POOL_SIZE = 200
NOISY_EVAL_SAMPLE_RATE = 16000

PERTURBATION_DOC = (
    "rir+additive_noise: every row gets one RIR (RT60 U[0.1,0.5]s randomized "
    "room from a fixed 200-entry pool) then one ESC-50 noise clip at "
    "SNR U[5,25] dB, both always applied, in that order"
)
THRESHOLD_RULE = (
    "scored at the clean-val chosen operating threshold per grammar "
    "(choose_operating_threshold on the CLEAN val split); never re-swept on "
    "noisy val"
)


def _seed_for(seed: int, tag: str) -> int:
    """Deterministic 32-bit unsigned sub-seed for one named RNG stream
    (e.g. the per-split draw sequence or the RIR pool build). `zlib.crc32`
    is a fixed polynomial hash -- stable across processes, platforms, and
    Python runs (unlike `hash()`), which is what the cross-run
    determinism contract requires."""
    return zlib.crc32(f"{seed}:{tag}".encode("utf-8"))


def build_rir_pool_for_seed(
    seed: int,
    pool_size: int = NOISY_EVAL_RIR_POOL_SIZE,
    sample_rate: int = NOISY_EVAL_SAMPLE_RATE,
) -> list[torch.Tensor]:
    """The fixed RIR pool for one eval run: `build_rir_pool` (the same
    randomized-room generator the train augmenter uses) seeded from `seed`
    via `_seed_for(seed, "rir_pool")`. Built once per run and shared by the
    val and test passes."""
    return build_rir_pool(pool_size, sample_rate=sample_rate, seed=_seed_for(seed, "rir_pool"))


def build_perturbations(
    dataset: VCMDataset,
    rir_pool: list[torch.Tensor],
    seed: int,
) -> list[tuple[int, int, float]]:
    """Per-row perturbation spec for one split: a list (one entry per
    dataset row, in manifest order) of `(rir_index, noise_index, snr_db)`,
    drawn from `torch.Generator().manual_seed(_seed_for(seed, split))`.

    The noise pool is the split's OWN `background_noise` rows (the same
    pool `VCMDataset` feeds the train augmenter), loaded via its cached
    `_noise_pool()`. A split with no `background_noise` rows raises
    `ValueError` rather than silently producing clean audio labeled
    "noisy" -- a manifest without a noise pool cannot support this gate.

    The returned spec is pure data (ints/float): identical (dataset row
    order, rir_pool, seed) inputs always yield an identical list, which is
    what makes two runs on two checkpoints directly comparable.
    """
    noise_pool = dataset._noise_pool()
    if not noise_pool:
        raise ValueError(
            f"noisy eval pass needs a noise pool but split {dataset.split!r} of "
            f"{dataset.manifest_path} has no `background_noise` rows"
        )
    generator = torch.Generator().manual_seed(_seed_for(seed, str(dataset.split)))
    perturbations: list[tuple[int, int, float]] = []
    for _ in dataset.rows:
        rir_index = int(torch.randint(0, len(rir_pool), (1,), generator=generator).item())
        noise_index = int(torch.randint(0, len(noise_pool), (1,), generator=generator).item())
        snr_db = SNR_MIN_DB + torch.rand((), generator=generator).item() * (SNR_MAX_DB - SNR_MIN_DB)
        perturbations.append((rir_index, noise_index, snr_db))
    return perturbations


def apply_perturbation(
    waveform: torch.Tensor,
    rir: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """One compound real-condition perturbation, both steps always applied
    in order: RIR (trimmed back to the input length by `apply_rir`), then
    additive noise at `snr_db` (SNR measured against the reverberated
    signal's power, since the room gain is part of the real condition).
    Pure function of its inputs; preserves the waveform's length."""
    reverberant = apply_rir(waveform, rir)
    return apply_noise(reverberant, noise, snr_db)


@torch.no_grad()
def decode_split_noisy(
    model,
    feature_extractor: LogMelFeatureExtractor,
    dataset: VCMDataset,
    grammar: Grammar,
    beam_width: int,
    device: str | torch.device,
    rir_pool: list[torch.Tensor],
    seed: int,
    required_command_margin: float | None = None,
    score_mode: str = "mean_frame",
) -> list[RowResult]:
    """`vcm.evaluate.decode_split` over a deterministically perturbed copy
    of the split: each row is loaded clean (the dataset's train-only
    augmenter gate is untouched -- `augmenter` is `None` on these
    eval-mode datasets), perturbed per `build_perturbations`, and decoded
    at the most permissive threshold (`-inf`) exactly like `decode_split`,
    so thresholding afterwards is pure arithmetic over cached results."""
    perturbations = build_perturbations(dataset, rir_pool, seed)
    noise_pool = dataset._noise_pool()

    results: list[RowResult] = []
    for i in range(len(dataset)):
        row = dataset.rows[i]
        example = dataset[i]
        rir_index, noise_index, snr_db = perturbations[i]
        waveform = apply_perturbation(
            example.waveform, rir_pool[rir_index], noise_pool[noise_index], snr_db
        )
        decoded = infer_waveform(
            model,
            feature_extractor,
            waveform,
            grammar,
            threshold=NEG_INF_THRESHOLD,
            beam_width=beam_width,
            device=device,
            required_command_margin=required_command_margin,
            score_mode=score_mode,
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


def _split_metrics(
    results: list[RowResult],
    threshold: float,
    raw_rows: list[dict],
    grammar: Grammar,
    manifest_path,
) -> dict:
    """One split's metric block at a fixed threshold. Key-for-key identical
    shape to `evaluate_grammar`'s `test_split` block (built from the same
    helpers) so a clean-vs-noisy comparison is a direct per-key diff."""
    target_rows = [r for r in results if r.bucket == "target_commands"]
    n_target = len(target_rows)
    n_accepted = sum(1 for r in target_rows if _accepted(r, threshold))
    n_exact_correct = sum(
        1 for r in target_rows if _accepted(r, threshold) and r.intent == r.label
    )
    return {
        "n_target_commands": n_target,
        "n_accepted": n_accepted,
        "n_exact_correct": n_exact_correct,
        "accept_rate": n_accepted / n_target if n_target else None,
        "exact_accuracy": n_exact_correct / n_target if n_target else None,
        "per_intent_confusion": confusion_counts(results, threshold),
        "false_accept_rate_babble": false_accept_stats(results, threshold, "babble"),
        "false_accept_rate_silence": false_accept_stats(results, threshold, "silence"),
        "speaker_group_breakdown": speaker_group_breakdown(
            results, threshold, manifest_path=manifest_path
        ),
        "slot_accuracy": slot_accuracy_breakdown(results, raw_rows, grammar, threshold),
    }


@torch.no_grad()
def evaluate_noisy_at_threshold(
    model,
    feature_extractor: LogMelFeatureExtractor,
    val_dataset: VCMDataset,
    test_dataset: VCMDataset,
    grammar: Grammar,
    grammar_label: str,
    beam_width: int,
    device: str | torch.device,
    rir_pool: list[torch.Tensor],
    seed: int,
    threshold: float,
    required_command_margin: float | None = None,
    score_mode: str = "mean_frame",
) -> dict:
    """The full fixed-seed noisy/reverb pass for one grammar: deterministic
    perturbed decodes of the val and test splits, scored at `threshold`
    (that grammar's CLEAN-val-chosen operating threshold, passed in by the
    caller -- this function never chooses a threshold). `val_split` is a
    diagnostic (does the clean-chosen operating point still behave on noisy
    val?); `test_split` is the gate's headline block."""
    return {
        "grammar": grammar_label,
        "seed": seed,
        "threshold": threshold,
        "val_split": _split_metrics(
            decode_split_noisy(
                model, feature_extractor, val_dataset, grammar, beam_width, device,
                rir_pool, seed, required_command_margin, score_mode,
            ),
            threshold,
            val_dataset.rows,
            grammar,
            val_dataset.manifest_path,
        ),
        "test_split": _split_metrics(
            decode_split_noisy(
                model, feature_extractor, test_dataset, grammar, beam_width, device,
                rir_pool, seed, required_command_margin, score_mode,
            ),
            threshold,
            test_dataset.rows,
            grammar,
            test_dataset.manifest_path,
        ),
    }
