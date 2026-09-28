"""Fixed-seed noisy/reverb eval pass for the wakeword DS-CNN (eval gate).

Ticket: .scratch/ambient-reverb-cooccurrence/tickets/00-RECAP.md T4. Mirrors
`me2-iteration3/ME2/src/me2_voicegen/vcm/noisy_eval.py` (the literal
reference implementation) and adapts its decode/scoring half to wakeword's
real clean-path scoring mechanism: the DS-CNN is a plain 3-class argmax
classifier (`wakeword.train.per_class_metrics` /
`wakeword_accent_recall`), with no operating threshold to sweep at all.

Why this gate exists: the training `Augmenter` is train-split-only and, on
the plain `wakeword-sesame` line, ran with `p_rir=0.0` and offline-ESC-50
noise only — so the normal eval report is 100% clean-signal and there was
no way to measure a wakeword checkpoint's robustness to a room with
background talk. This module adds the SEPARATE pass: every row of the val
and test splits gets one deterministic perturbation — one RIR from a fixed
in-memory pool, then one additive noise clip at a fixed SNR — and the
perturbed windows are scored through the SAME argmax path as the clean
eval (features are computed from the perturbed waveform; the model and the
classification rule are untouched).

Determinism contract (the point of the feature): given a manifest and a
seed, the perturbation applied to each row is pure data
(`(rir_index, noise_index, snr_db)`, drawn in manifest order from
`torch.Generator().manual_seed(zlib.crc32(f"{seed}:{tag}"))`), so the same
(manifest, seed) pair yields a bit-identical perturbed signal on every
run, against any checkpoint. Scores are therefore comparable
checkpoint-to-checkpoint by construction.

Threshold rule: VCM's gate scores the noisy decodes at the CLEAN-val-chosen
operating threshold and never re-sweeps it on noisy val. The wakeword
DS-CNN has no threshold to choose or sweep — its decision rule is argmax
over the three class scores — so the rule carries over trivially: the
identical argmax scoring path is applied to the perturbed waveforms, and
nothing is re-fit on the noisy split (checkpoint-to-checkpoint
comparability is preserved the same way).

Only `common.augment.apply_rir` / `apply_noise` / `build_rir_pool` are
reused here. SpecAugment is deliberately NOT part of this pass (a
training-only feature-space regularizer, not a real acoustic condition),
and neither is temporal-shift windowing (a train-only placement
regularizer; eval windows are deterministic `center_window`). Severity
ranges match what training intended to cover: RT60 U[0.1, 0.5] s rooms
(same `build_rir_pool` generation the train augmenter uses) and SNR
U[5, 25] dB (the train augmenter's `SNR_MIN_DB`/`SNR_MAX_DB`).
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from me2_voicegen.wakeword.dataset import WakewordDataset
from me2_voicegen.wakeword.model import LABELS, build_model
from me2_voicegen.wakeword.train import (
    DEFAULT_MANIFEST,
    DEFAULT_NOISE_ROOT,
    classify_ww_row_is_filipino,
    license_note,
    per_class_metrics,
)

NOISY_EVAL_RIR_POOL_SIZE = 200
NOISY_EVAL_SAMPLE_RATE = 16000

PERTURBATION_DOC = (
    "rir+additive_noise: every row gets one RIR (RT60 U[0.1,0.5]s randomized "
    "room from a fixed 200-entry pool) then one noise clip from the run's "
    "--noise-root at SNR U[5,25] dB, both always applied, in that order"
)
THRESHOLD_RULE = (
    "scored through the same argmax per-class path as the clean eval "
    "(wakeword.train.per_class_metrics / wakeword_accent_recall); the "
    "DS-CNN has no operating threshold to sweep, so the vcm.noisy_eval "
    "rule (clean-val chosen threshold, never re-swept on noisy val) holds "
    "trivially — nothing is re-fit on the noisy split"
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
    dataset: WakewordDataset,
    rir_pool: list[torch.Tensor],
    seed: int,
) -> list[tuple[int, int, float]]:
    """Per-row perturbation spec for one split: a list (one entry per
    dataset row, in manifest order) of `(rir_index, noise_index, snr_db)`,
    drawn from `torch.Generator().manual_seed(_seed_for(seed, split))`.

    The noise pool is the run's `--noise-root` wavs (the same pool the
    train-time augmenter mixes from), loaded via the dataset's cached
    `_noise_pool()`. A run with no noise pool raises `ValueError` rather
    than silently producing clean audio labeled "noisy" -- a manifest run
    without a noise pool cannot support this gate.

    The returned spec is pure data (ints/float): identical (dataset row
    order, rir_pool, seed) inputs always yield an identical list, which is
    what makes two runs on two checkpoints directly comparable.
    """
    noise_pool = dataset._noise_pool()
    if not noise_pool:
        raise ValueError(
            f"noisy eval pass needs a noise pool but split {dataset.split!r} of "
            f"{dataset.manifest_path} has no noise wavs under "
            f"{dataset.noise_root} (pass --noise-root to the run)"
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
    Pure function of its inputs; preserves the waveform's length.

    The input is the dataset's own eval window (VAD-cropped content,
    center-windowed to 1.5 s) — i.e. exactly what the clean eval path
    feeds the model — so the noisy pass perturbs the same signal the clean
    pass scores, one acoustic condition at a time."""
    reverberant = apply_rir(waveform, rir)
    return apply_noise(reverberant, noise, snr_db)


class _FeatureLabelSet(Dataset):
    """In-memory `(features, label)` pairs shaped exactly like
    `wakeword.dataset.collate_fn`'s batch output, so the clean-path
    `per_class_metrics` scoring function can be reused unchanged."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {"features": self.features[index], "labels": self.labels[index]}


def _stacked_collate(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


@torch.no_grad()
def score_split_noisy(
    model: nn.Module,
    feature_extractor: LogMelFeatureExtractor,
    dataset: WakewordDataset,
    rir_pool: list[torch.Tensor],
    seed: int,
    device: str | torch.device,
) -> dict:
    """One split's deterministic noisy/reverb pass: every row is loaded
    clean (the dataset's train-only augmenter gate is untouched —
    `augmenter` is `None` on these eval-mode datasets), perturbed per
    `build_perturbations`, and scored through the SAME argmax path as the
    clean eval: `per_class_metrics` on the perturbed features (reused
    verbatim, via a loader shaped like the collate output) plus
    `wakeword_accent_recall`'s `_wakeword_`-by-accent bucketing over the
    same predictions."""
    perturbations = build_perturbations(dataset, rir_pool, seed)
    noise_pool = dataset._noise_pool()

    features: list[torch.Tensor] = []
    labels: list[int] = []
    for i in range(len(dataset)):
        example = dataset[i]
        rir_index, noise_index, snr_db = perturbations[i]
        waveform = apply_perturbation(
            example.waveform, rir_pool[rir_index], noise_pool[noise_index], snr_db
        )
        features.append(feature_extractor(waveform))
        labels.append(example.label)
    feature_tensor = torch.stack(features)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    def _loader() -> DataLoader:
        # A fresh loader per pass: torch DataLoaders are single-iteration.
        return DataLoader(
            _FeatureLabelSet(feature_tensor, label_tensor),
            batch_size=64,
            shuffle=False,
            collate_fn=_stacked_collate,
        )

    metrics = per_class_metrics(model, _loader(), device)

    # `_wakeword_` recall by accent, same bucketing as
    # wakeword.train.wakeword_accent_recall (which re-loads clean waveforms
    # from the dataset and so cannot be reused as-is for a perturbed pass).
    wakeword_idx = LABELS.index("_wakeword_")
    counts = {"filipino": [0, 0], "non_filipino": [0, 0]}  # [correct, total]
    row_idx = 0
    for batch in _loader():
        preds = model(batch["features"].to(device)).argmax(dim=-1).cpu()
        for i in range(batch["labels"].shape[0]):
            row = dataset.rows[row_idx]
            row_idx += 1
            if batch["labels"][i].item() != wakeword_idx:
                continue
            bucket = "filipino" if classify_ww_row_is_filipino(row) else "non_filipino"
            counts[bucket][1] += 1
            if preds[i].item() == wakeword_idx:
                counts[bucket][0] += 1
    accent_recall = {
        bucket: {"correct": correct, "total": total, "recall": (correct / total) if total else None}
        for bucket, (correct, total) in counts.items()
    }

    return {
        "n_rows": len(dataset),
        **metrics,
        "accent_recall": accent_recall,
    }


@torch.no_grad()
def evaluate_noisy(
    model: nn.Module,
    feature_extractor: LogMelFeatureExtractor,
    val_dataset: WakewordDataset,
    test_dataset: WakewordDataset,
    rir_pool: list[torch.Tensor],
    seed: int,
    device: str | torch.device,
) -> dict:
    """The full fixed-seed noisy/reverb pass: deterministic perturbed
    scoring of the val and test splits. `val_split` is a diagnostic (does
    the model still behave on noisy val?); `test_split` is the gate's
    headline block."""
    return {
        "seed": seed,
        "val_split": score_split_noisy(model, feature_extractor, val_dataset, rir_pool, seed, device),
        "test_split": score_split_noisy(model, feature_extractor, test_dataset, rir_pool, seed, device),
    }


def load_checkpoint(path: Path, device: str | torch.device) -> tuple[nn.Module, dict]:
    """Same load path as `wakeword.train --eval-only` (preset read from the
    checkpoint, state dict loaded strictly), kept as a function so tests
    can substitute a stub model without touching disk."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["preset"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, {k: v for k, v in ckpt.items() if k != "model_state_dict"}


def _noisy_report_md(report: dict, noisy: dict) -> list[str]:
    lines = [
        "# Wakeword DS-CNN noisy/reverb eval report",
        "",
        f"Checkpoint: `{report['checkpoint_path']}` (preset={report['checkpoint_meta'].get('preset')}, "
        f"epoch={report['checkpoint_meta'].get('epoch')}).",
        "",
        f"## Noisy/reverb eval gate (fixed seed {noisy['seed']})",
        "",
        f"Perturbation: {report['perturbation']}. The perturbation is fixed by "
        f"(`manifest, seed {noisy['seed']}`): identical inputs give a bit-identical "
        "perturbed signal on every run, against any checkpoint.",
        "",
        f"**Threshold rule:** {report['threshold_rule']}.",
        "",
    ]
    for split in ("val_split", "test_split"):
        block = noisy[split]
        lines += [f"### {split} ({block['n_rows']} rows)", "",
                  "| label | precision | recall | f1 | support |",
                  "|---|---|---|---|---|"]
        for label in LABELS:
            m = block["per_class"][label]
            lines.append(f"| `{label}` | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
        lines += ["", "`_wakeword_` recall by voice accent:", "",
                  "| group | recall | correct | total |",
                  "|---|---|---|---|"]
        for bucket in ("filipino", "non_filipino"):
            b = block["accent_recall"][bucket]
            recall_str = f"{b['recall']:.3f}" if b["recall"] is not None else "n/a"
            lines.append(f"| {bucket} | {recall_str} | {b['correct']} | {b['total']} |")
        lines.append("")
    lines += [report["license"], ""]
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="e.g. cuda:0, cpu")
    parser.add_argument("--noise-root", type=Path, default=DEFAULT_NOISE_ROOT)
    parser.add_argument(
        "--noisy-eval-seed",
        type=int,
        required=True,
        help=(
            "fixed seed for the noisy/reverb pass: one deterministic "
            "RIR+additive-noise perturbation per val/test row, scored through "
            "the clean eval's argmax path. Same manifest + same seed => "
            "bit-identical perturbed signal on every run, so scores are "
            "comparable checkpoint-to-checkpoint"
        ),
    )
    parser.add_argument(
        "--noisy-eval-rir-pool-size",
        type=int,
        default=NOISY_EVAL_RIR_POOL_SIZE,
        help=(
            "RIR pool size for the noisy eval pass (default 200, same as "
            "the training augmenter). The pool size is part of the result's "
            "identity -- changing it changes the perturbation, so keep it "
            "fixed when comparing checkpoints"
        ),
    )
    parser.add_argument(
        "--eval-splits",
        default="val,test",
        help="comma-separated splits to score (default 'val,test')",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    seed = args.noisy_eval_seed
    rir_pool_size = args.noisy_eval_rir_pool_size if args.noisy_eval_rir_pool_size > 0 else NOISY_EVAL_RIR_POOL_SIZE

    model, checkpoint_meta = load_checkpoint(args.checkpoint, device)
    model.eval()
    feature_extractor = LogMelFeatureExtractor()

    rir_pool = build_rir_pool_for_seed(seed, rir_pool_size)
    splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    noisy: dict = {"seed": seed}
    for split in splits:
        dataset = WakewordDataset(
            args.manifest, split=split, augmenter=None, shift=False, noise_root=args.noise_root
        )
        noisy[f"{split}_split"] = score_split_noisy(model, feature_extractor, dataset, rir_pool, seed, device)

    report = {
        "license": license_note(args.manifest),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "manifest_path": str(args.manifest),
        "device": str(device),
        "noise_root": str(args.noise_root),
        "rir_pool_size": rir_pool_size,
        "rt60_range_s": [RT60_MIN, RT60_MAX],
        "snr_range_db": [SNR_MIN_DB, SNR_MAX_DB],
        "perturbation": PERTURBATION_DOC,
        "threshold_rule": THRESHOLD_RULE,
        **noisy,
    }

    metadata_dir = args.out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "noisy_eval_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    (metadata_dir / "noisy_eval_report.md").write_text("\n".join(_noisy_report_md(report, noisy)), encoding="utf-8")

    for split in splits:
        block = noisy[f"{split}_split"]
        wk = block["per_class"]["_wakeword_"]
        print(
            f"{split}: _wakeword_ recall={wk['recall']:.3f} f1={wk['f1']:.3f} "
            f"(seed={seed}, pool={rir_pool_size}, SNR U[{SNR_MIN_DB:.0f},{SNR_MAX_DB:.0f}] dB)"
        )
    print(f"wrote {metadata_dir / 'noisy_eval_report.json'}")
    print(f"wrote {metadata_dir / 'noisy_eval_report.md'}")


if __name__ == "__main__":
    main()
