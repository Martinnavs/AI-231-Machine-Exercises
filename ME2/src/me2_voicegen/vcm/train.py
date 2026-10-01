"""CTC training CLI for the toy VCM acoustic model (`vcm.model`).

Trains `MatchboxNetCTC` against `out/conversions/v2/test_set/manifest.csv`'s
`train` split (or any other manifest passed via `--manifest`), with Task
02's RIR + additive-noise + SpecAugment pipeline all active by default.
`--preset` selects the model capacity: `default` (~254k params) is the
usual choice; `optionc` (~754k params) is a mid-scale config for testing
whether more parameters improve decode accuracy on the same manifest;
`optiond` (~1.01M params) is the normal-scale config;
`spec-scale` (~2.14M params) exists only for size-budget reporting, per
ticket 04's Non-Goals, and is not expected to be trained here.

License note (docs/VCM-CONTRACT.md section 8): `background_noise` (feeding
the `silence` bucket) is ESC-50, CC-BY-NC-SA-4.0. Any checkpoint trained on
this `test_set/` therefore inherits CC-BY-NC-SA-4.0 (non-commercial,
share-alike) regardless of whether a given run's train split happened to
sample a `background_noise` row -- this note is written into every
checkpoint and loss-history JSON this CLI produces.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from me2_voicegen.common.augment import Augmenter
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.dataset import VCMDataset, collate_fn
from me2_voicegen.vcm.model import (
    MODEL_TYPE_KEY,
    PRESETS,
    build_model,
    estimated_int8_bytes,
    model_type_for_config,
    param_count,
)
from me2_voicegen.vcm.text import normalize_text, resolve_transcript

LICENSE_NOTE = (
    "Checkpoint trained on out/conversions/v2/test_set/, which includes "
    "background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. "
    "Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this "
    "data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike "
    "on redistribution."
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
)


def build_train_collate(feature_extractor: LogMelFeatureExtractor, augmenter: Augmenter | None):
    def _collate(batch):
        out = collate_fn(batch, feature_extractor)
        if augmenter is not None and augmenter.p_specaugment > 0:
            feats = out["features"]
            lengths = out["input_lengths"]
            augmented = torch.zeros_like(feats)
            for i in range(feats.shape[0]):
                valid = int(lengths[i].item())
                if valid == 0:
                    continue
                augmented[i, :, :valid] = augmenter.augment_features(feats[i, :, :valid])
            out["features"] = augmented
        return out

    return _collate


def greedy_decode(logits: torch.Tensor) -> str:
    """`(T, alphabet_size)` logits for one utterance -> decoded text."""
    ids = logits.argmax(dim=-1).tolist()
    return alphabet.decode(alphabet.collapse(ids))


def _batchnorm_modules(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


BatchNormSnapshot = list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def snapshot_batchnorm_stats(model: nn.Module) -> BatchNormSnapshot:
    """Clone every BatchNorm layer's `running_mean`/`running_var`/
    `num_batches_tracked` so a step that turns out non-finite can be
    undone. A `forward()` call updates these running stats as a side
    effect *before* the loss is even computed, so a post-hoc
    `if not finite: continue` on the loss can skip the backward/optimizer
    step but cannot un-poison stats already written into the BN buffers --
    that's what leaves `eval()` mode permanently broken. Snapshotting
    before the forward pass and restoring on a bad step is what actually
    prevents the poisoning."""
    return [
        (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for m in _batchnorm_modules(model)
    ]


def restore_batchnorm_stats(model: nn.Module, snapshot: BatchNormSnapshot) -> None:
    for module, (running_mean, running_var, num_batches_tracked) in zip(
        _batchnorm_modules(model), snapshot
    ):
        module.running_mean.copy_(running_mean)
        module.running_var.copy_(running_var)
        module.num_batches_tracked.copy_(num_batches_tracked)


def ctc_min_frames(target_ids: list[int]) -> int:
    """Minimum output frames CTC needs for a target: its length plus one blank
    between each pair of adjacent repeated symbols."""
    return len(target_ids) + sum(1 for a, b in zip(target_ids, target_ids[1:]) if a == b)


def count_ctc_infeasible(
    output_lengths: torch.Tensor, target_ids: torch.Tensor, target_len: torch.Tensor
) -> int:
    """Count batch items whose (concatenated) target cannot fit in its output
    length. `zero_infinity=True` silently zeroes these, so callers log the count."""
    count = 0
    offset = 0
    ids = target_ids.tolist()
    for out_len, n in zip(output_lengths.tolist(), target_len.tolist()):
        if ctc_min_frames(ids[offset : offset + n]) > out_len:
            count += 1
        offset += n
    return count


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    stats: dict | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_len = batch["target_len"].to(device)

        logits = model(features)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        output_lengths = model.output_lengths(input_lengths)
        if stats is not None:
            stats["ctc_infeasible"] = stats.get("ctc_infeasible", 0) + count_ctc_infeasible(
                output_lengths, target_ids, target_len
            )
        loss = criterion(log_probs, target_ids, output_lengths, target_len)

        n = features.shape[0]
        total_loss += float(loss.item()) * n
        total_examples += n
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def sample_decodes(
    model: nn.Module,
    dataset: VCMDataset,
    feature_extractor: LogMelFeatureExtractor,
    device: torch.device,
    n_samples: int = 3,
) -> list[tuple[str, str]]:
    """Greedy-decode a few held-out loss-bearing val clips, for human
    sanity-checking. Returns `[(expected, decoded), ...]`."""
    model.eval()
    samples = []
    for idx in dataset.loss_bearing_indices[:n_samples]:
        row = dataset.rows[idx]
        transcript = resolve_transcript(row)
        expected = normalize_text(transcript) if transcript else "<empty>"

        example = dataset[idx]
        features = feature_extractor(example.waveform).unsqueeze(0).to(device)
        logits = model(features)[0]
        decoded = greedy_decode(logits.cpu())
        samples.append((expected, decoded))
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("out/vcm"))
    parser.add_argument(
        "--preset",
        default="default",
        choices=sorted(PRESETS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--device", default=None, help="e.g. cuda:2, cuda:0, cpu (no auto-pick; caller chooses a free index)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=10, help="epochs without val-loss improvement before early stop")
    parser.add_argument("--amp", action="store_true", default=None, help="default: on for cuda, off for cpu")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--p-rir", type=float, default=0.3)
    parser.add_argument("--p-noise", type=float, default=0.5)
    parser.add_argument("--p-specaugment", type=float, default=0.5)
    parser.add_argument("--time-check-every", type=int, default=50, help="check the wall-clock cap every N train batches")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--onecycle-epochs",
        type=int,
        default=None,
        help=(
            "epochs the OneCycleLR schedule is shaped for (its own anneal-to-zero "
            "horizon). Defaults to --max-epochs (tracks the actual epoch budget) so "
            "the schedule can't finish annealing to ~0 and then freeze partway "
            "through a longer run -- a fixed default here previously caused exactly "
            "that (see ticket 04's Execution Log). Only set this lower than "
            "--max-epochs deliberately, e.g. to intentionally anneal faster than "
            "the full budget."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.onecycle_epochs is None:
        args.onecycle_epochs = args.max_epochs
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cpu")
    use_amp = args.amp if args.amp is not None else device.type == "cuda"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out_dir / "checkpoints"
    metadata_dir = args.out_dir / "metadata"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = LogMelFeatureExtractor()
    train_augmenter = Augmenter(
        p_rir=args.p_rir,
        p_noise=args.p_noise,
        p_specaugment=args.p_specaugment,
        seed=args.seed,
    )

    train_dataset = VCMDataset(args.manifest, split="train", augmenter=train_augmenter)
    val_dataset = VCMDataset(args.manifest, split="val", augmenter=None)

    train_subset = Subset(train_dataset, train_dataset.loss_bearing_indices)
    val_subset = Subset(val_dataset, val_dataset.loss_bearing_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_train_collate(feature_extractor, train_augmenter),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=build_train_collate(feature_extractor, None),
    )

    config = PRESETS[args.preset]
    model = build_model(args.preset).to(device)
    model_type = model_type_for_config(config)
    total_stride = getattr(model, "total_stride", 1)
    print(f"model preset={args.preset} params={param_count(model):,} est_int8_bytes={estimated_int8_bytes(model):,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # One-cycle (built-in LR warmup, then anneal) rather than a bare cosine
    # decay from the full LR: an early diagnostic run showed the model
    # collapsing to an always-predict-blank CTC solution within a couple
    # of epochs under a flat high starting LR -- see this ticket's
    # Execution Log. Warmup + grad clipping (below) is the standard fix,
    # not a hyperparameter search.
    steps_per_epoch = max(len(train_loader), 1)
    onecycle_total_steps = steps_per_epoch * args.onecycle_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=onecycle_total_steps,
        pct_start=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler_steps_taken = 0

    epoch0_val_stats: dict = {}
    epoch0_val_loss = run_eval(model, val_loader, criterion, device, epoch0_val_stats)
    if epoch0_val_stats.get("ctc_infeasible"):
        print(f"WARNING: {epoch0_val_stats['ctc_infeasible']} CTC-infeasible val items")
    print(f"epoch 0 (pre-train) val_loss={epoch0_val_loss:.4f}")

    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.monotonic()
    deadline_hit = False
    nan_or_inf_seen = False
    checkpoint_written = False
    epoch = 0
    ctc_infeasible_total_train = 0

    for epoch in range(1, args.max_epochs + 1):
        elapsed_s = time.monotonic() - start_time
        if elapsed_s >= args.max_minutes * 60:
            deadline_hit = True
            break

        model.train()
        train_loss_total = 0.0
        train_examples = 0
        ctc_infeasible_train = 0
        for step, batch in enumerate(train_loader):
            if step % args.time_check_every == 0:
                if time.monotonic() - start_time >= args.max_minutes * 60:
                    deadline_hit = True
                    break

            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_len = batch["target_len"].to(device)

            bn_snapshot = snapshot_batchnorm_stats(model)
            optimizer.zero_grad(set_to_none=True)
            output_lengths = model.output_lengths(input_lengths)
            ctc_infeasible_train += count_ctc_infeasible(
                output_lengths.cpu(), target_ids.cpu(), target_len.cpu()
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(features)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, target_ids, output_lengths, target_len)

            if not torch.isfinite(loss):
                nan_or_inf_seen = True
                restore_batchnorm_stats(model, bn_snapshot)
                print(
                    f"WARNING: non-finite loss at epoch {epoch} step {step}: {loss.item()} "
                    "-- batch skipped, BatchNorm running stats restored to pre-step values"
                )
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler_steps_taken < onecycle_total_steps:
                scheduler.step()
                scheduler_steps_taken += 1

            n = features.shape[0]
            train_loss_total += float(loss.item()) * n
            train_examples += n

        if deadline_hit:
            break

        train_loss = train_loss_total / max(train_examples, 1)
        val_stats: dict = {}
        val_loss = run_eval(model, val_loader, criterion, device, val_stats)
        ctc_infeasible_val = val_stats.get("ctc_infeasible", 0)
        ctc_infeasible_total_train += ctc_infeasible_train
        if ctc_infeasible_train or ctc_infeasible_val:
            print(
                f"WARNING: {ctc_infeasible_train} train / {ctc_infeasible_val} val "
                "CTC-infeasible items (zeroed by zero_infinity)"
            )
        elapsed_s = time.monotonic() - start_time

        samples = sample_decodes(model, val_dataset, feature_extractor, device)
        print(f"epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed_s={elapsed_s:.1f}")
        for expected, decoded in samples:
            print(f"  expected={expected!r} decoded={decoded!r}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": elapsed_s,
                "ctc_infeasible_train": ctc_infeasible_train,
                "ctc_infeasible_val": ctc_infeasible_val,
                "samples": [{"expected": e, "decoded": d} for e, d in samples],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "preset": args.preset,
                    MODEL_TYPE_KEY: model_type,
                    "config": dataclasses.asdict(config),
                    "alphabet_size": alphabet.ALPHABET_SIZE,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "license": LICENSE_NOTE,
                },
                checkpoints_dir / "checkpoint.pt",
            )
            checkpoint_written = True
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping: no val_loss improvement in {args.patience} epochs")
                break

    total_wall_s = time.monotonic() - start_time
    loss_history = {
        "license": LICENSE_NOTE,
        "seed": args.seed,
        "preset": args.preset,
        "model_type": model_type,
        "total_stride": total_stride,
        "ctc_infeasible_total_train": ctc_infeasible_total_train,
        "device": str(device),
        "max_minutes": args.max_minutes,
        "epoch0_val_loss": epoch0_val_loss,
        "epoch0_ctc_infeasible_val": epoch0_val_stats.get("ctc_infeasible", 0),
        "epochs_run": epoch,
        "deadline_hit": deadline_hit,
        "nan_or_inf_seen": nan_or_inf_seen,
        "best_val_loss": best_val_loss,
        "total_wall_clock_s": total_wall_s,
        "history": history,
    }
    with (metadata_dir / "loss_history.json").open("w") as f:
        json.dump(loss_history, f, indent=2)

    print(
        f"done: epochs={epoch} best_val_loss={best_val_loss:.4f} "
        f"wall_clock_s={total_wall_s:.1f} deadline_hit={deadline_hit} nan_or_inf_seen={nan_or_inf_seen}"
    )

    if not checkpoint_written:
        raise SystemExit(
            f"ERROR: training run finished after {epoch} epoch(s) without ever writing a "
            f"checkpoint (best_val_loss={best_val_loss}). This is a failed run, not a normal "
            "'done' exit -- see loss_history.json for per-epoch detail."
        )


if __name__ == "__main__":
    main()
