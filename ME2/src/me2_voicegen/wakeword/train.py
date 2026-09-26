"""Cross-entropy training CLI for the wakeword DS-CNN (`wakeword.model`),
feature `wakeword-dscnn` (feature-engineering/wakeword-dscnn/SPEC.md).

Trains `DSCNN` against `out/conversions/v2/wakeword/manifest.csv`'s `train`
split (or any other manifest passed via `--manifest`) to classify each
fixed-window clip as `_wakeword_`/`_unknown_`/`_silence_`. Dynamic SNR
mixing and SpecAugment (`common.augment.Augmenter`, RIR left at its
default-off `p_rir=0.0` -- the handoff doc explicitly excludes RIR for
this toy model) and temporal-shift windowing (`wakeword.augment.
shift_waveform`, on by default) are both active for the train split only.

License note (docs/WAKEWORD-DATASET-CONTRACT.md section 7): the final
assembled dataset includes `background_noise` (ESC-50, CC-BY-NC-SA-4.0)
additively mixed into `adversaries_noisy`/`positives_converted_noisy`. Any
checkpoint trained on it therefore inherits CC-BY-NC-SA-4.0 (non-commercial,
share-alike) regardless of whether a given run's train split happened to
sample a `*_noisy` row -- this note is written into every checkpoint and
eval-report this CLI produces.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from me2_voicegen.common.augment import Augmenter
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS, WakewordDataset, collate_fn
from me2_voicegen.wakeword.model import (
    LABELS,
    PRESETS,
    build_model,
    estimated_int8_bytes,
    param_count,
)

LICENSE_NOTE = (
    "Checkpoint trained on out/conversions/v2/wakeword/, which includes "
    "background_noise (ESC-50, CC-BY-NC-SA-4.0) additively mixed into "
    "adversaries_noisy/positives_converted_noisy. Per "
    "docs/WAKEWORD-DATASET-CONTRACT.md section 7, any checkpoint trained on "
    "this data inherits CC-BY-NC-SA-4.0: non-commercial use only, "
    "share-alike on redistribution."
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "out" / "conversions" / "v2" / "wakeword" / "manifest.csv"
)
DEFAULT_NOISE_ROOT = (
    Path(__file__).resolve().parents[3] / "out" / "conversions" / "v2" / "background_noise"
)


def build_train_collate(feature_extractor: LogMelFeatureExtractor):
    def _collate(batch):
        return collate_fn(batch, feature_extractor)

    return _collate


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        n = features.shape[0]
        total_loss += float(loss.item()) * n
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_examples += n
    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


@torch.no_grad()
def per_class_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Confusion matrix + per-class precision/recall/F1 on `loader` --
    SPEC.md's Edges: `_silence_` is 10% of the dataset vs. 45%/45%, so
    per-class metrics are reported rather than a single overall accuracy
    figure that would hide that imbalance."""
    model.eval()
    n_classes = len(LABELS)
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        preds = model(features).argmax(dim=-1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t][p] += 1

    per_class: dict[str, dict] = {}
    for i, label in enumerate(LABELS):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted_positive = sum(confusion[r][i] for r in range(n_classes))
        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    return {"confusion_matrix": confusion, "labels": list(LABELS), "per_class": per_class}


_FILIPINO_REF_VOICE_RE = re.compile(r"^(tagalog|ilonggo)")
_FILIPINO_SOURCE_DATASETS = frozenset({"fil50_persona", "fil50_persona_noisy"})


def classify_ww_row_is_filipino(row: dict) -> bool:
    """Same "what counts as Filipino" definition
    `accent_balance.plan_jobs`/`accent_balance.collate` use for `_wakeword_`
    rows -- kept as a local copy rather than an import so this module (which
    every `wakeword-train`/`wakeword-bench` Makefile recipe touches) doesn't
    pick up a dependency on the `accent-balance-fil50` feature package for a
    two-line check."""
    if row.get("source_dataset") in _FILIPINO_SOURCE_DATASETS:
        return True
    return bool(_FILIPINO_REF_VOICE_RE.match(row.get("ref_voice") or ""))


@torch.no_grad()
def wakeword_accent_recall(model: nn.Module, dataset: WakewordDataset, device: torch.device) -> dict:
    """`_wakeword_`-label recall split by Filipino- vs non-Filipino-voiced
    row (feature `accent-balance-fil50`,
    .scratch/accent-balance-fil50/tickets/00-RECAP.md T6) -- there was
    previously no accent breakdown for the wakeword model at all, only the
    pooled per-class table `per_class_metrics` already reports. Iterates
    `dataset` directly (not shuffled) so batch order lines up with
    `dataset.rows` one-to-one, the same assumption `WakewordDataset.rows`'s
    own docstring establishes for `__getitem__`."""
    model.eval()
    feature_extractor = LogMelFeatureExtractor()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=build_train_collate(feature_extractor))
    wakeword_idx = LABELS.index("_wakeword_")
    counts = {"filipino": [0, 0], "non_filipino": [0, 0]}  # [correct, total]

    row_idx = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"]
        preds = model(features).argmax(dim=-1).cpu()
        for i in range(labels.shape[0]):
            row = dataset.rows[row_idx]
            row_idx += 1
            if labels[i].item() != wakeword_idx:
                continue
            bucket = "filipino" if classify_ww_row_is_filipino(row) else "non_filipino"
            counts[bucket][1] += 1
            if preds[i].item() == wakeword_idx:
                counts[bucket][0] += 1

    return {
        bucket: {"correct": correct, "total": total, "recall": (correct / total) if total else None}
        for bucket, (correct, total) in counts.items()
    }


def write_eval_report(
    *,
    model: nn.Module,
    val_dataset: WakewordDataset,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_meta: dict,
    checkpoint_path: Path,
    manifest_path: Path,
    metadata_dir: Path,
) -> dict:
    """Shared by the end of a training run and `--eval-only` mode, so both
    paths produce byte-for-byte the same report shape."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metrics = per_class_metrics(model, val_loader, device)
    accent_recall = wakeword_accent_recall(model, val_dataset, device)

    eval_report = {
        "license": LICENSE_NOTE,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_meta": checkpoint_meta,
        "manifest_path": str(manifest_path),
        "device": str(device),
        **metrics,
        "accent_recall": accent_recall,
    }
    with (metadata_dir / "eval_report.json").open("w") as f:
        json.dump(eval_report, f, indent=2)

    md_lines = [
        "# Wakeword DS-CNN eval report",
        "",
        f"Checkpoint: `{eval_report['checkpoint_path']}` (preset={checkpoint_meta.get('preset')}, epoch={checkpoint_meta.get('epoch')}).",
        "",
        "| label | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for label in LABELS:
        m = metrics["per_class"][label]
        md_lines.append(f"| `{label}` | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")

    md_lines += [
        "",
        "## `_wakeword_` recall by voice accent",
        "",
        "| group | recall | correct | total |",
        "|---|---|---|---|",
    ]
    for bucket in ("filipino", "non_filipino"):
        b = accent_recall[bucket]
        recall_str = f"{b['recall']:.3f}" if b["recall"] is not None else "n/a"
        md_lines.append(f"| {bucket} | {recall_str} | {b['correct']} | {b['total']} |")

    md_lines += ["", eval_report["license"], ""]
    (metadata_dir / "eval_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"wrote {metadata_dir / 'eval_report.json'}")
    print(f"wrote {metadata_dir / 'eval_report.md'}")
    return eval_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("out/wakeword"))
    parser.add_argument("--preset", default="default", choices=list(PRESETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--device", default=None, help="e.g. cuda:0, cpu (no auto-pick; caller chooses a free index)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=10, help="epochs without val-loss improvement before early stop")
    parser.add_argument("--amp", action="store_true", default=None, help="default: on for cuda, off for cpu")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--p-noise", type=float, default=0.5, help="dynamic SNR-mixing probability (handoff doc's requirement)")
    parser.add_argument("--p-specaugment", type=float, default=0.5)
    parser.add_argument("--noise-root", type=Path, default=DEFAULT_NOISE_ROOT)
    parser.add_argument(
        "--no-shift",
        dest="shift",
        action="store_false",
        default=True,
        help="disable temporal-shift windowing augmentation (on by default, per the handoff doc)",
    )
    parser.add_argument("--time-check-every", type=int, default=50, help="check the wall-clock cap every N train batches")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--onecycle-epochs", type=int, default=None, help="defaults to --max-epochs")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="load --checkpoint and write eval_report.json/.md against --eval-split, without training "
        "(feature accent-balance-fil50: scores an existing checkpoint, e.g. out/wakeword, against a "
        "new manifest without retraining)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="required with --eval-only")
    parser.add_argument("--eval-split", default="val", help="split to evaluate in --eval-only mode (default: val, matching the end-of-training report)")
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

    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--eval-only requires --checkpoint")
        ckpt = torch.load(args.checkpoint, map_location=device)
        eval_model = build_model(ckpt["preset"]).to(device)
        eval_model.load_state_dict(ckpt["model_state_dict"])
        eval_feature_extractor = LogMelFeatureExtractor()
        eval_dataset = WakewordDataset(args.manifest, split=args.eval_split, augmenter=None, shift=False)
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=build_train_collate(eval_feature_extractor),
        )
        write_eval_report(
            model=eval_model, val_dataset=eval_dataset, val_loader=eval_loader, device=device,
            checkpoint_meta={k: v for k, v in ckpt.items() if k != "model_state_dict"},
            checkpoint_path=args.checkpoint, manifest_path=args.manifest, metadata_dir=metadata_dir,
        )
        return

    feature_extractor = LogMelFeatureExtractor()
    train_augmenter = Augmenter(p_noise=args.p_noise, p_specaugment=args.p_specaugment, seed=args.seed)
    train_generator = torch.Generator().manual_seed(args.seed)

    train_dataset = WakewordDataset(
        args.manifest,
        split="train",
        augmenter=train_augmenter,
        shift=args.shift,
        noise_root=args.noise_root,
        generator=train_generator,
    )
    val_dataset = WakewordDataset(args.manifest, split="val", augmenter=None, shift=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_train_collate(feature_extractor),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=build_train_collate(feature_extractor),
    )

    config = PRESETS[args.preset]
    model = build_model(args.preset).to(device)
    print(f"model preset={args.preset} params={param_count(model):,} est_int8_bytes={estimated_int8_bytes(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    onecycle_total_steps = steps_per_epoch * args.onecycle_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=onecycle_total_steps, pct_start=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler_steps_taken = 0

    epoch0_val_loss, epoch0_val_acc = run_eval(model, val_loader, criterion, device)
    print(f"epoch 0 (pre-train) val_loss={epoch0_val_loss:.4f} val_acc={epoch0_val_acc:.4f}")

    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_time = time.monotonic()
    deadline_hit = False
    nan_or_inf_seen = False
    checkpoint_written = False
    epoch = 0

    for epoch in range(1, args.max_epochs + 1):
        elapsed_s = time.monotonic() - start_time
        if elapsed_s >= args.max_minutes * 60:
            deadline_hit = True
            break

        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for step, batch in enumerate(train_loader):
            if step % args.time_check_every == 0:
                if time.monotonic() - start_time >= args.max_minutes * 60:
                    deadline_hit = True
                    break

            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                nan_or_inf_seen = True
                print(f"WARNING: non-finite loss at epoch {epoch} step {step}: {loss.item()} -- batch skipped")
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
        val_loss, val_acc = run_eval(model, val_loader, criterion, device)
        elapsed_s = time.monotonic() - start_time
        print(f"epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} elapsed_s={elapsed_s:.1f}")
        train_dataset.log_fallback_summary()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_s": elapsed_s,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "preset": args.preset,
                    "config": dataclasses.asdict(config),
                    "labels": list(LABELS),
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "window_seconds": WAKEWORD_WINDOW_SECONDS,
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
        "device": str(device),
        "max_minutes": args.max_minutes,
        "epoch0_val_loss": epoch0_val_loss,
        "epoch0_val_acc": epoch0_val_acc,
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

    ckpt = torch.load(checkpoints_dir / "checkpoint.pt", map_location=device)
    best_model = build_model(ckpt["preset"]).to(device)
    best_model.load_state_dict(ckpt["model_state_dict"])
    write_eval_report(
        model=best_model, val_dataset=val_dataset, val_loader=val_loader, device=device,
        checkpoint_meta={k: v for k, v in ckpt.items() if k != "model_state_dict"},
        checkpoint_path=checkpoints_dir / "checkpoint.pt", manifest_path=args.manifest, metadata_dir=metadata_dir,
    )


if __name__ == "__main__":
    main()
