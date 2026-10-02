"""CTC training CLI for the toy VCM acoustic model (`vcm.model`).

Trains `MatchboxNetCTC` against `out/conversions/v2/test_set/manifest.csv`'s
`train` split (or any other manifest passed via `--manifest`), with Task
02's RIR + additive-noise + SpecAugment pipeline all active by default
and time-stretch (0.85x-1.15x, pre-mel) OFF by default (`--p-timestretch`).
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from me2_voicegen.common.augment import Augmenter
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.dataset import VCMDataset, collate_fn
from me2_voicegen.vcm.perturbation_plan import plan_summary, write_plan
from me2_voicegen.vcm.model import (
    MODEL_TYPE_KEY,
    PRESETS,
    build_model,
    estimated_int8_bytes,
    model_type_for_config,
    param_count,
)
from me2_voicegen.vcm.semantic_labels import IGNORE, INTENT_CLASSES, SLOT_INTENTS, SLOTS, slot_head_name
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


DEFAULT_HEADS_LOSS_WEIGHTS = "intent=0.3,slot=0.1"


def parse_heads_loss_weights(spec: str) -> dict[str, float]:
    weights = {"intent": 0.3, "slot": 0.1}
    for part in spec.split(","):
        key, _, value = part.partition("=")
        if key.strip() not in weights:
            raise ValueError(f"--heads-loss-weights keys must be intent/slot, got {part!r}")
        weights[key.strip()] = float(value)
    return weights


def heads_ce(out, intent_id: torch.Tensor, slot_targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean cross-entropy of the intent head and, averaged over the slot heads
    that have at least one labelled row in the batch, of the slot heads. A term
    with no labelled rows is a constant 0 (CE over all-ignored rows is NaN)."""
    zero = out.intent_logits.new_zeros(())
    intent = (
        F.cross_entropy(out.intent_logits.float(), intent_id, ignore_index=IGNORE)
        if (intent_id != IGNORE).any()
        else zero
    )
    terms = [
        F.cross_entropy(out.slot_logits[slot_head_name(name)].float(), slot_targets[:, k], ignore_index=IGNORE)
        for k, name in enumerate(SLOT_INTENTS)
        if (slot_targets[:, k] != IGNORE).any()
    ]
    return {"intent": intent, "slot": torch.stack(terms).mean() if terms else zero}


def joint_loss(
    ctc: torch.Tensor, terms: dict[str, torch.Tensor], ctc_weight: float, weights: dict[str, float]
) -> torch.Tensor:
    """`ctc_weight*CTC + w_intent*CE(intent) + w_slot*mean slot CE`. At
    `ctc_weight == 0` the CTC term is left out of the graph entirely, so the
    CTC linear gets no gradient (not just a zero-scaled one: 0*inf would be NaN)."""
    loss = weights["intent"] * terms["intent"] + weights["slot"] * terms["slot"]
    return loss + ctc_weight * ctc if ctc_weight != 0 else loss


class HeadsMeter:
    """Validation accumulator for the heads: CE sums and accuracies over
    labelled rows only. `exact` = predicted intent right and, for slotted
    intents, the true intent's slot head right (rows with no slot target are
    left out of `exact`, not counted wrong)."""

    def __init__(self) -> None:
        self.intent_n = self.intent_correct = 0
        self.intent_ce = 0.0
        self.slot_n = [0] * len(SLOT_INTENTS)
        self.slot_correct = [0] * len(SLOT_INTENTS)
        self.slot_ce = [0.0] * len(SLOT_INTENTS)
        self.exact_n = self.exact_correct = 0

    @torch.no_grad()
    def add(self, out, intent_id: torch.Tensor, slot_targets: torch.Tensor) -> None:
        valid = intent_id != IGNORE
        pred = out.intent_logits.argmax(dim=-1)
        self.intent_n += int(valid.sum())
        self.intent_correct += int(((pred == intent_id) & valid).sum())
        if valid.any():
            self.intent_ce += float(F.cross_entropy(out.intent_logits.float(), intent_id, ignore_index=IGNORE, reduction="sum"))
        slotted_ids = torch.tensor([INTENT_CLASSES.index(n) for n in SLOT_INTENTS], device=intent_id.device)
        considered = valid & ~torch.isin(intent_id, slotted_ids)
        correct = (pred == intent_id) & considered
        for k, name in enumerate(SLOT_INTENTS):
            logits = out.slot_logits[slot_head_name(name)]
            target = slot_targets[:, k]
            labelled = target != IGNORE
            if not labelled.any():
                continue
            slot_pred = logits.argmax(dim=-1)
            self.slot_n[k] += int(labelled.sum())
            self.slot_correct[k] += int(((slot_pred == target) & labelled).sum())
            self.slot_ce[k] += float(F.cross_entropy(logits.float(), target, ignore_index=IGNORE, reduction="sum"))
            rows = labelled & (intent_id == INTENT_CLASSES.index(name))
            considered = considered | rows
            correct = correct | (rows & (pred == intent_id) & (slot_pred == target))
        self.exact_n += int(considered.sum())
        self.exact_correct += int(correct.sum())

    def summary(self) -> dict[str, float | int]:
        heads = [ce / n for ce, n in zip(self.slot_ce, self.slot_n) if n]
        return {
            "intent_ce": self.intent_ce / max(self.intent_n, 1),
            "slot_ce": sum(heads) / len(heads) if heads else 0.0,
            "intent_acc": self.intent_correct / max(self.intent_n, 1),
            "slot_acc": sum(self.slot_correct) / max(sum(self.slot_n), 1),
            "exact_acc": self.exact_correct / max(self.exact_n, 1),
            "n_intent": self.intent_n,
            "n_slot": sum(self.slot_n),
            "n_exact": self.exact_n,
        }


def build_train_collate(
    feature_extractor: LogMelFeatureExtractor, augmenter: Augmenter | None, with_labels: bool = False
):
    def _collate(batch):
        out = collate_fn(batch, feature_extractor, with_labels=with_labels)
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
    heads_meter: HeadsMeter | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_len = batch["target_len"].to(device)

        if heads_meter is not None:
            out = model.forward_heads(features, input_lengths)
            logits = out.ctc_logits
            heads_meter.add(out, batch["intent_id"].to(device), batch["slot_targets"].to(device))
        else:
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
    parser.add_argument("--p-timestretch", type=float, default=0.0)
    parser.add_argument(
        "--noise-source",
        choices=("manifest", "dataset"),
        default="manifest",
        help="additive-noise pool: the manifest's own background_noise rows (default), or 'dataset': this "
        "manifest's noise-only rows plus deterministic synthetic coloured noise, nothing external",
    )
    parser.add_argument(
        "--perturbation-plan",
        choices=("online", "table"),
        default="online",
        help="'online' (default): per-clip coin flips from the augmenter's generator. 'table': a reproducible, "
        "class-stratified recipe per (clip, epoch) from vcm.perturbation_plan, a pure function of the seed; the "
        "waveform probabilities (--p-rir/--p-noise/--p-babble/--p-timestretch) become per-class quotas",
    )
    parser.add_argument(
        "--dump-plan",
        action="store_true",
        help="with --perturbation-plan table: write each epoch's recipes (gzipped CSV) and a per-class summary "
        "under <out-dir>/metadata/perturbation_plan/",
    )
    parser.add_argument(
        "--p-babble",
        type=float,
        default=0.0,
        help="probability of mixing one of the train split's babble rows in as quiet background speech "
        "(12-25 dB SNR); needs --noise-source dataset to have a pool",
    )
    parser.add_argument("--time-check-every", type=int, default=50, help="check the wall-clock cap every N train batches")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=None,
        help="heads presets only (default 1.0). 0 = classifier-only: the CTC head gets no gradient "
        "and the checkpoint is chosen on the heads' validation loss instead of CTC val loss",
    )
    parser.add_argument(
        "--heads-loss-weights",
        default=None,
        help=f"heads presets only (default {DEFAULT_HEADS_LOSS_WEIGHTS!r}): weights of the intent CE and the mean slot CE",
    )
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

    use_heads = bool(getattr(PRESETS[args.preset], "heads", False))
    if not use_heads and (args.ctc_weight is not None or args.heads_loss_weights is not None):
        raise SystemExit("--ctc-weight/--heads-loss-weights only apply to a heads preset (quartznet5x3-heads)")
    ctc_weight = 1.0 if args.ctc_weight is None else args.ctc_weight
    heads_weights = parse_heads_loss_weights(args.heads_loss_weights or DEFAULT_HEADS_LOSS_WEIGHTS)

    device = torch.device(args.device) if args.device else torch.device("cpu")
    use_amp = args.amp if args.amp is not None else device.type == "cuda"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out_dir / "checkpoints"
    metadata_dir = args.out_dir / "metadata"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = LogMelFeatureExtractor()
    table = args.perturbation_plan == "table"
    # In table mode the waveform steps come from the recipe table, so the augmenter keeps only SpecAugment.
    train_augmenter = Augmenter(
        p_rir=0.0 if table else args.p_rir,
        p_noise=0.0 if table else args.p_noise,
        p_specaugment=args.p_specaugment,
        p_timestretch=0.0 if table else args.p_timestretch,
        p_babble=0.0 if table else args.p_babble,
        seed=args.seed,
    )

    train_dataset = VCMDataset(
        args.manifest, split="train", augmenter=train_augmenter, semantic_labels=use_heads,
        noise_source=args.noise_source,
    )
    if table:
        train_dataset.enable_perturbation_plan(
            train_dataset.plan_config(args.p_timestretch, args.p_rir, args.p_noise, args.p_babble), seed=args.seed
        )
        plan_dir = args.out_dir / "metadata" / "perturbation_plan"
        if args.dump_plan:
            plan_dir.mkdir(parents=True, exist_ok=True)
    if args.noise_source != "manifest" or args.p_babble > 0:
        print(
            f"noise: source={args.noise_source} p_noise={args.p_noise} pool={len(train_dataset._noise_pool())} "
            f"p_babble={args.p_babble} babble_pool={len(train_dataset._babble_pool())}"
        )
    val_dataset = VCMDataset(args.manifest, split="val", augmenter=None, semantic_labels=use_heads)
    if use_heads:
        print(f"head label sources (loss-bearing rows) train={dict(train_dataset.semantic_label_counts)} "
              f"val={dict(val_dataset.semantic_label_counts)}")

    train_subset = Subset(train_dataset, train_dataset.loss_bearing_indices)
    val_subset = Subset(val_dataset, val_dataset.loss_bearing_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=build_train_collate(feature_extractor, train_augmenter, with_labels=use_heads),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=build_train_collate(feature_extractor, None, with_labels=use_heads),
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

    def heads_select_loss(summary: dict) -> float:
        return heads_weights["intent"] * summary["intent_ce"] + heads_weights["slot"] * summary["slot_ce"]

    epoch0_val_stats: dict = {}
    epoch0_meter = HeadsMeter() if use_heads else None
    epoch0_val_loss = run_eval(model, val_loader, criterion, device, epoch0_val_stats, epoch0_meter)
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

        if table:
            plan = train_dataset.set_epoch(epoch)
            if args.dump_plan:
                write_plan(plan_dir / f"epoch_{epoch:03d}.csv.gz", plan, train_dataset.rows, epoch)
                if epoch == 1:
                    (plan_dir / "summary_epoch_001.json").write_text(json.dumps(plan_summary(plan, train_dataset.rows), indent=2))

        model.train()
        train_loss_total = 0.0
        train_examples = 0
        train_term_totals: dict[str, float] = {}
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
                if use_heads:
                    heads_out = model.forward_heads(features, input_lengths)
                    logits = heads_out.ctc_logits
                else:
                    logits = model(features)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, target_ids, output_lengths, target_len)
                if use_heads:
                    ctc_term = loss
                    terms = heads_ce(heads_out, batch["intent_id"].to(device), batch["slot_targets"].to(device))
                    loss = joint_loss(ctc_term, terms, ctc_weight, heads_weights)

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
            if use_heads:
                for name, value in (("ctc", ctc_term), ("intent", terms["intent"]), ("slot", terms["slot"])):
                    train_term_totals[name] = train_term_totals.get(name, 0.0) + float(value.item()) * n

        if deadline_hit:
            break

        train_loss = train_loss_total / max(train_examples, 1)
        val_stats: dict = {}
        meter = HeadsMeter() if use_heads else None
        val_loss = run_eval(model, val_loader, criterion, device, val_stats, meter)
        val_heads = meter.summary() if meter is not None else None
        # Checkpoint/early-stop on CTC val loss (as every other run); a classifier-only run
        # (ctc_weight 0) never trains its CTC head, so it is selected on the heads' own val loss.
        select_loss = heads_select_loss(val_heads) if use_heads and ctc_weight == 0 else val_loss
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
        if val_heads is not None:
            print(
                f"  heads val: intent_acc={val_heads['intent_acc']:.4f} slot_acc={val_heads['slot_acc']:.4f} "
                f"exact_acc={val_heads['exact_acc']:.4f} intent_ce={val_heads['intent_ce']:.4f} slot_ce={val_heads['slot_ce']:.4f}"
            )
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
                **(
                    {
                        "ctc_loss_train": train_term_totals["ctc"] / max(train_examples, 1),
                        "intent_loss_train": train_term_totals["intent"] / max(train_examples, 1),
                        "slot_loss_train": train_term_totals["slot"] / max(train_examples, 1),
                        "ctc_loss_val": val_loss,
                        "intent_loss_val": val_heads["intent_ce"],
                        "slot_loss_val": val_heads["slot_ce"],
                        "intent_acc_val": val_heads["intent_acc"],
                        "slot_acc_val": val_heads["slot_acc"],
                        "exact_acc_val": val_heads["exact_acc"],
                        "heads_val_counts": {k: val_heads[k] for k in ("n_intent", "n_slot", "n_exact")},
                        "select_loss": select_loss,
                    }
                    if val_heads is not None
                    else {}
                ),
            }
        )

        if select_loss < best_val_loss:
            best_val_loss = select_loss
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
        **({"perturbation_plan": "table"} if table else {}),
        **(
            {"noise_source": args.noise_source, "p_babble": args.p_babble}
            if args.noise_source != "manifest" or args.p_babble > 0
            else {}
        ),
        **(
            {
                "heads": True,
                "ctc_weight": ctc_weight,
                "heads_loss_weights": heads_weights,
                "selected_on": "ctc_val_loss" if ctc_weight != 0 else "heads_val_loss",
                "semantic_label_counts": {
                    "train": dict(train_dataset.semantic_label_counts),
                    "val": dict(val_dataset.semantic_label_counts),
                },
            }
            if use_heads
            else {}
        ),
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
