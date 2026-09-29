#!/usr/bin/env python
"""Dump VCM log-posteriors (CPU) for the dense-phonetic-scoring pilot.

Experiment-only (`.scratch/dense-phonetic-scoring-pilot`); never imported by
production. For each (split, condition) writes, under `--out-dir`:

  logits/<split>_<condition>.npz   `logp` float32 (sum T, 29), `offsets` int64 (N+1)
  logits/<split>_<condition>.rows.csv  index, filename, bucket, label, group_id, source_dataset
  logits/<split>_<condition>.meta.json checkpoint/manifest/versions/shas

`condition` is `clean` or `noisy_s0`. Noisy mode applies the fixed-seed
perturbation from `vcm/noisy_eval.py`, which exists only in the
`me2-iteration3` worktree; run it with
`PYTHONPATH=<iteration3>/ME2/src`. Row order = manifest order restricted to
target_commands/babble/silence buckets (background_noise rows feed the noise
pool but are never decoded); `index` is the dataset index, so perturbation
draws stay aligned with `noisy_eval.decode_split_noisy`.

`--fixture-out DIR` extracts a small real val-only subset from existing
dumps (see `select_fixture_rows`). Checkpoints and fixtures derived from
ESC-50 inherit CC-BY-NC-SA-4.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

KEEP_BUCKETS = ("target_commands", "babble", "silence")
CONDITIONS = ("clean", "noisy_s0")
NOISY_SEED = 0
NOISY_ENV_HINT = (
    "noisy_s0 needs vcm/noisy_eval.py, which lives only in the me2-iteration3 "
    "worktree: run with PYTHONPATH=<repo>/me2-iteration3/ME2/src"
)
ROWS_FIELDS = ["index", "filename", "bucket", "label", "group_id", "source_dataset"]


def _import_noisy_eval():
    try:
        from me2_voicegen.vcm import noisy_eval
    except ImportError as exc:
        raise SystemExit(f"{NOISY_ENV_HINT} ({exc})") from exc
    return noisy_eval


def _kept_indices(dataset) -> list[int]:
    return [i for i, r in enumerate(dataset.rows) if r["bucket"] in KEEP_BUCKETS]


def _shard_worker(args) -> list[tuple[int, np.ndarray]]:
    checkpoint, manifest, split, condition, indices = args
    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.dataset import VCMDataset
    from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform

    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint, device="cpu")
    extractor = LogMelFeatureExtractor()
    dataset = VCMDataset(manifest, split=split, augmenter=None)
    perturbations = noise_pool = rir_pool = noisy_eval = None
    if condition == "noisy_s0":
        noisy_eval = _import_noisy_eval()
        rir_pool = noisy_eval.build_rir_pool_for_seed(NOISY_SEED)
        perturbations = noisy_eval.build_perturbations(dataset, rir_pool, NOISY_SEED)
        noise_pool = dataset._noise_pool()
    out = []
    for i in indices:
        waveform = dataset[i].waveform
        if noisy_eval is not None:
            rir_i, noise_i, snr_db = perturbations[i]
            waveform = noisy_eval.apply_perturbation(
                waveform, rir_pool[rir_i], noise_pool[noise_i], snr_db
            )
        out.append((i, logp_for_waveform(model, extractor, waveform, device="cpu")))
    return out


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_split(
    checkpoint: Path,
    manifest: Path,
    split: str,
    condition: str,
    out_dir: Path,
    workers: int = 32,
) -> dict:
    """Compute and write one (split, condition) dump. Returns the meta dict."""
    from me2_voicegen.vcm.dataset import VCMDataset

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if condition == "noisy_s0":
        _import_noisy_eval()
    dataset = VCMDataset(manifest, split=split, augmenter=None)
    kept = _kept_indices(dataset)
    n_shards = max(1, min(workers, len(kept)))
    shards = [kept[k::n_shards] for k in range(n_shards)]
    jobs = [(str(checkpoint), str(manifest), split, condition, s) for s in shards]
    if n_shards == 1:
        parts = [_shard_worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=n_shards) as pool:
            parts = list(pool.map(_shard_worker, jobs))
    by_index = {i: lp for part in parts for i, lp in part}
    logps = [by_index[i].astype(np.float32) for i in kept]
    offsets = np.zeros(len(kept) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([lp.shape[0] for lp in logps])

    log_dir = out_dir / "logits"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{condition}"
    np.savez_compressed(log_dir / f"{stem}.npz", logp=np.concatenate(logps), offsets=offsets)
    with (log_dir / f"{stem}.rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROWS_FIELDS)
        writer.writeheader()
        for i in kept:
            r = dataset.rows[i]
            writer.writerow(
                {
                    "index": i,
                    "filename": r.get("path", ""),
                    "bucket": r["bucket"],
                    "label": r["label"],
                    "group_id": r.get("group_id") or "",
                    "source_dataset": r.get("source_dataset") or "",
                }
            )
    repo = Path(__file__).resolve().parents[1]
    meta = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "split": split,
        "condition": condition,
        "seed": NOISY_SEED if condition == "noisy_s0" else None,
        "n_rows": len(kept),
        "torch": torch.__version__,
        "git_sha_main": _git_sha(repo),
    }
    if condition == "noisy_s0":
        ne = sys.modules["me2_voicegen.vcm.noisy_eval"]
        meta["noisy_eval_path"] = ne.__file__
        meta["noisy_eval_sha256"] = _sha256(Path(ne.__file__))
        meta["git_sha_noisy_tree"] = _git_sha(Path(ne.__file__).resolve().parents[4])
    (log_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_dump(out_dir: Path, split: str, condition: str):
    """-> (list of (T, 29) logp arrays, list of rows dicts)."""
    stem = out_dir / "logits" / f"{split}_{condition}"
    z = np.load(f"{stem}.npz")
    logp, offsets = z["logp"], z["offsets"]
    with open(f"{stem}.rows.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [logp[offsets[k]:offsets[k + 1]] for k in range(len(rows))], rows


def select_fixture_rows(rows: list[dict], falsely_accepted: set[int]) -> list[int]:
    """Positions (into `rows`) for the small val fixture: 2 STOP + 2 TIME
    targets, 4 other-intent targets, 4 babble, 4 silence, and >= 2
    reject-probe rows in `falsely_accepted` (positions the unchanged
    decoder accepts at -0.1). Deterministic: first matches in row order."""
    chosen: list[int] = []

    def take(pred, n):
        got = 0
        for pos, r in enumerate(rows):
            if got >= n:
                break
            if pos not in chosen and pred(pos, r):
                chosen.append(pos)
                got += 1

    tgt = lambda r: r["bucket"] == "target_commands"
    take(lambda p, r: tgt(r) and r["label"] == "STOP", 2)
    take(lambda p, r: tgt(r) and r["label"] == "TIME", 2)
    take(lambda p, r: tgt(r) and r["label"] not in ("STOP", "TIME"), 4)
    take(lambda p, r: r["bucket"] == "babble" and p in falsely_accepted, 2)
    take(lambda p, r: r["bucket"] == "silence" and p in falsely_accepted, 1)
    take(lambda p, r: r["bucket"] == "babble", 4)
    take(lambda p, r: r["bucket"] == "silence", 4)
    return sorted(chosen)


def write_fixture(out_dir: Path, fixture_dir: Path, threshold: float = -0.1) -> dict:
    """Extract the val fixture (clean + noisy_s0) from existing full dumps."""
    from me2_voicegen.vcm.decoder import decode_utterance
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

    fixture_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for condition in CONDITIONS:
        logps, rows = load_dump(out_dir, "val", condition)
        accepted = set()
        for pos, (lp, r) in enumerate(zip(logps, rows)):
            if r["bucket"] == "target_commands":
                continue
            d = decode_utterance(lp, OPTIONB_GRAMMAR, threshold, beam_width=50)
            if not d.no_match:
                accepted.add(pos)
        picks = select_fixture_rows(rows, accepted)
        sel = [logps[p] for p in picks]
        offsets = np.zeros(len(sel) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([s.shape[0] for s in sel])
        np.savez_compressed(
            fixture_dir / f"val_{condition}.npz", logp=np.concatenate(sel), offsets=offsets
        )
        with (fixture_dir / f"val_{condition}.rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ROWS_FIELDS)
            w.writeheader()
            w.writerows(rows[p] for p in picks)
        summary[condition] = {
            "n_rows": len(picks),
            "n_falsely_accepted": sum(1 for p in picks if p in accepted),
        }
    (fixture_dir / "README.txt").write_text(
        "Real val-row VCM log-posteriors (checkpoint option-d-fil50-ambient-rir-135m) for "
        "tests/test_vcm_dense_pilot.py. Produced by scripts/dense_pilot_dump.py "
        "--fixture-out. Derived from a model trained on ESC-50-derived data: "
        "CC-BY-NC-SA-4.0.\n"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", default="val,test")
    p.add_argument("--conditions", default="clean,noisy_s0")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--fixture-out", type=Path, default=None,
                   help="Extract the val fixture from existing dumps in --out-dir and exit.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.fixture_out is not None:
        print(json.dumps(write_fixture(args.out_dir, args.fixture_out), indent=2))
        return
    if args.checkpoint is None or args.manifest is None:
        raise SystemExit("--checkpoint and --manifest are required for a dump")
    for split in args.splits.split(","):
        for condition in args.conditions.split(","):
            meta = dump_split(
                args.checkpoint, args.manifest, split, condition, args.out_dir, args.workers
            )
            print(f"{split}/{condition}: {meta['n_rows']} rows", flush=True)


if __name__ == "__main__":
    main()
