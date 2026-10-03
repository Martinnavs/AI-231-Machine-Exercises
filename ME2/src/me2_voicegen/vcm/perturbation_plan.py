"""Reproducible, class-stratified perturbation recipes for training (feature ctc-attention).

The online augmenter flips coins per clip from a generator that every DataLoader worker holds its own copy of, so
two runs with the same seed do not perturb the same clips the same way. Here the perturbation of every training
clip in every epoch is instead a row of a table, a pure function of `(seed, epoch, class)`: independent of workers,
shuffle order and thread count, and dumpable as a CSV to audit exactly what each clip saw.

Stratification: clips are grouped by `class_key` (intent+slot, plus variation and source for ai231-style manifests). Inside a class each perturbation type gets an
exact quota (`p * n`, randomised rounding so the expectation is `p`) assigned to a random subset of the class, so every
class receives the intended share of reverb / noise / babble / stretch instead of a share left to chance. Continuous
parameters (stretch factor, SNRs) are Latin-hypercube spread across the clips that received them, and pool indices
(RIR, noise, babble clip) are dealt from a shuffled deck so pool entries are used evenly.
"""

from __future__ import annotations

import csv
import gzip
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from me2_voicegen.common.augment import BABBLE_SNR_MAX_DB, BABBLE_SNR_MIN_DB, SNR_MAX_DB, SNR_MIN_DB, Recipe
from me2_voicegen.common.augment import TSTRETCH_FACTOR_MAX, TSTRETCH_FACTOR_MIN

PLAN_FIELDS = ["epoch", "row", "filename", "class", "stretch", "rir", "noise", "noise_snr_db", "babble", "babble_snr_db", "noise_offset", "babble_offset"]


@dataclass(frozen=True)
class PlanConfig:
    p_stretch: float = 0.0
    p_rir: float = 0.0
    p_noise: float = 0.0
    p_babble: float = 0.0
    n_rir: int = 0
    n_noise: int = 0
    n_babble: int = 0
    stretch_range: tuple[float, float] = (TSTRETCH_FACTOR_MIN, TSTRETCH_FACTOR_MAX)
    noise_snr_range: tuple[float, float] = (SNR_MIN_DB, SNR_MAX_DB)
    babble_snr_range: tuple[float, float] = (BABBLE_SNR_MIN_DB, BABBLE_SNR_MAX_DB)
    skip_prenoised: bool = False  # no added noise / babble on clips that already carry noise (`is_prenoised`)
    random_offsets: bool = False  # seeded start offset into each noise / babble clip (separate RNG stream)

    def __post_init__(self) -> None:
        for name, p, n in (("rir", self.p_rir, self.n_rir), ("noise", self.p_noise, self.n_noise), ("babble", self.p_babble, self.n_babble)):
            if p > 0 and n < 1:
                raise ValueError(f"p_{name}={p} but the {name} pool is empty")


def class_key(row: Mapping[str, str]) -> str:
    """Stratum of a manifest row. Manifests with a `variation` column (ai231 and later) group by
    `label | slot_value | variation | source_dataset`, e.g. `TIMER|10 seconds|Timer 10 seconds|optionb`, so every one of the
    93 variations, and real vs persona clips within it, get the same perturbation shares; out-of-scope rows have an empty
    variation and fall into one stratum per source. Older manifests (no `variation` column) keep the intent+slot class,
    e.g. `ALARM|6 am`, so runs on them reproduce as before."""
    base = f"{row['label']}|{(row.get('slot_value') or '').lower()}"
    if "variation" not in row:
        return base
    return f"{base}|{row.get('variation') or ''}|{row.get('source_dataset') or ''}"


def is_prenoised(row: Mapping[str, str]) -> bool:
    """Clips that already contain noise: ai231 group-synthetic `*_noisy.wav` (named in `source_relpath`) and rows mixed
    offline by the ambient overlay (`*_ambient` sources)."""
    name = f"{row.get('source_relpath', '')} {row.get('filename', '')}"
    return "_noisy" in name or (row.get("source_dataset") or "").endswith("_ambient")


def _quota(rng: np.random.Generator, n: int, p: float) -> int:
    """`p * n` clips with randomised rounding (so the long-run share is exactly `p`)."""
    exact = p * n
    base = int(np.floor(exact))
    return min(n, base + int(rng.random() < exact - base))


def _spread(rng: np.random.Generator, k: int, lo: float, hi: float) -> np.ndarray:
    """k values covering [lo, hi] evenly (Latin hypercube: one per 1/k-wide stratum), in random order."""
    u = (np.arange(k) + rng.random(k)) / max(k, 1)
    rng.shuffle(u)
    return lo + u * (hi - lo)


def _deal(rng: np.random.Generator, k: int, pool_size: int) -> list[int]:
    """k pool indices dealt from a shuffled deck, reshuffled when it runs out: even use of every pool entry."""
    out: list[int] = []
    while len(out) < k:
        out += [int(x) for x in rng.permutation(pool_size)]
    return out[:k]


def build_epoch_plan(
    rows: Sequence[Mapping[str, str]],
    indices: Sequence[int],
    cfg: PlanConfig,
    seed: int,
    epoch: int,
) -> dict[int, Recipe]:
    """Recipes for `indices` (the rows actually trained on, i.e. the quotas are over those) in `epoch`."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i in sorted(indices):
        groups[class_key(rows[i])].append(i)
    plan: dict[int, Recipe] = {i: Recipe() for i in indices}
    for key, members in sorted(groups.items()):
        rng = np.random.default_rng(zlib.crc32(f"{seed}:{epoch}:{key}".encode("utf-8")))
        n = len(members)
        fields: dict[int, dict] = {i: {} for i in members}

        def pick(p: float, pool: Sequence[int] = members) -> list[int]:
            return [pool[j] for j in rng.permutation(len(pool))[: _quota(rng, len(pool), p)]]

        # with skip_prenoised the noise / babble quotas are over the clean clips only; otherwise `eligible` is `members`
        # itself and the RNG stream (so every recipe) is identical to the plain table
        eligible = [i for i in members if not is_prenoised(rows[i])] if cfg.skip_prenoised else members

        chosen = pick(cfg.p_stretch)
        for i, f in zip(chosen, _spread(rng, len(chosen), *cfg.stretch_range)):
            fields[i]["stretch"] = float(f)
        chosen = pick(cfg.p_rir)
        for i, r in zip(chosen, _deal(rng, len(chosen), cfg.n_rir)):
            fields[i]["rir"] = r
        chosen = pick(cfg.p_noise, eligible)
        for i, j, snr in zip(chosen, _deal(rng, len(chosen), cfg.n_noise), _spread(rng, len(chosen), *cfg.noise_snr_range)):
            fields[i].update(noise=j, noise_snr_db=float(snr))
        chosen = pick(cfg.p_babble, eligible)
        for i, j, snr in zip(chosen, _deal(rng, len(chosen), cfg.n_babble), _spread(rng, len(chosen), *cfg.babble_snr_range)):
            fields[i].update(babble=j, babble_snr_db=float(snr))
        if cfg.random_offsets:  # own stream, so all other fields stay as without offsets
            orng = np.random.default_rng(zlib.crc32(f"{seed}:{epoch}:{key}:offsets".encode("utf-8")))
            for i in members:
                if "noise" in fields[i]:
                    fields[i]["noise_offset"] = float(orng.random())
                if "babble" in fields[i]:
                    fields[i]["babble_offset"] = float(orng.random())
        for i in members:
            plan[i] = Recipe(**fields[i])
    return plan


def plan_rows(plan: Mapping[int, Recipe], rows: Sequence[Mapping[str, str]], epoch: int) -> list[dict]:
    return [
        {
            "epoch": epoch, "row": i, "filename": rows[i].get("filename", ""), "class": class_key(rows[i]),
            "stretch": r.stretch, "rir": r.rir, "noise": r.noise, "noise_snr_db": r.noise_snr_db,
            "babble": r.babble, "babble_snr_db": r.babble_snr_db,
            "noise_offset": r.noise_offset, "babble_offset": r.babble_offset,
        }
        for i, r in sorted(plan.items())
    ]


def write_plan(path: Path, plan: Mapping[int, Recipe], rows: Sequence[Mapping[str, str]], epoch: int) -> None:
    """Gzipped CSV, one line per trained clip; empty cell = step not applied."""
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        for r in plan_rows(plan, rows, epoch):
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def plan_summary(plan: Mapping[int, Recipe], rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, int]]:
    """Per class: clips, and how many received each perturbation (and none at all)."""
    out: dict[str, dict[str, int]] = {}
    for i, r in plan.items():
        s = out.setdefault(class_key(rows[i]), {"n": 0, "stretch": 0, "rir": 0, "noise": 0, "babble": 0, "unperturbed": 0})
        s["n"] += 1
        s["stretch"] += r.stretch is not None
        s["rir"] += r.rir is not None
        s["noise"] += r.noise is not None
        s["babble"] += r.babble is not None
        s["unperturbed"] += r == Recipe()
    return dict(sorted(out.items()))
