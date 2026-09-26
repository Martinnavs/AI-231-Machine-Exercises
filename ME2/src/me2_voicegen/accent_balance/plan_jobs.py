"""T2 (.scratch/accent-balance-fil50/tickets/00-RECAP.md): compute the
per-split Filipino deficit for VCM + wakeword and emit a synthesis job list.

`FILIPINO_REFERENCE_SPEAKER_IDS` (VCM) and the `ref_voice` regex (wakeword)
are the same "what counts as Filipino today" definitions the eval code and
the RECAP ticket use -- kept here as local copies (not imported) because
`vcm.evaluate` is intentionally a heavier import (torch-adjacent modules)
this stage doesn't otherwise need; the two definitions are the ticket's own
documented source of truth, not derived independently.

Job text:
  - VCM: each new job copies (text, label) from a same-split NON-Filipino
    target-command row, sampled with replacement -- the synthetic Filipino
    half then mirrors the real intent/phrasing distribution instead of
    inventing its own.
  - wakeword: every job is the literal wakeword utterance, "Computer.".

Voice pool (`--voice-sources`):
  - `all` (default): each split's own voices.csv rows, unchanged.
  - `references`: every ref_ voice in the CSV for EVERY split, ignoring the
    split column (references are train-only in the CSV, but Phase 1 uses
    them in all splits -- a justified cross-split timbre exception, since
    references timbre already spans every split via converted wakeword
    positives; the split-disjointness checks carry a matching scoped
    `ref_` exemption). fsc_ voices excluded.
  - `sapinsapin`: only the split's own fsc_ voices. ref_ excluded.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from random import Random

logger = logging.getLogger(__name__)

FILIPINO_VCM_SPEAKER_IDS: frozenset[str] = frozenset({f"s{n}" for n in range(68, 81)} | {"s89", "s90", "s100"})
FILIPINO_WW_REF_RE = re.compile(r"^(tagalog|ilonggo)")

JOBS_FIELDS = ["job_id", "model", "split", "voice_id", "text", "label", "source_row_ref", "noisy_target"]
SPLITS = ("train", "val", "test")


def _is_filipino_vcm(row: dict) -> bool:
    return row["source_dataset"] == "optionb" and row["group_id"] in FILIPINO_VCM_SPEAKER_IDS


def _is_filipino_ww(row: dict) -> bool:
    return bool(FILIPINO_WW_REF_RE.match(row.get("ref_voice") or ""))


def load_voices(voices_csv: Path) -> dict[str, list[dict]]:
    """split -> list of voice rows (from build_refs.py's voices.csv)."""
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    with voices_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] not in by_split:
                raise ValueError(f"{voices_csv}: unknown split {row['split']!r} for voice {row['voice_id']!r}")
            by_split[row["split"]].append(row)
    for split, voices in by_split.items():
        if not voices:
            raise ValueError(f"{voices_csv}: no voices assigned to split {split!r}")
    return by_split


VOICE_SOURCES = ("all", "references", "sapinsapin")


def voice_pool_for_split(voices_by_split: dict[str, list[dict]], split: str, voice_sources: str) -> list[dict]:
    """The voice pool for `split` under the given `--voice-sources` mode
    (see the module docstring): `all` is the split's own rows;
    `references` is every ref_ voice in the CSV regardless of its split
    column; `sapinsapin` is the split's own fsc_ voices only."""
    if voice_sources == "all":
        return voices_by_split[split]
    if voice_sources == "references":
        return [v for vs in voices_by_split.values() for v in vs if v["voice_id"].startswith("ref_")]
    return [v for v in voices_by_split[split] if v["voice_id"].startswith("fsc_")]


def count_vcm_deficit(manifest_path: Path) -> dict[str, dict[str, int]]:
    """split -> {non_filipino, filipino, deficit} for bucket==target_commands."""
    counts = {s: {"non_filipino": 0, "filipino": 0} for s in SPLITS}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "target_commands":
                continue
            split = row["split"]
            if split not in counts:
                continue
            counts[split]["filipino" if _is_filipino_vcm(row) else "non_filipino"] += 1
    for split, c in counts.items():
        c["deficit"] = max(0, c["non_filipino"] - c["filipino"])
    return counts


def count_ww_deficit(manifest_path: Path) -> dict[str, dict[str, int]]:
    counts = {s: {"non_filipino": 0, "filipino": 0} for s in SPLITS}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] != "_wakeword_":
                continue
            split = row["split"]
            if split not in counts:
                continue
            counts[split]["filipino" if _is_filipino_ww(row) else "non_filipino"] += 1
    for split, c in counts.items():
        c["deficit"] = max(0, c["non_filipino"] - c["filipino"])
    return counts


def load_vcm_non_filipino_rows(manifest_path: Path) -> dict[str, list[dict]]:
    """split -> non-Filipino target_commands rows (source of text/label to
    mirror), and the split's noisy-row fraction (filename contains
    `_noisy`)."""
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "target_commands" or row["split"] not in by_split:
                continue
            if not _is_filipino_vcm(row):
                by_split[row["split"]].append(row)
    return by_split


def noisy_fraction_vcm(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if "_noisy" in r["filename"]) / len(rows)


def noisy_fraction_ww(manifest_path: Path) -> dict[str, float]:
    by_split: dict[str, list[bool]] = {s: [] for s in SPLITS}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] != "_wakeword_" or row["split"] not in by_split or _is_filipino_ww(row):
                continue
            by_split[row["split"]].append(row["source_dataset"].endswith("_noisy"))
    return {s: (sum(v) / len(v) if v else 0.0) for s, v in by_split.items()}


def build_jobs(
    *,
    model: str,
    split: str,
    n_jobs: int,
    voices: list[dict],
    rng: Random,
    text_source: list[dict] | None,
    noisy_fraction: float,
    job_id_start: int,
) -> list[dict]:
    if n_jobs <= 0:
        return []
    if not voices:
        raise ValueError(f"{model}/{split}: no voices available to plan {n_jobs} jobs")

    voice_cycle = list(voices)
    rng.shuffle(voice_cycle)

    jobs: list[dict] = []
    for i in range(n_jobs):
        voice = voice_cycle[i % len(voice_cycle)]
        if model == "vcm":
            if not text_source:
                raise ValueError(f"vcm/{split}: no non-Filipino rows to sample text from")
            src = rng.choice(text_source)
            text, label, ref = src["transcript"], src["label"], src["filename"]
        else:
            text, label, ref = "Computer.", "_wakeword_", ""
        jobs.append(
            {
                "job_id": f"{model}_{split}_{job_id_start + i:06d}",
                "model": model,
                "split": split,
                "voice_id": voice["voice_id"],
                "text": text,
                "label": label,
                "source_row_ref": ref,
                "noisy_target": "1" if rng.random() < noisy_fraction else "0",
            }
        )
    return jobs


def plan(
    *,
    voices_csv: Path,
    vcm_manifest: Path,
    wakeword_manifest: Path,
    overgen: float,
    seed: int,
    pilot: int | None,
    voice_sources: str = "all",
) -> tuple[list[dict], dict]:
    if voice_sources not in VOICE_SOURCES:
        raise ValueError(f"unknown voice_sources {voice_sources!r}, expected one of {VOICE_SOURCES}")
    rng = Random(seed)
    voices_by_split = load_voices(voices_csv)

    vcm_counts = count_vcm_deficit(vcm_manifest)
    ww_counts = count_ww_deficit(wakeword_manifest)
    vcm_text_rows = load_vcm_non_filipino_rows(vcm_manifest)
    ww_noisy_frac = noisy_fraction_ww(wakeword_manifest)

    jobs: list[dict] = []
    stats = {"vcm": vcm_counts, "wakeword": ww_counts}

    if pilot is not None:
        # Pilot mode: a small, fixed count per model, drawn only from each
        # split's real voice pool but capped so at least 12 voices get
        # exercised in total, split evenly across prompt_source. Uses the
        # train split's voice pool for every job (T2 spec: "train-pool
        # voices only, split evenly between prompt_source values, across at
        # least 12 voices"), and doesn't touch val/test manifests at all.
        # The pool honors `--voice-sources` the same way the non-pilot path
        # does (for `references`, that's every ref_ voice, not just the
        # train-split rows).
        train_voices = voice_pool_for_split(voices_by_split, "train", voice_sources)
        by_source: dict[str, list[dict]] = {}
        for v in train_voices:
            by_source.setdefault(v["prompt_source"], []).append(v)
        min_voices = min(12, len(train_voices))
        if min_voices < 2:
            raise ValueError("not enough train-split voices to run a pilot")

        for model in ("vcm", "wakeword"):
            n = pilot
            sources = sorted(by_source)
            per_source = max(1, n // len(sources))
            picked: list[dict] = []
            for source in sources:
                pool = list(by_source[source])
                rng.shuffle(pool)
                take = pool[: max(1, min(per_source, len(pool)))]
                picked.extend(take)
            if len(picked) > n:
                picked = picked[:n]
            text_source = vcm_text_rows["train"] if model == "vcm" else None
            noisy_frac = noisy_fraction_vcm(vcm_text_rows["train"]) if model == "vcm" else ww_noisy_frac["train"]
            model_jobs = build_jobs(
                model=model,
                split="train",
                n_jobs=n,
                voices=picked,
                rng=rng,
                text_source=text_source,
                noisy_fraction=noisy_frac,
                job_id_start=0,
            )
            jobs.extend(model_jobs)
        return jobs, stats

    for model, counts, text_rows_by_split, noisy_frac_by_split in (
        ("vcm", vcm_counts, vcm_text_rows, {s: noisy_fraction_vcm(vcm_text_rows[s]) for s in SPLITS}),
        ("wakeword", ww_counts, None, ww_noisy_frac),
    ):
        job_id_counter = 0
        for split in SPLITS:
            deficit = counts[split]["deficit"]
            n_jobs = int(deficit * overgen + 0.999) if deficit else 0
            text_source = text_rows_by_split[split] if text_rows_by_split is not None else None
            model_jobs = build_jobs(
                model=model,
                split=split,
                n_jobs=n_jobs,
                voices=voice_pool_for_split(voices_by_split, split, voice_sources),
                rng=rng,
                text_source=text_source,
                noisy_fraction=noisy_frac_by_split[split],
                job_id_start=job_id_counter,
            )
            job_id_counter += len(model_jobs)
            jobs.extend(model_jobs)

    return jobs, stats


def write_jobs_csv(out_path: Path, jobs: list[dict]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOBS_FIELDS)
        writer.writeheader()
        writer.writerows(jobs)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices", type=Path, required=True, help="build_refs.py's voices.csv")
    parser.add_argument("--vcm-manifest", type=Path, required=True)
    parser.add_argument("--wakeword-manifest", type=Path, required=True)
    parser.add_argument("--overgen", type=float, default=1.15)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pilot", type=int, default=None, help="plan N jobs per model from the train voice pool only, ignore the real deficit")
    parser.add_argument("--voice-sources", choices=VOICE_SOURCES, default="all",
                        help="voice pool per split: 'all' (default, today's behavior), "
                        "'references' (every ref_ voice in every split, split column ignored), "
                        "'sapinsapin' (only the split's own fsc_ voices)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    jobs, stats = plan(
        voices_csv=args.voices,
        vcm_manifest=args.vcm_manifest,
        wakeword_manifest=args.wakeword_manifest,
        overgen=args.overgen,
        seed=args.seed,
        pilot=args.pilot,
        voice_sources=args.voice_sources,
    )
    dest = write_jobs_csv(args.out, jobs)

    for model, counts in stats.items():
        print(f"{model} deficit by split:")
        for split in SPLITS:
            c = counts[split]
            print(f"  {split}: non_filipino={c['non_filipino']} filipino={c['filipino']} deficit={c['deficit']}")
    by_model = {}
    for j in jobs:
        by_model[j["model"]] = by_model.get(j["model"], 0) + 1
    print(f"wrote {dest}: {len(jobs)} jobs {by_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
