"""Seedable ambient (babble) noise mixing core for the ambient-noise-overlay
feature (feature-engineering/ambient-noise-overlay/SPEC.md).

Pure plan/mix/selection functions -- no CLI, no staging, no transcriber:
`dataset_tools.mix_ambient_noise` (stages mix/finalize) is the only consumer
besides tests. The SNR-based additive mixing reuses
`common.augment.apply_noise`, the same primitive the online `Augmenter` and
the wakeword's `mix_background_noise.py` already use, so the noise math stays
one code path across the repo.

Determinism: `plan_jobs` draws everything (row selection, chunk picks, SNRs)
from one `random.Random(seed)` in the caller's row order, so a fixed seed
plus a fixed eligible-row list reproduces the identical plan.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from me2_voicegen.common.augment import apply_noise

# Pilot human gate 2026-09-27 (SPEC Approvals): the [−5, 0) dB band was
# judged too high-noise and excluded -- draws are uniform over [0, 30] dB.
SNR_MIN_DB = 0.0
SNR_MAX_DB = 30.0
P_MIX_DEFAULT = 0.5
NUM_ATTEMPTS = 2
AMBIENT_SUFFIX = "_ambient"

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_BITS = 16

SESAME_WAKEWORD_TEXT = "Sesame"
V1_PAIR_UNSAFE_MARKERS = (" - ", "/", "\\")

CORPUS_MANIFEST_FIELDS = ["filename", "source_file", "split", "start_s", "duration", "rms"]

_UNSAFE_SLUG_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


class AmbientNoiseError(ValueError):
    """Invalid ambient-noise corpus, plan, or mix input (loud, never retried)."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    path: Path
    split: str
    start_s: float
    duration: float


@dataclass(frozen=True)
class AmbientJob:
    model: str
    source_row: dict
    attempt: int
    chunk: Chunk
    snr_db: float
    expected_text: str


def _probe(path: Path) -> None:
    info = torchaudio.info(str(path))
    if (info.sample_rate, info.num_channels, info.bits_per_sample) != (
        REQUIRED_SR,
        REQUIRED_CHANNELS,
        REQUIRED_BITS,
    ):
        raise AmbientNoiseError(
            f"{path}: {info.sample_rate}Hz/{info.num_channels}ch/"
            f"{info.bits_per_sample}bit != required {REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/"
            f"{REQUIRED_BITS}bit"
        )


def load_chunk_pool(corpus_root: str | Path, split: str) -> list[Chunk]:
    """Load one split's chunk pool from a corpus manifest, probe-verifying
    every chunk's format on load (the first load of each chunk)."""
    corpus_root = Path(corpus_root)
    manifest_path = corpus_root / "manifest.csv"
    if not manifest_path.is_file():
        raise AmbientNoiseError(f"corpus manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pool: list[Chunk] = []
    for row in sorted(rows, key=lambda r: r["filename"]):
        if row["split"] != split:
            continue
        wav_path = corpus_root / "audio" / row["filename"]
        if not wav_path.is_file():
            raise AmbientNoiseError(f"corpus chunk missing on disk: {wav_path}")
        _probe(wav_path)
        pool.append(
            Chunk(
                chunk_id=Path(row["filename"]).stem,
                path=wav_path,
                split=split,
                start_s=float(row["start_s"]),
                duration=float(row["duration"]),
            )
        )
    if not pool:
        raise AmbientNoiseError(f"corpus pool for split {split!r} is empty: {manifest_path}")
    return pool


def expected_text_for(model: str, row: dict) -> str:
    """The v1-pair expected text for one row: the transcript for VCM,
    "Sesame" for `_wakeword_` rows, and "" (free-decode gate) for
    `_unknown_` rows. Raises on anything not mixable."""
    if model == "vcm":
        text = str(row.get("transcript", ""))
        if not text.strip():
            raise AmbientNoiseError(f"vcm row {row.get('filename')!r}: empty transcript, not mixable")
        return text
    if model == "wakeword":
        label = row.get("label", "")
        if label == "_wakeword_":
            return SESAME_WAKEWORD_TEXT
        if label == "_unknown_":
            return ""
        raise AmbientNoiseError(f"wakeword row {row.get('filename')!r}: label {label!r} is not mixable")
    raise AmbientNoiseError(f"unknown model {model!r} (expected 'vcm' or 'wakeword')")


def plan_jobs(
    model: str,
    source_rows: list[dict],
    pools: dict[str, list[Chunk]],
    *,
    p_mix: float = P_MIX_DEFAULT,
    snr_min_db: float = SNR_MIN_DB,
    snr_max_db: float = SNR_MAX_DB,
    seed: int = 0,
) -> list[AmbientJob]:
    """Deterministic plan: each eligible row in order is selected with
    probability `p_mix`; each selected row gets two jobs (attempt 1 and 2),
    each with an independently drawn SNR uniform in [snr_min_db, snr_max_db]
    and a chunk drawn (with replacement) from the row's own split pool, the
    two attempts always using distinct chunks. No filesystem access."""
    if not 0.0 <= p_mix <= 1.0:
        raise AmbientNoiseError(f"p_mix must be in [0, 1], got {p_mix}")
    if snr_min_db > snr_max_db:
        raise AmbientNoiseError(f"snr_min_db {snr_min_db} > snr_max_db {snr_max_db}")

    rng = random.Random(seed)
    jobs: list[AmbientJob] = []
    for row in source_rows:
        if rng.random() >= p_mix:
            continue
        split = row["split"]
        pool = pools.get(split)
        if not pool:
            raise AmbientNoiseError(f"no corpus pool for split {split!r} (row {row['filename']!r})")
        expected_text = expected_text_for(model, row)

        chunk1 = pool[rng.randrange(len(pool))]
        snr1 = rng.uniform(snr_min_db, snr_max_db)
        jobs.append(
            AmbientJob(model=model, source_row=row, attempt=1, chunk=chunk1, snr_db=snr1, expected_text=expected_text)
        )

        chunk2 = pool[rng.randrange(len(pool))]
        while chunk2.chunk_id == chunk1.chunk_id:
            chunk2 = pool[rng.randrange(len(pool))]
        snr2 = rng.uniform(snr_min_db, snr_max_db)
        jobs.append(
            AmbientJob(model=model, source_row=row, attempt=2, chunk=chunk2, snr_db=snr2, expected_text=expected_text)
        )
    return jobs


def job_key(job: AmbientJob) -> str:
    """Stable key shared with the CLI's qa_plan.csv / report parsing."""
    return f"{job.source_row.get('group_id', '')}|{job.source_row.get('filename', '')}|a{job.attempt}"


def select_passing(jobs: list[AmbientJob], passed: dict[str, bool]) -> list[AmbientJob]:
    """Option B, per attempt independently: keep an attempt iff
    passed[job_key(job)] is True -- a row therefore ships with 0, 1, or 2
    siblings, every stored one gate-passed. A missing entry == gate failure
    (never passes). Plan order is preserved."""
    return [job for job in jobs if passed.get(job_key(job), False)]


def validate_v1_pair_text(text: str, filename: str) -> None:
    """Reject expected texts that would break v1-pair naming (the transcriber
    splits the stem on the first " - "; "/" and "\\" escape the shim dir).
    Never rewritten -- the source transcript must be fixed."""
    if not text.strip():
        raise AmbientNoiseError(f"row {filename!r}: empty expected text, cannot build a v1-pair filename")
    for marker in V1_PAIR_UNSAFE_MARKERS:
        if marker in text:
            raise AmbientNoiseError(
                f"row {filename!r}: expected text {text!r} contains {marker!r} -- unsafe for "
                "v1-pair naming; fix the source transcript, this is not silently rewritten"
            )


def v1_pair_violations(jobs: list[AmbientJob]) -> list[str]:
    """Collect (not raise on the first) every v1-pair naming violation in a
    plan, so the CLI can fail once listing all offending rows."""
    violations: list[str] = []
    for job in jobs:
        if not job.expected_text:
            continue
        try:
            validate_v1_pair_text(job.expected_text, str(job.source_row.get("filename", "?")))
        except AmbientNoiseError as exc:
            violations.append(str(exc))
    return violations


def _slug(value: str) -> str:
    slug = _UNSAFE_SLUG_CHARS.sub("-", value).strip("-")
    if not slug:
        raise AmbientNoiseError(f"cannot derive a filename-safe slug from {value!r}")
    return slug


def ambient_filename_from(group_id: str, filename: str, attempt: int, chunk_id: str) -> str:
    """`<group_slug>__<source_stem>__amb<attempt>-<chunk_stem>.wav` (the
    `mix_background_noise.dest_filename` pattern); (group_id, filename) is
    the pair both manifests already guarantee unique. `chunk_id` may be the
    bare stem or the corpus filename (with .wav)."""
    group_slug = _slug(str(group_id))
    source_slug = _slug(Path(str(filename)).stem)
    chunk_slug = _slug(Path(str(chunk_id)).stem)
    return f"{group_slug}__{source_slug}__amb{attempt}-{chunk_slug}.wav"


def ambient_filename(job: AmbientJob) -> str:
    return ambient_filename_from(
        job.source_row.get("group_id", ""),
        job.source_row.get("filename", ""),
        job.attempt,
        job.chunk.chunk_id,
    )


def mix_one(job: AmbientJob, source_path: str | Path, out_path: str | Path) -> None:
    """Load the source clip and the job's chunk, mix at `job.snr_db` (the
    chunk is looped/trimmed to the source's length by `apply_noise`), clamp
    to ±1.0, and write a 16 kHz mono PCM_16 wav. Probe-verifies both inputs
    (contractually 16k/mono/16-bit; refused, not resampled)."""
    source_path = Path(source_path)
    out_path = Path(out_path)
    if not source_path.is_file():
        raise AmbientNoiseError(f"source clip missing on disk: {source_path}")
    _probe(source_path)
    _probe(job.chunk.path)

    source_wave, _ = torchaudio.load(str(source_path))
    if source_wave.dim() == 2:
        source_wave = source_wave.mean(dim=0)
    noise_wave, _ = torchaudio.load(str(job.chunk.path))
    if noise_wave.dim() == 2:
        noise_wave = noise_wave.mean(dim=0)

    mixed = apply_noise(source_wave, noise_wave, job.snr_db)
    mixed = torch.clamp(mixed, -1.0, 1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), mixed.unsqueeze(0), REQUIRED_SR, bits_per_sample=REQUIRED_BITS)
