"""Synthetic incomplete-prefix probe generator (docs/INCOMPLETE-GRAMMAR-
REJECTION.md, Step 5). Crops a real, complete Option B command clip at a
designated whole-word prefix's trailing word boundary -- using forced
alignment against the trained acoustic model, not a guessed cut point -- and
pads it with a bucketed duration of trailing silence, producing accented,
microphone-realistic "incomplete command" audio for margin calibration
(`vcm.optionb.calibration`, ticket 06) without recording any new speech.

Selection, crop-frame mapping, silence-bucket padding, and failure counting
are pure functions of already-resolved manifest rows, a synthetic-or-real
`(logp, waveform)` pair, and text -- exercised end to end by
`tests/test_vcm_incomplete_probes.py` with synthetic near-one-hot posteriors
built by the repo's `make_posterior` idiom, never a checkpoint or a WAV file.
Only `main()` (and the small `load_waveform`/`compute_logp` adapters it
wires up) touches torch, torchaudio, or the filesystem.

See docs/INCOMPLETE-GRAMMAR-REJECTION.md's "Probe-manifest contract" section
for the manifest schema this module writes -- fixed there (not here) as the
shared contract with ticket 06's calibration tool.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

import numpy as np

from me2_voicegen.common.features import HOP_LENGTH, SAMPLE_RATE
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.segment_scorer import ForcedAlignment, force_align
from me2_voicegen.vcm.text import CONVERSIONS_V2_DIR, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Fixed defaults / contract constants.
# ---------------------------------------------------------------------------

OPTIONB_MANIFEST = CONVERSIONS_V2_DIR / "optionb" / "manifest.csv"
DEFAULT_AUDIO_ROOT = OPTIONB_MANIFEST.parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "out" / "vcm" / "optionb-optiond" / "checkpoints" / "checkpoint.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "out" / "vcm" / "optionb" / "incomplete_probes"

DEFAULT_CAP_PER_PREFIX = 20
DEFAULT_GRACE_FRAMES = 3
"""~30ms. Chosen from a Task-05 manual inspection of 10 real val alignments
(docs/INCOMPLETE-GRAMMAR-REJECTION.md / this ticket's Execution Log): the
observed gap between a prefix's last aligned character and the next word's
first aligned character ranged 2-31 frames (median ~9-10). `crop_end_frame`
is `min(last_char_frame + 1 + grace, next_word_first_char_frame)`, so *any*
grace value is already clamped to never cross the next word -- grace only
controls whether the crop lands at its own natural point (small grace,
typical case) or gets clamped flush against the next word's onset (grace >=
gap, the rare tight-gap case, e.g. the observed 2-frame gap). 3 frames keeps
a little trailing-consonant room without being large enough to usually hit
the clamp."""
DEFAULT_SILENCE_BUCKETS_S: tuple[float, ...] = (0.0, 0.3, 1.0, 2.0)
QUIET_WINDOW_S = 0.1
QUIET_SEARCH_BACKOFF_FRAMES = 20
"""CTC spikes lag acoustic onset (and lead into the trailing blank on the
tail side), so the frames immediately adjacent to `start_frame`/`end_frame`
are often already speech, not silence (R2-1 reviewer reproduction on 40 real
val clips: median +24.7dB, 20/35 above +20dB, comparing the last 50ms of the
naive lead-in region to its first 50ms). Back off
`QUIET_SEARCH_BACKOFF_FRAMES` frames from both `start_frame` (searching
backwards) and `end_frame` (searching forwards) before searching for a quiet
window."""
QUIET_RELATIVE_DB = 20.0
"""A candidate window is only accepted if its RMS is at least this many dB
below the aligned speech span's RMS (R2-5, tickets/05's Review Feedback
(re-review)): "a region exists" is not "the region is actually quiet" --
`LogMelFeatureExtractor` normalizes per-utterance, so a mismatched-level
pad is an out-of-distribution shift on the whole utterance, not a neutral
silence. Falls back to `digital_zero` (never a loud fragment) when no
candidate in either the lead-in or the tail search region clears this bar."""
MIN_CROP_S = 0.15
AUDIT_SAMPLE_PER_PREFIX = 3

SOURCE_DATASET = "optionb_incomplete_probe"

FAILURE_FORCE_ALIGN = "force_align_error"
FAILURE_NO_GAP = "no_gap"
FAILURE_CROP_TOO_SHORT = "crop_too_short"
FAILURE_MISSING_AUDIO = "missing_audio"

MANIFEST_FIELDS: tuple[str, ...] = (
    "filename",
    "path",
    "bucket",
    "label",
    "split",
    "group_id",
    "source_dataset",
    "source_filename",
    "source_path",
    "source_transcript",
    "prefix",
    "prefix_word_count",
    "char_overlap",
    "grace_frames",
    "crop_end_frame",
    "crop_end_sample",
    "trailing_silence_s",
    "silence_source",
    "alignment_log_prob",
    "aligner_checkpoint",
    "aligner_checkpoint_sha256",
)


# ---------------------------------------------------------------------------
# Selection: which source rows go with which designated prefix.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    """One (source row, designated prefix) pairing selected for probing."""

    source_row: Mapping[str, str]
    prefix: str

    @property
    def prefix_word_count(self) -> int:
        return len(self.prefix.split())


def has_char_overlap(text: str, prefix: str) -> bool:
    """True iff `text` shares `prefix` as a leading character run without a
    following word boundary (e.g. text="reminders", prefix="remind") -- the
    false-match this module must never select as a probe target."""
    if not text.startswith(prefix):
        return False
    if text == prefix:
        return False
    return text[len(prefix)] != " "


def is_word_boundary_prefix(text: str, prefix: str) -> bool:
    """True iff `prefix` is a proper whole-word prefix of `text`, i.e. there
    is at least one more word after it (`text[len(prefix)] == ' '`)."""
    return text.startswith(prefix + " ")


def select_target_rows(
    rows: list[Mapping[str, str]],
    prefixes: frozenset[str],
    split: str,
    resolve_and_normalize: Callable[[Mapping[str, str]], str | None],
    cap_per_prefix: int = DEFAULT_CAP_PER_PREFIX,
    seed: int = 0,
    source_dataset: str = "optionb",
) -> list[ProbeSpec]:
    """Deterministic, seeded selection of up to `cap_per_prefix` source rows
    per designated prefix, from `rows` filtered to `split`/`source_dataset`.

    A row is a candidate for `prefix` iff its resolved-and-normalized
    transcript is a proper whole-word prefix match (`is_word_boundary_prefix`);
    this excludes character-overlap false matches by construction (a text
    like "reminders ..." never `startswith("remind ")`).
    """
    specs: list[ProbeSpec] = []
    for prefix in sorted(prefixes):
        candidates: list[Mapping[str, str]] = []
        for row in rows:
            if row.get("split") != split or row.get("source_dataset") != source_dataset:
                continue
            text = resolve_and_normalize(row)
            if text is None:
                continue
            if is_word_boundary_prefix(text, prefix):
                candidates.append(row)
        candidates.sort(key=lambda r: r.get("filename", ""))
        if len(candidates) > cap_per_prefix:
            rng = random.Random(f"{seed}:{split}:{prefix}")
            candidates = rng.sample(candidates, cap_per_prefix)
            candidates.sort(key=lambda r: r.get("filename", ""))
        specs.extend(ProbeSpec(source_row=row, prefix=prefix) for row in candidates)
    return specs


# ---------------------------------------------------------------------------
# Crop-frame mapping.
# ---------------------------------------------------------------------------


class CropResult(NamedTuple):
    last_char_frame: int
    next_word_first_char_frame: int
    crop_end_frame: int
    crop_end_sample: int


class CropFailure(NamedTuple):
    reason: str


def _frames_for_char_index(alignment: ForcedAlignment, char_index: int) -> list[int]:
    target_state = 2 * char_index + 1
    return [i for i, state in enumerate(alignment.state_path) if state == target_state]


def compute_crop(
    alignment: ForcedAlignment,
    text: str,
    prefix: str,
    grace_frames: int,
    frame_hop: int = HOP_LENGTH,
) -> CropResult | CropFailure:
    """Map the last character of `prefix` and the first character of the
    following word to aligned frames (state `2*char_index + 1` is character
    `char_index`, per `segment_scorer.ForcedAlignment.state_path`), then
    compute `crop_end_frame = min(last_char_frame + 1 + grace, next_word_
    first_char_frame)` -- the `min` guarantees the crop never reaches the
    next word's own first aligned frame, for any `grace_frames`.

    `frame_hop` (waveform samples per posterior frame, from
    `pipeline.frame_hop_samples`; `HOP_LENGTH` default = stride-1) converts
    `crop_end_frame` -- a posterior frame index -- to a sample offset."""
    if not is_word_boundary_prefix(text, prefix):
        raise ValueError(f"{prefix!r} is not a word-boundary prefix of {text!r}")

    last_char_index = len(prefix) - 1
    next_char_index = len(prefix) + 1

    last_frames = _frames_for_char_index(alignment, last_char_index)
    next_frames = _frames_for_char_index(alignment, next_char_index)
    if not last_frames or not next_frames:
        raise AssertionError(
            "non-blank CTC state was never visited -- impossible for a finite "
            "force_align path"
        )

    last_char_frame = last_frames[-1]
    next_word_first_char_frame = next_frames[0]

    if next_word_first_char_frame - (last_char_frame + 1) <= 0:
        return CropFailure(FAILURE_NO_GAP)

    crop_end_frame = min(last_char_frame + 1 + grace_frames, next_word_first_char_frame)
    crop_end_sample = crop_end_frame * frame_hop

    if crop_end_sample < int(round(MIN_CROP_S * SAMPLE_RATE)):
        return CropFailure(FAILURE_CROP_TOO_SHORT)

    return CropResult(
        last_char_frame=last_char_frame,
        next_word_first_char_frame=next_word_first_char_frame,
        crop_end_frame=crop_end_frame,
        crop_end_sample=crop_end_sample,
    )


# ---------------------------------------------------------------------------
# Trailing-silence padding.
# ---------------------------------------------------------------------------

SILENCE_SOURCE_NONE = "none"
SILENCE_SOURCE_LEAD_IN = "lead_in_room_tone"
SILENCE_SOURCE_TAIL = "tail_room_tone"
SILENCE_SOURCE_DIGITAL_ZERO = "digital_zero"


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def _scan_min_rms_window(
    region: np.ndarray, window_samples: int, stride: int
) -> tuple[np.ndarray, float] | None:
    if region.size < window_samples:
        return None
    best_window: np.ndarray | None = None
    best_rms = np.inf
    for offset in range(0, region.size - window_samples + 1, stride):
        window = region[offset : offset + window_samples]
        rms = _rms(window)
        if rms < best_rms:
            best_rms = rms
            best_window = window
    assert best_window is not None
    return best_window, best_rms


def find_quiet_window(
    waveform: np.ndarray,
    alignment: ForcedAlignment,
    sample_rate: int = SAMPLE_RATE,
    backoff_frames: int = QUIET_SEARCH_BACKOFF_FRAMES,
    window_s: float = QUIET_WINDOW_S,
    stride: int = HOP_LENGTH,
    relative_db: float = QUIET_RELATIVE_DB,
    frame_hop: int = HOP_LENGTH,
) -> tuple[np.ndarray | None, str]:
    """Search both the lead-in (`[0, (start_frame - backoff_frames) *
    frame_hop)`) and the tail (`[(end_frame + 1 + backoff_frames) * frame_hop,
    len(waveform))`) for the minimum-RMS `window_s`-long window, then accept
    it only if its RMS is at least `relative_db` dB below the aligned speech
    span's RMS (R2-5: existence alone does not mean quiet -- a
    same-level-as-speech "silence" pad is worse than no fix at all).
    Returns `(None, SILENCE_SOURCE_DIGITAL_ZERO)` if neither region yields an
    acceptable window (including when the speech span itself measures as
    silence, since a relative threshold is meaningless against a zero
    reference).

    `alignment.start_frame`/`end_frame` are posterior frame indices, so every
    frame-to-sample boundary here uses `frame_hop` (waveform samples per
    posterior frame, from `pipeline.frame_hop_samples`; the `HOP_LENGTH`
    default is the stride-1 behavior). The `stride` parameter is a different
    thing -- the search step size in sample space, not a posterior-frame
    conversion -- and stays independent of the model's stride."""
    waveform = np.asarray(waveform)
    window_samples = int(round(window_s * sample_rate))
    backoff_samples = backoff_frames * frame_hop

    lead_in_end = alignment.start_frame * frame_hop - backoff_samples
    lead_in_region = waveform[: max(lead_in_end, 0)]

    tail_start = (alignment.end_frame + 1) * frame_hop + backoff_samples
    tail_region = waveform[min(tail_start, waveform.size) :]

    speech_start = alignment.start_frame * frame_hop
    speech_end = (alignment.end_frame + 1) * frame_hop
    speech_rms = _rms(waveform[speech_start:speech_end])

    candidates: list[tuple[float, np.ndarray, str]] = []
    lead_in_result = _scan_min_rms_window(lead_in_region, window_samples, stride)
    if lead_in_result is not None:
        window, rms = lead_in_result
        candidates.append((rms, window, SILENCE_SOURCE_LEAD_IN))
    tail_result = _scan_min_rms_window(tail_region, window_samples, stride)
    if tail_result is not None:
        window, rms = tail_result
        candidates.append((rms, window, SILENCE_SOURCE_TAIL))

    if not candidates or speech_rms <= 0.0:
        return None, SILENCE_SOURCE_DIGITAL_ZERO

    candidates.sort(key=lambda c: c[0])
    best_rms, best_window, best_source = candidates[0]

    threshold = speech_rms * (10.0 ** (-relative_db / 20.0))
    if best_rms > threshold:
        return None, SILENCE_SOURCE_DIGITAL_ZERO

    return best_window, best_source


def build_trailing_silence(
    quiet_window: np.ndarray | None,
    quiet_source: str,
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, str]:
    """Tile `quiet_window` (from `find_quiet_window`, along with its
    `quiet_source` label) to fill `duration_s` seconds. Falls back to
    digital zero when `quiet_window` is `None`, regardless of `quiet_source`."""
    n_target = int(round(duration_s * sample_rate))
    if n_target <= 0:
        return np.zeros(0, dtype=np.float32), SILENCE_SOURCE_NONE

    if quiet_window is None or quiet_window.size == 0:
        return np.zeros(n_target, dtype=np.float32), SILENCE_SOURCE_DIGITAL_ZERO

    reps = -(-n_target // quiet_window.size)
    tiled = np.tile(quiet_window, reps)[:n_target]
    return tiled.astype(np.float32), quiet_source


# ---------------------------------------------------------------------------
# Output-path safety.
# ---------------------------------------------------------------------------


def assert_safe_manifest_output_path(
    output_path: Path,
    forbidden_path: Path = OPTIONB_MANIFEST,
) -> None:
    """Refuse to let a probe manifest overwrite the training manifest."""
    if Path(output_path).resolve() == Path(forbidden_path).resolve():
        raise ValueError(
            f"refusing to write the probe manifest to the training manifest "
            f"path {forbidden_path}"
        )


# ---------------------------------------------------------------------------
# Orchestration: pure given injected I/O callables (torch/file access lives
# only in main()'s adapters below).
# ---------------------------------------------------------------------------


def _slugify_prefix(prefix: str) -> str:
    return prefix.replace(" ", "_").replace("'", "")


def _slugify_bucket(duration_s: float) -> str:
    return f"{duration_s:g}".replace(".", "p")


@dataclasses.dataclass
class ProbeAudio:
    filename: str
    samples: np.ndarray


@dataclasses.dataclass
class GenerationResult:
    manifest_rows: list[dict] = dataclasses.field(default_factory=list)
    audio: list[ProbeAudio] = dataclasses.field(default_factory=list)
    failure_counts: Counter = dataclasses.field(default_factory=Counter)
    failures: list[dict] = dataclasses.field(default_factory=list)


def generate_probes(
    specs: list[ProbeSpec],
    split: str,
    grace_frames: int,
    silence_buckets: tuple[float, ...],
    aligner_checkpoint_path: str,
    aligner_checkpoint_sha256: str,
    load_waveform: Callable[[Mapping[str, str]], np.ndarray],
    compute_logp: Callable[[np.ndarray], np.ndarray],
    resolve_and_normalize: Callable[[Mapping[str, str]], str | None],
    frame_hop: int = HOP_LENGTH,
) -> GenerationResult:
    """Build every probe variant for every `spec`. Every failure mode named
    in the ticket (missing audio, `force_align` `ValueError`, no gap, crop
    too short) is caught here and counted, never raised out of this
    function -- one bad source row must not abort the whole run.

    `frame_hop` (waveform samples per posterior frame, from
    `pipeline.frame_hop_samples`; `HOP_LENGTH` default = stride-1) is
    forwarded to `compute_crop` and `find_quiet_window` for their
    frame-index-to-sample conversions."""
    result = GenerationResult()

    for spec in specs:
        row = spec.source_row
        source_filename = row.get("filename", "")

        try:
            waveform = load_waveform(row)
        except (OSError, FileNotFoundError) as exc:
            result.failure_counts[FAILURE_MISSING_AUDIO] += 1
            result.failures.append(
                {
                    "source_filename": source_filename,
                    "prefix": spec.prefix,
                    "reason": FAILURE_MISSING_AUDIO,
                    "detail": str(exc),
                }
            )
            continue

        text = resolve_and_normalize(row)
        if text is None:
            result.failure_counts[FAILURE_MISSING_AUDIO] += 1
            result.failures.append(
                {
                    "source_filename": source_filename,
                    "prefix": spec.prefix,
                    "reason": FAILURE_MISSING_AUDIO,
                    "detail": "resolve_transcript returned None",
                }
            )
            continue

        logp = compute_logp(waveform)

        try:
            alignment = force_align(logp, text)
        except ValueError as exc:
            result.failure_counts[FAILURE_FORCE_ALIGN] += 1
            result.failures.append(
                {
                    "source_filename": source_filename,
                    "prefix": spec.prefix,
                    "reason": FAILURE_FORCE_ALIGN,
                    "detail": str(exc),
                }
            )
            continue

        crop = compute_crop(alignment, text, spec.prefix, grace_frames, frame_hop=frame_hop)
        if isinstance(crop, CropFailure):
            result.failure_counts[crop.reason] += 1
            result.failures.append(
                {
                    "source_filename": source_filename,
                    "prefix": spec.prefix,
                    "reason": crop.reason,
                    "detail": "",
                }
            )
            continue

        quiet_window, quiet_source = find_quiet_window(waveform, alignment, frame_hop=frame_hop)
        cropped = np.asarray(waveform)[: crop.crop_end_sample]

        for bucket_s in silence_buckets:
            silence, silence_source = build_trailing_silence(quiet_window, quiet_source, bucket_s)
            probe_samples = np.concatenate([cropped, silence]).astype(np.float32)
            filename = (
                f"{Path(source_filename).stem}__{_slugify_prefix(spec.prefix)}"
                f"__sil{_slugify_bucket(bucket_s)}s.wav"
            )
            path = f"audio/{split}/{filename}"
            result.audio.append(ProbeAudio(filename=filename, samples=probe_samples))
            result.manifest_rows.append(
                {
                    "filename": filename,
                    "path": path,
                    "bucket": "incomplete_prefix",
                    "label": "",
                    "split": split,
                    "group_id": row.get("group_id", ""),
                    "source_dataset": SOURCE_DATASET,
                    "source_filename": source_filename,
                    "source_path": row.get("path", ""),
                    "source_transcript": text,
                    "prefix": spec.prefix,
                    "prefix_word_count": spec.prefix_word_count,
                    "char_overlap": has_char_overlap(text, spec.prefix),
                    "grace_frames": grace_frames,
                    "crop_end_frame": crop.crop_end_frame,
                    "crop_end_sample": crop.crop_end_sample,
                    "trailing_silence_s": bucket_s,
                    "silence_source": silence_source,
                    "alignment_log_prob": alignment.log_probability,
                    "aligner_checkpoint": aligner_checkpoint_path,
                    "aligner_checkpoint_sha256": aligner_checkpoint_sha256,
                }
            )

    return result


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


ALL_FAILURE_REASONS: tuple[str, ...] = (
    FAILURE_FORCE_ALIGN,
    FAILURE_NO_GAP,
    FAILURE_CROP_TOO_SHORT,
    FAILURE_MISSING_AUDIO,
)


def build_generation_report(
    specs: list[ProbeSpec],
    result: GenerationResult,
    split: str,
) -> dict:
    failure_counts = {reason: result.failure_counts.get(reason, 0) for reason in ALL_FAILURE_REASONS}
    return {
        "split": split,
        "source_rows_selected": len(specs),
        "probe_rows_written": len(result.manifest_rows),
        "failure_counts": failure_counts,
        "failures": result.failures,
    }


def build_audit_sample(
    manifest_rows: list[dict],
    seed: int = 0,
    per_prefix: int = AUDIT_SAMPLE_PER_PREFIX,
) -> list[dict]:
    """Stratified sample of up to `per_prefix` distinct source rows per
    prefix, for human listening -- one representative row per chosen source
    row, using the *largest* silence bucket (R2-1: the padded tail must be
    included, not just the cropped prefix, so a listener can actually catch
    a padding source that turns out not to be quiet)."""
    by_prefix: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in manifest_rows:
        by_source = by_prefix[row["prefix"]]
        existing = by_source.get(row["source_filename"])
        if existing is None or row["trailing_silence_s"] > existing["trailing_silence_s"]:
            by_source[row["source_filename"]] = row

    sample: list[dict] = []
    for prefix in sorted(by_prefix):
        candidates = sorted(by_prefix[prefix].values(), key=lambda r: r["source_filename"])
        if len(candidates) > per_prefix:
            rng = random.Random(f"audit:{seed}:{prefix}")
            candidates = rng.sample(candidates, per_prefix)
            candidates.sort(key=lambda r: r["source_filename"])
        sample.extend(candidates)
    return sample


# ---------------------------------------------------------------------------
# I/O: manifest CSV, WAV, checkpoint hash, real-model adapters, CLI.
# ---------------------------------------------------------------------------


def write_probe_manifest(rows: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_audit_sample(rows: list[dict], audit_path: Path) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_generation_report(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_probe_audio(audio: list[ProbeAudio], out_dir: Path, split: str) -> None:
    import soundfile as sf

    audio_dir = out_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    for item in audio:
        sf.write(str(audio_dir / item.filename), item.samples, SAMPLE_RATE, subtype="PCM_16")


def _load_manifest_rows(manifest_path: Path) -> list[dict]:
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_waveform_from_row(audio_root: Path) -> Callable[[Mapping[str, str]], np.ndarray]:
    import torchaudio

    def _load(row: Mapping[str, str]) -> np.ndarray:
        wav_path = audio_root / row["path"]
        if not wav_path.exists():
            raise FileNotFoundError(str(wav_path))
        waveform, _sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        return waveform.numpy().astype(np.float32)

    return _load


def _resolve_and_normalize_row(row: Mapping[str, str]) -> str | None:
    from me2_voicegen.vcm.text import normalize_text, resolve_transcript

    transcript = resolve_transcript(dict(row))
    if transcript is None:
        return None
    return normalize_text(transcript)


def _compute_logp_fn(model, device: str) -> Callable[[np.ndarray], np.ndarray]:
    """Adapter around a caller-loaded `model` (see `main`): keeps torch and
    the feature extractor out of the pure core's import surface."""
    import torch

    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.pipeline import logp_for_waveform

    feature_extractor = LogMelFeatureExtractor()

    def _compute(waveform: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        return logp_for_waveform(model, feature_extractor, tensor, device=device)

    return _compute


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic incomplete-prefix probes by force-aligning "
            "and cropping real Option B command clips at a designated "
            "whole-word prefix's word boundary (docs/"
            "INCOMPLETE-GRAMMAR-REJECTION.md, Step 5)."
        )
    )
    parser.add_argument("--manifest", type=Path, default=OPTIONB_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cap-per-prefix", type=int, default=DEFAULT_CAP_PER_PREFIX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grace-frames", type=int, default=DEFAULT_GRACE_FRAMES)
    parser.add_argument(
        "--silence-bucket-s",
        type=float,
        action="append",
        default=None,
        help="repeatable; defaults to 0.0 0.3 1.0 2.0",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audio_root = args.audio_root or args.manifest.parent
    out_dir = args.out_dir or (DEFAULT_OUT_DIR / args.split)
    silence_buckets = tuple(args.silence_bucket_s) if args.silence_bucket_s else DEFAULT_SILENCE_BUCKETS_S

    manifest_out_path = out_dir / "manifest.csv"
    assert_safe_manifest_output_path(manifest_out_path)

    rows = _load_manifest_rows(args.manifest)
    specs = select_target_rows(
        rows,
        OPTIONB_GRAMMAR.incomplete_prefixes,
        args.split,
        _resolve_and_normalize_row,
        cap_per_prefix=args.cap_per_prefix,
        seed=args.seed,
    )

    from me2_voicegen.vcm.pipeline import frame_hop_samples, load_checkpoint

    checkpoint_sha256 = sha256_of_file(args.checkpoint)
    model, _checkpoint = load_checkpoint(args.checkpoint, device=args.device, weights_only=True)
    frame_hop = frame_hop_samples(model)
    compute_logp = _compute_logp_fn(model, args.device)
    load_waveform = _load_waveform_from_row(audio_root)

    result = generate_probes(
        specs,
        args.split,
        args.grace_frames,
        silence_buckets,
        aligner_checkpoint_path=str(args.checkpoint),
        aligner_checkpoint_sha256=checkpoint_sha256,
        load_waveform=load_waveform,
        compute_logp=compute_logp,
        resolve_and_normalize=_resolve_and_normalize_row,
        frame_hop=frame_hop,
    )

    write_probe_audio(result.audio, out_dir, args.split)
    write_probe_manifest(result.manifest_rows, manifest_out_path)
    write_generation_report(
        build_generation_report(specs, result, args.split), out_dir / "generation_report.json"
    )
    write_audit_sample(
        build_audit_sample(result.manifest_rows, seed=args.seed), out_dir / "audit_sample.csv"
    )

    print(
        f"wrote {len(result.manifest_rows)} probe clips from {len(specs)} source rows "
        f"to {manifest_out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
