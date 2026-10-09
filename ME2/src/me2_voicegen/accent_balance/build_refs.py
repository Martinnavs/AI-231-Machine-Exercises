"""T1 (.scratch/accent-balance-fil50/tickets/00-RECAP.md): build the Filipino
voice-prompt pool used by every later stage.

Two prompt sources:
  - `sapinsapin`: `filipino_speech_corpus/manifest.csv` speakers (Tagalog/Taglish
    speech). Their own clips are concatenated, with a short silence gap, into an
    8-12s prompt; the prompt text is the concatenated clips' own sentences.
  - `references`: the existing `tagalog*`/`ilonggo*` accented-English voices in
    `~/cosy-voice-data/References`. Each is cut to an 8-12s low-energy boundary and
    transcribed fresh (the source `.txt` sidecars, where present, transcribe the
    FULL original clip, not this cut, so they are never reused as prompt text).

Split assignment (RECAP D2/T1): sapinsapin speakers are split 70/15/15 into
train/val/test, stratified by gender, seeded. References voices are train-only
(they're already used in every existing wakeword split via voice conversion, so
they cannot serve as a held-out voice for this feature).

Output: `<out-dir>/<voice_id>.wav` (16kHz mono PCM16) + `<voice_id>.txt` (prompt
text) + `<out-dir>/voices.csv`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import soundfile as sf

from me2_voicegen.dataset_tools.transcribe import transcribe_cached
from me2_voicegen.wakeword.fetch_positives import (
    ManifestValidationError,
    PathTraversalError,
    probe_wav,
    resolve_under,
    sanitize_component,
)

logger = logging.getLogger(__name__)

REQUIRED_SR = 16000
MIN_PROMPT_SECONDS = 8.0
MAX_PROMPT_SECONDS = 12.0
HARD_CAP_SECONDS = 25.0  # CosyVoice2's own inference-time assert (generate_conversions.MAX_PROMPT_SECONDS)
GAP_SECONDS = 0.15
SPLIT_SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}

VOICES_FIELDS = ["voice_id", "prompt_source", "split", "prompt_seconds", "prompt_text", "origin"]


@dataclass
class Clip:
    path: Path
    sentence: str
    duration: float
    speech_type: str


def load_fsc_speakers(manifest_path: Path) -> dict[str, list[Clip]]:
    """speaker_id -> usable clips (real file, non-empty non-tag sentence),
    sorted `read` speech_type first (Ticket T1's stated preference) then by
    filename for determinism."""
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ManifestValidationError(f"fsc manifest not found: {manifest_path}")
    audio_root = manifest_path.parent / "audio"

    by_speaker: dict[str, list[Clip]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sentence = (row.get("sentence") or "").strip()
            if not sentence or "[" in sentence:
                continue
            try:
                duration = float(row["duration"])
            except (KeyError, ValueError):
                continue
            if duration <= 0:
                continue
            clip_path = resolve_under(audio_root, audio_root / row["filename"])
            if not clip_path.is_file():
                continue
            speaker_id = row["speaker_id"]
            by_speaker.setdefault(speaker_id, []).append(
                Clip(path=clip_path, sentence=sentence, duration=duration, speech_type=row.get("speech_type") or "")
            )

    for speaker_id, clips in by_speaker.items():
        clips.sort(key=lambda c: (c.speech_type != "read", c.path.name))
    return by_speaker


def build_sapinsapin_prompt(clips: list[Clip], out_wav: Path, out_txt: Path) -> float | None:
    """Concatenate `clips` (already speaker-sorted) until the total is within
    [MIN_PROMPT_SECONDS, MAX_PROMPT_SECONDS], separated by GAP_SECONDS of
    silence. Returns the written duration, or None if the speaker's clips
    can't reach MIN_PROMPT_SECONDS at all (caller skips that speaker)."""
    sr = REQUIRED_SR
    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    chunks: list[np.ndarray] = []
    sentences: list[str] = []
    total = 0.0

    for clip in clips:
        if total >= MIN_PROMPT_SECONDS:
            break
        if total > 0 and total + clip.duration > MAX_PROMPT_SECONDS:
            continue
        data, native_sr = sf.read(str(clip.path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if native_sr != sr:
            import librosa

            data = librosa.resample(data, orig_sr=native_sr, target_sr=sr)
        if chunks:
            chunks.append(gap)
            total += GAP_SECONDS
        chunks.append(data)
        sentences.append(clip.sentence)
        total += clip.duration

    if total < MIN_PROMPT_SECONDS:
        return None

    audio = np.concatenate(chunks)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sr, subtype="PCM_16")
    text = " ".join(sentences)
    out_txt.write_text(text + "\n", encoding="utf-8")

    probe = probe_wav(out_wav)
    return probe.duration


def cut_low_energy(data: np.ndarray, sr: int) -> np.ndarray:
    """Cut `data` (mono float32) to end at the lowest-RMS point within
    [MIN_PROMPT_SECONDS, MAX_PROMPT_SECONDS] (or at len(data) if shorter than
    MIN_PROMPT_SECONDS), so the prompt doesn't end mid-word."""
    total_seconds = len(data) / sr
    if total_seconds <= MIN_PROMPT_SECONDS:
        return data

    hi = min(total_seconds, MAX_PROMPT_SECONDS)
    lo = MIN_PROMPT_SECONDS
    frame = int(0.02 * sr)
    lo_i, hi_i = int(lo * sr), int(hi * sr)
    if hi_i <= lo_i + frame:
        return data[:hi_i]

    best_i, best_rms = hi_i, float("inf")
    for i in range(lo_i, hi_i, frame):
        window = data[i : i + frame]
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) if len(window) else 0.0
        if rms < best_rms:
            best_rms, best_i = rms, i
    return data[:best_i]


def build_reference_prompt(mp3_path: Path, out_wav: Path, out_txt: Path, whisper_model: str) -> float:
    import librosa

    data, native_sr = librosa.load(str(mp3_path), sr=None, mono=True)
    cut = cut_low_energy(data.astype(np.float32), native_sr)
    if native_sr != REQUIRED_SR:
        cut = librosa.resample(cut, orig_sr=native_sr, target_sr=REQUIRED_SR)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), cut, REQUIRED_SR, subtype="PCM_16")

    probe = probe_wav(out_wav)
    if probe.duration > HARD_CAP_SECONDS:
        raise ManifestValidationError(f"{out_wav}: cut prompt is {probe.duration:.1f}s, over the {HARD_CAP_SECONDS}s hard cap")

    text = transcribe_cached(out_wav, model_size=whisper_model)
    if not text.strip():
        raise ManifestValidationError(f"{mp3_path}: whisper produced an empty transcript for the cut prompt")
    out_txt.write_text(text.strip() + "\n", encoding="utf-8")
    return probe.duration


def assign_splits(voice_ids: list[str], genders: dict[str, str], seed: int) -> dict[str, str]:
    """70/15/15 train/val/test, stratified by gender bucket, seeded, largest-
    remainder rounding within each bucket (same spirit as
    wakeword/convert_positives.stratified_sample_refs's apportionment)."""
    rng = Random(seed)
    by_gender: dict[str, list[str]] = {}
    for vid in voice_ids:
        by_gender.setdefault(genders.get(vid, "unknown"), []).append(vid)

    assignment: dict[str, str] = {}
    for gender in sorted(by_gender):
        group = sorted(by_gender[gender])
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * SPLIT_SHARES["train"])
        n_val = round(n * SPLIT_SHARES["val"])
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for vid in group[:n_train]:
            assignment[vid] = "train"
        for vid in group[n_train : n_train + n_val]:
            assignment[vid] = "val"
        for vid in group[n_train + n_val :]:
            assignment[vid] = "test"
    return assignment


def build(fsc_manifest: Path, old_refs_dir: Path, out_dir: Path, seed: int, whisper_model: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen_ids: set[str] = set()

    speakers = load_fsc_speakers(fsc_manifest)
    genders: dict[str, str] = {}
    sapinsapin_voice_ids: list[str] = []
    pending_sapinsapin: dict[str, tuple[list[Clip], float, str]] = {}

    for speaker_id in sorted(speakers):
        speaker_id_safe = sanitize_component(speaker_id, "fsc speaker_id")
        voice_id = f"fsc_{speaker_id_safe}"
        clips = speakers[speaker_id]
        out_wav = out_dir / f"{voice_id}.wav"
        out_txt = out_dir / f"{voice_id}.txt"
        duration = build_sapinsapin_prompt(clips, out_wav, out_txt)
        if duration is None:
            logger.warning("skipping fsc speaker %s: not enough usable audio to reach %.1fs", speaker_id, MIN_PROMPT_SECONDS)
            continue
        if duration > HARD_CAP_SECONDS:
            raise ManifestValidationError(f"{out_wav}: {duration:.1f}s over the {HARD_CAP_SECONDS}s hard cap")
        text = out_txt.read_text(encoding="utf-8").strip()
        if not text:
            raise ManifestValidationError(f"{out_txt}: empty prompt text")
        if voice_id in seen_ids:
            raise ManifestValidationError(f"duplicate voice_id: {voice_id}")
        seen_ids.add(voice_id)
        sapinsapin_voice_ids.append(voice_id)
        # gender is per-row metadata in the fsc manifest, constant per speaker in practice;
        # take the first clip's row via a second lookup rather than threading it through Clip.
        pending_sapinsapin[voice_id] = (clips, duration, text)

    # second pass over the raw manifest just for gender, keyed by speaker_id (cheap, avoids
    # widening the Clip dataclass for one field only used here)
    with fsc_manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = f"fsc_{row['speaker_id']}"
            if vid in pending_sapinsapin and vid not in genders:
                genders[vid] = (row.get("gender") or "unknown").strip() or "unknown"

    split_by_voice = assign_splits(sapinsapin_voice_ids, genders, seed)
    for voice_id in sapinsapin_voice_ids:
        _, duration, text = pending_sapinsapin[voice_id]
        speaker_id = voice_id[len("fsc_") :]
        rows.append(
            {
                "voice_id": voice_id,
                "prompt_source": "sapinsapin",
                "split": split_by_voice[voice_id],
                "prompt_seconds": f"{duration:.3f}",
                "prompt_text": text,
                "origin": f"sapinsapin:speaker_{speaker_id}",
            }
        )

    if not old_refs_dir.is_dir():
        raise ManifestValidationError(f"--old-refs-dir not found: {old_refs_dir}")
    ref_paths = sorted(
        p for p in old_refs_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav") and (p.stem.startswith("tagalog") or p.stem.startswith("ilonggo"))
    )
    if not ref_paths:
        raise ManifestValidationError(f"no tagalog*/ilonggo* files found under {old_refs_dir}")
    for ref_path in ref_paths:
        voice_id = f"ref_{ref_path.stem}"
        if voice_id in seen_ids:
            raise ManifestValidationError(f"duplicate voice_id: {voice_id}")
        seen_ids.add(voice_id)
        out_wav = out_dir / f"{voice_id}.wav"
        out_txt = out_dir / f"{voice_id}.txt"
        duration = build_reference_prompt(ref_path, out_wav, out_txt, whisper_model)
        text = out_txt.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "voice_id": voice_id,
                "prompt_source": "references",
                "split": "train",
                "prompt_seconds": f"{duration:.3f}",
                "prompt_text": text,
                "origin": f"references:{ref_path.name}",
            }
        )

    return rows


def write_voices_csv(out_dir: Path, rows: list[dict]) -> Path:
    dest = out_dir / "voices.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VOICES_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fsc-manifest", type=Path, required=True)
    parser.add_argument("--old-refs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        rows = build(args.fsc_manifest, args.old_refs_dir, args.out_dir, args.seed, args.whisper_model)
    except (ManifestValidationError, PathTraversalError) as exc:
        logger.error("%s", exc)
        return 1
    dest = write_voices_csv(args.out_dir, rows)
    n_sapinsapin = sum(1 for r in rows if r["prompt_source"] == "sapinsapin")
    n_refs = sum(1 for r in rows if r["prompt_source"] == "references")
    by_split = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"wrote {dest}: {len(rows)} voices ({n_sapinsapin} sapinsapin, {n_refs} references)")
    print(f"by split: {by_split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
