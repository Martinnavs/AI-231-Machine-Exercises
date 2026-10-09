"""Slot-bearing TTS eval set: 2 personas x 12 slot-bearing phrases, synthesized
via the existing CosyVoice2 zero-shot pipeline (see generate_personas.py /
synthesis/), for exercising Task 03's grammar decoder's $NUMBER/$ARTIST/
$PERSONA/$TIME_UNIT/am-pm slot extraction on real (non-synthetic-posterior)
audio.

Framing (see docs/VCM-CONTRACT.md and Task 04): the acoustic model saw zero
slot-word audio during training, so this eval set quantifies that KNOWN
training gap - it is not proof that slot extraction is correct. The actual
correctness proof for slot extraction lives in Task 03's synthetic-posterior
tests. This set must never be added to the training split (see NON-GOALS in
the owning ticket); every row this module emits has split="eval".

CosyVoice2 outputs 24kHz float audio; every other source in this pipeline
(see out/conversions/v2/test_set/manifest.csv) is 16kHz mono 16-bit PCM, so
every clip written here is resampled and the resample is verified by reading
the written file back (not by trusting the resample call's intent).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, build_config
from me2_voicegen.generation.cosyvoice_env import OUT_DIR
from me2_voicegen.generation.personas import Persona, load_personas
from me2_voicegen.synthesis.base import SynthesisResult
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class, list_backends

logger = logging.getLogger(__name__)

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2

DEFAULT_OUT_DIR = OUT_DIR / "vcm" / "slot_eval"

RMS_SILENCE_FLOOR = 1e-4


@dataclass(frozen=True)
class SlotPhrase:
    """One eval phrase: its intent label (an `INTENT_PHRASES` key, per
    docs/VCM-CONTRACT.md), the exact text to synthesize, and the slot values
    it's expected to exercise."""

    intent: str
    text: str
    slots: dict[str, str]


# 12 phrases, 2 personas each = 24 clips. Collectively touch $NUMBER,
# $ARTIST, $PERSONA, $TIME_UNIT, and am/pm at least once each (see
# docs/VCM-CONTRACT.md section 3/the owning ticket's "Established" section
# for the slot vocabularies these values are drawn from).
SLOT_PHRASES: tuple[SlotPhrase, ...] = (
    SlotPhrase("ALARM", "set an alarm for five am", {"NUMBER": "five", "AMPM": "am"}),
    SlotPhrase("ALARM", "set an alarm for seven pm", {"NUMBER": "seven", "AMPM": "pm"}),
    SlotPhrase("CALL", "call mom", {"PERSONA": "mom"}),
    SlotPhrase("CALL", "call dad", {"PERSONA": "dad"}),
    SlotPhrase("PLAY_MUSIC", "play music by taylor swift", {"ARTIST": "taylor swift"}),
    SlotPhrase("PLAY_MUSIC", "play music by the weeknd", {"ARTIST": "the weeknd"}),
    SlotPhrase("PLAY_MUSIC", "play music by bad bunny", {"ARTIST": "bad bunny"}),
    SlotPhrase("PLAY_MUSIC", "play music by drake", {"ARTIST": "drake"}),
    SlotPhrase("PLAY_MUSIC", "play music by billie eilish", {"ARTIST": "billie eilish"}),
    SlotPhrase(
        "TIMER", "set a timer for twenty minutes", {"NUMBER": "twenty", "TIME_UNIT": "minutes"}
    ),
    SlotPhrase(
        "TIMER", "set a timer for thirty seconds", {"NUMBER": "thirty", "TIME_UNIT": "seconds"}
    ),
    SlotPhrase("TIMER", "set a timer for one hour", {"NUMBER": "one", "TIME_UNIT": "hour"}),
)

MANIFEST_FIELDS = [
    "filename",
    "path",
    "bucket",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
]


def slot_categories_covered(phrases: tuple[SlotPhrase, ...] = SLOT_PHRASES) -> set[str]:
    """The set of slot categories these phrases collectively exercise, plus
    'AMPM' folded into a synthetic 'TIME_UNIT'-adjacent check for am/pm
    coverage. Used by tests to assert nothing was silently dropped."""
    categories: set[str] = set()
    for phrase in phrases:
        categories.update(phrase.slots)
    return categories


def eval_clip_filename(persona_name: str, index: int) -> str:
    return f"{persona_name}_slot{index:02d}.wav"


def resample_mono_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """audio: (channels, samples) float32, per SynthesisResult's contract.
    Returns mono float32 samples at REQUIRED_SR. Uses soundfile-free linear
    resampling only when orig_sr == REQUIRED_SR already (no-op); otherwise
    defers to librosa, matching this repo's existing resample convention in
    build_test_set.py's write_clip()."""
    mono = audio[0] if audio.shape[0] == 1 else audio.mean(axis=0)
    if orig_sr == REQUIRED_SR:
        return mono.astype(np.float32)

    import librosa

    return librosa.resample(mono.astype(np.float32), orig_sr=orig_sr, target_sr=REQUIRED_SR)


def write_eval_clip(result: SynthesisResult, dest_path: Path) -> tuple[int, int, int]:
    """Resample result.audio to 16kHz mono 16-bit PCM and write it to
    dest_path, then read the written file back to verify the format actually
    landed as required (not just that resample_mono_16k was called with the
    right intent). Raises RuntimeError on mismatch."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    mono_16k = resample_mono_16k(result.audio, result.sample_rate)
    sf.write(str(dest_path), mono_16k, REQUIRED_SR, subtype="PCM_16")

    with wave.open(str(dest_path), "rb") as w:
        sr_w, ch_w, sw_w = w.getframerate(), w.getnchannels(), w.getsampwidth()
    if (sr_w, ch_w, sw_w) != (REQUIRED_SR, REQUIRED_CHANNELS, REQUIRED_SAMPWIDTH):
        raise RuntimeError(
            f"{dest_path}: written format {sr_w}Hz/{ch_w}ch/{sw_w * 8}bit != "
            f"required {REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/{REQUIRED_SAMPWIDTH * 8}bit"
        )
    return sr_w, ch_w, sw_w


def clip_rms(path: Path) -> float:
    data, _ = sf.read(str(path), dtype="float32")
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def assert_non_silent(path: Path, floor: float = RMS_SILENCE_FLOOR) -> float:
    """Reads the written clip back and computes its RMS; raises RuntimeError
    if it's at or below `floor` (i.e. looks like silence, not real speech)."""
    rms = clip_rms(path)
    if rms <= floor:
        raise RuntimeError(f"{path}: RMS {rms:.6g} <= silence floor {floor:.6g}")
    return rms


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def build_manifest_row(
    *,
    filename: str,
    dest_path: Path,
    out_root: Path,
    intent: str,
    persona_name: str,
    index: int,
) -> dict:
    duration = wav_duration(dest_path)
    return {
        "filename": filename,
        "path": str(dest_path.relative_to(out_root)),
        "bucket": "vcm_slot_eval",
        "label": intent,
        "duration": f"{duration:.6f}",
        "sample_rate": str(REQUIRED_SR),
        "resampled": "True",
        "source_dataset": "vcm_slot_eval_synthetic",
        # Freshly synthesized for this eval set - there's no separate upstream
        # corpus file this clip was copied from, so source_relpath points at
        # itself (the clip IS the source), per this module's own convention.
        "source_relpath": filename,
        "group_id": persona_name,
        "split": "eval",
    }


def write_manifest(out_root: Path, rows: list[dict]) -> Path:
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def synthesize_slot_eval_set(
    personas: list[Persona],
    synthesizer,
    out_root: Path,
    phrases: tuple[SlotPhrase, ...] = SLOT_PHRASES,
) -> list[dict]:
    rows: list[dict] = []
    for persona in personas:
        for index, phrase in enumerate(phrases):
            filename = eval_clip_filename(persona.name, index)
            dest_path = out_root / filename
            result = synthesizer.synthesize(phrase.text, prompt=persona.prompt)
            write_eval_clip(result, dest_path)
            assert_non_silent(dest_path)
            rows.append(
                build_manifest_row(
                    filename=filename,
                    dest_path=dest_path,
                    out_root=out_root,
                    intent=phrase.intent,
                    persona_name=persona.name,
                    index=index,
                )
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize the 2-persona x 12-slot-phrase eval-only TTS set "
            "(not training data - see docs/VCM-CONTRACT.md)."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to a persona manifest JSON file (see personas.example.json)",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=list_backends(),
        help=f"synthesis backend to use (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"directory to write eval clips + manifest.csv to (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="device hint forwarded to the backend's constructor if it accepts one",
    )
    parser.add_argument(
        "--opt",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="backend-specific constructor option, repeatable",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    out_root = args.out_dir or DEFAULT_OUT_DIR

    try:
        personas = load_personas(args.manifest)
    except (ValueError, OSError) as exc:
        logger.error("manifest validation failed: %s", exc)
        return 1

    try:
        backend_cls = get_backend_class(args.backend)
        config = build_config(backend_cls, args.backend, args.device, args.opt)
        synthesizer = create_synthesizer(args.backend, **config)
    except Exception:
        logger.exception("failed to construct backend %r", args.backend)
        return 1

    out_root.mkdir(parents=True, exist_ok=True)
    try:
        rows = synthesize_slot_eval_set(personas, synthesizer, out_root)
    except Exception:
        logger.exception("slot eval synthesis failed")
        return 1

    manifest_path = write_manifest(out_root, rows)
    print(f"wrote {len(rows)} clips + manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
