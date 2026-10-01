"""Build wake-word + command replay sessions for streaming evaluation.

Each session is one wav: room tone throughout, with a real wake-word clip
followed by a real command clip:

    [lead 0.5-1.5 s] [wake word] [gap 0.1-0.4 s] [command] [tail 3.5 s]

The room tone is a same-split `silence`-bucket clip from the VCM manifest
(looped to length), mixed under the speech at SNR U[15, 25] dB relative to
the command. Command selection matches the earlier streaming recall check
(the P4 check in docs/DENSE-SCORING-DECISION.md): every PAUSE/STOP/TIME clip of the split
plus the first 30 of each other intent, in manifest order.

The true end of the command's speech is found by CTC forced alignment of the
clean command clip with a stride-1 checkpoint (10 ms frames), so latency can
be measured from when the speaker actually stopped, not from the clip end.

Everything is a pure function of (manifests, split, seed): the same inputs
give the same sessions.

    uv run python -m me2_voicegen.vcm.streaming.session_replay --split test \
        --out-dir out/vcm/wakeword-sliding/sessions-test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio

from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform
from me2_voicegen.vcm.segment_scorer import force_align
from me2_voicegen.vcm.text import normalize_text, resolve_transcript

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VCM_MANIFEST = PROJECT_ROOT / "out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv"
DEFAULT_WAKEWORD_MANIFEST = PROJECT_ROOT / "out/conversions/v2/wakeword-sesame/manifest.csv"
DEFAULT_ALIGN_CHECKPOINT = PROJECT_ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/checkpoints/checkpoint.pt"
FOCUS = ("PAUSE", "STOP", "TIME")
PER_INTENT = 30
# Long enough that a fixed-period policy (decodes 3 s after the last wake-word
# detection) always completes its period inside the session.
TAIL_S = 3.5
FRAME_S = 0.01  # stride-1 alignment model: one posterior frame per 10 ms hop


def _load(path: Path) -> torch.Tensor:
    wave, sr = torchaudio.load(str(path))
    wave = wave.mean(0)
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    return wave


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt().clamp_min(1e-8))


def select_commands(rows: list[dict], split: str) -> list[dict]:
    seen: dict[str, int] = {}
    chosen = []
    for r in rows:
        if r["split"] != split or r["bucket"] != "target_commands":
            continue
        n = seen.get(r["label"], 0)
        if r["label"] in FOCUS or n < PER_INTENT:
            chosen.append(r)
        seen[r["label"]] = n + 1
    return chosen


def speech_bounds(model, extractor, wave: torch.Tensor, transcript: str) -> tuple[float, float]:
    """(start_s, end_s) of the transcript inside `wave`, from forced alignment."""
    logp = np.asarray(logp_for_waveform(model, extractor, wave, device="cpu"))
    alignment = force_align(logp, normalize_text(transcript))
    return alignment.start_frame * FRAME_S, (alignment.end_frame + 1) * FRAME_S


def build(
    split: str,
    out_dir: Path,
    seed: int = 0,
    vcm_manifest: Path = DEFAULT_VCM_MANIFEST,
    wakeword_manifest: Path = DEFAULT_WAKEWORD_MANIFEST,
    align_checkpoint: Path = DEFAULT_ALIGN_CHECKPOINT,
    limit: int | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    vcm_rows = list(csv.DictReader(vcm_manifest.open(newline="", encoding="utf-8")))
    commands = select_commands(vcm_rows, split)[:limit]
    beds = [r for r in vcm_rows if r["split"] == split and r["bucket"] == "silence"]
    wake_rows = [
        r for r in csv.DictReader(wakeword_manifest.open(newline="", encoding="utf-8"))
        if r["split"] == split and r["label"] == "_wakeword_"
    ]
    if not beds or not wake_rows:
        raise SystemExit(f"split {split!r} needs silence rows and wake-word positives")

    model, _ = load_checkpoint(align_checkpoint, device="cpu", weights_only=True)
    extractor = LogMelFeatureExtractor()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sessions = []
    for i, row in enumerate(commands):
        command = _load(vcm_manifest.parent / row["path"])
        wake_row = rng.choice(wake_rows)
        wake = _load(wakeword_manifest.parent / wake_row["path"])
        bed_row = rng.choice(beds)
        bed = _load(vcm_manifest.parent / bed_row["path"])
        lead_s = rng.uniform(0.5, 1.5)
        gap_s = rng.uniform(0.1, 0.4)
        snr_db = rng.uniform(15.0, 25.0)

        transcript = resolve_transcript(row)
        start_s, end_s = speech_bounds(model, extractor, command, transcript)

        lead, gap, tail = (torch.zeros(int(s * SAMPLE_RATE)) for s in (lead_s, gap_s, TAIL_S))
        speech = torch.cat([lead, wake, gap, command, tail])
        reps = int(np.ceil(speech.numel() / max(1, bed.numel())))
        bed = bed.repeat(reps)[: speech.numel()]
        bed = bed * (_rms(command) / _rms(bed)) * 10 ** (-snr_db / 20)
        session = (speech + bed).clamp(-1.0, 1.0)

        command_at = lead.numel() + wake.numel() + gap.numel()
        name = f"{i:04d}_{row['label']}.wav"
        torchaudio.save(str(audio_dir / name), session.unsqueeze(0), SAMPLE_RATE)
        sessions.append({
            "file": f"audio/{name}",
            "label": row["label"],
            "transcript": transcript,
            "command_filename": row["filename"],
            "wakeword_filename": wake_row["filename"],
            "bed_filename": bed_row["filename"],
            "snr_db": round(snr_db, 2),
            "wake_end_s": round((lead.numel() + wake.numel()) / SAMPLE_RATE, 4),
            "command_start_s": round(command_at / SAMPLE_RATE, 4),
            "speech_start_s": round(command_at / SAMPLE_RATE + start_s, 4),
            "speech_end_s": round(command_at / SAMPLE_RATE + end_s, 4),
            "duration_s": round(session.numel() / SAMPLE_RATE, 4),
        })
    meta = {"split": split, "seed": seed, "n": len(sessions), "vcm_manifest": str(vcm_manifest),
            "wakeword_manifest": str(wakeword_manifest), "align_checkpoint": str(align_checkpoint),
            "sessions": sessions}
    (out_dir / "sessions.json").write_text(json.dumps(meta, indent=1))
    return sessions


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vcm-manifest", type=Path, default=DEFAULT_VCM_MANIFEST)
    p.add_argument("--wakeword-manifest", type=Path, default=DEFAULT_WAKEWORD_MANIFEST)
    p.add_argument("--align-checkpoint", type=Path, default=DEFAULT_ALIGN_CHECKPOINT)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    s = build(a.split, a.out_dir, a.seed, a.vcm_manifest, a.wakeword_manifest, a.align_checkpoint, a.limit)
    print(f"wrote {len(s)} sessions to {a.out_dir}")


if __name__ == "__main__":
    main()
