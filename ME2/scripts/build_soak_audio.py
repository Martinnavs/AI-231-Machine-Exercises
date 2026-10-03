"""Build the wake word + gap + command soak audio for a manifest split (default: the ai231 holdout).

Each session is one 16 kHz mono wav:

    [lead 0.5-1.5 s] [wake word] [gap 0.0-1.0 s, steps of 0.1 s] [command] [tail 3.5 s]

Every row of the split becomes one session (the holdout: 186 commands and 16 out-of-scope clips, which must NOT trigger).
The whole session is convolved with a room impulse response from the fixed seeded pool of `vcm.noisy_eval` and mixed with
ambient noise from the dataset's own `background_noise` clips (public; never ESC-50) at an SNR measured against the
reverberated speech (wake word + command only, not the silent lead and tail). Every choice is a pure function of the seed.

    uv run python scripts/build_soak_audio.py --out-dir out/soak/holdout-wake-gap-v1 [--split holdout] [--seed 0]

`sessions.json` (and the flat `sessions.csv`) hold the truth for every session: label, slot, where the wake word ends, the gap,
where the command's speech starts and ends (forced alignment of the clean command clip), the RIR index, the noise clip and the SNR.
Use `scripts/soak_run.py` to score a run; the same audio can be replayed on any device.
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

from me2_voicegen.common.augment import apply_rir
from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.vcm.noisy_eval import build_rir_pool_for_seed
from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform
from me2_voicegen.vcm.segment_scorer import force_align
from me2_voicegen.vcm.text import normalize_text, resolve_transcript

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VCM = ROOT / "out/conversions/v2/ai231-v2/manifest.csv"
DEFAULT_WAKE = ROOT / "out/conversions/v2/wakeword-sesame/manifest.csv"
DEFAULT_ALIGN = ROOT / "out/vcm/hybrid-ctcwide-clsxl/ctc-wide/checkpoints/checkpoint.pt"
TAIL_S = 3.5


def _load(path: Path) -> torch.Tensor:
    wave, sr = torchaudio.load(str(path))
    wave = wave.mean(0)
    return torchaudio.functional.resample(wave, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wave


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt().clamp_min(1e-8))


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def speech_bounds(model, extractor, wave: torch.Tensor, transcript: str) -> tuple[float, float] | None:
    """(start_s, end_s) of `transcript` in `wave` by CTC forced alignment, or None if it cannot be aligned."""
    logp = np.asarray(logp_for_waveform(model, extractor, wave, device="cpu"))
    try:
        a = force_align(logp, normalize_text(transcript))
    except Exception:
        return None
    frame_s = wave.numel() / SAMPLE_RATE / logp.shape[0]
    return a.start_frame * frame_s, (a.end_frame + 1) * frame_s


def mix(session: torch.Tensor, active: torch.Tensor, rir: torch.Tensor, noise: torch.Tensor, snr_db: float,
        noise_offset: float) -> torch.Tensor:
    """Reverberate the whole session, then add noise at `snr_db` against the reverberated speech (`active` marks it)."""
    wet = apply_rir(session, rir)
    speech_rms = _rms(wet[active]) if bool(active.any()) else _rms(wet)
    # tile / rotate the noise clip to the session length, then scale it to the target SNR
    if noise.numel() > wet.numel():
        start = int(noise_offset * (noise.numel() - wet.numel()))
        noise = noise[start:start + wet.numel()]
    else:
        noise = torch.roll(noise, -int(noise_offset * noise.numel()))
        noise = noise.repeat(int(np.ceil(wet.numel() / noise.numel())))[: wet.numel()]
    noise = noise * (speech_rms / _rms(noise)) * 10 ** (-snr_db / 20)
    out = wet + noise
    peak = float(out.abs().max())
    return out / peak * 0.98 if peak > 0.98 else out


def build(out_dir: Path, split: str, seed: int, vcm_manifest: Path, wake_manifest: Path, wake_split: str,
          align_checkpoint: Path, rir_pool_size: int, snr_range: tuple[float, float], limit: int | None,
          sample: int | None = None) -> list[dict]:
    rng = random.Random(f"soak-{seed}")
    rows = [r for r in _read(vcm_manifest) if r["split"] == split]
    if sample is not None:  # a seeded random subset that keeps the command / out-of-scope mix
        rows = random.Random(f"soak-sample-{seed}").sample(rows, min(sample, len(rows)))
    rows = rows[:limit]
    # ambient noise: the dataset's own noise-only clips (public); other splits only, so nothing leaks into the test clips
    noises = [r for r in _read(vcm_manifest)
              if r["bucket"] == "silence" and r["source_dataset"] == "background_noise" and r["split"] != split]
    wakes = [r for r in _read(wake_manifest)
             if r["split"] == wake_split and r["label"] == "_wakeword_" and not r["noise_source_file"]]
    if not noises or not wakes:
        raise SystemExit("need background_noise rows in another split and clean wake-word positives")
    rirs = build_rir_pool_for_seed(seed, pool_size=rir_pool_size)
    model, _ = load_checkpoint(align_checkpoint, device="cpu", weights_only=True)
    extractor = LogMelFeatureExtractor()
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    sessions = []
    for i, row in enumerate(rows):
        command = _load(vcm_manifest.parent / row["path"])
        wake_row = rng.choice(wakes)
        wake = _load(wake_manifest.parent / wake_row["path"])
        noise_row = rng.choice(noises)
        noise = _load(vcm_manifest.parent / noise_row["path"])
        lead_s = rng.randint(5, 15) / 10
        gap_s = rng.randint(0, 10) / 10          # 0.0 ... 1.0 s in 0.1 s steps
        snr_db = round(rng.uniform(*snr_range), 1)
        rir_i = rng.randrange(len(rirs))
        noise_offset = rng.random()

        n = lambda s: int(round(s * SAMPLE_RATE))  # noqa: E731
        lead, gap, tail = (torch.zeros(n(s)) for s in (lead_s, gap_s, TAIL_S))
        session = torch.cat([lead, wake, gap, command, tail])
        active = torch.zeros(session.numel(), dtype=torch.bool)
        wake_start = lead.numel()
        command_start = lead.numel() + wake.numel() + gap.numel()
        active[wake_start:wake_start + wake.numel()] = True
        active[command_start:command_start + command.numel()] = True
        mixed = mix(session, active, rirs[rir_i], noise, snr_db, noise_offset)

        is_command = row["bucket"] == "target_commands"
        bounds = speech_bounds(model, extractor, command, resolve_transcript(row)) if is_command else None
        start_s, end_s = bounds if bounds else (0.0, command.numel() / SAMPLE_RATE)
        name = f"{i:04d}_{row['label']}.wav"
        torchaudio.save(str(out_dir / "audio" / name), mixed.unsqueeze(0), SAMPLE_RATE)
        sessions.append({
            "file": f"audio/{name}", "label": row["label"], "is_oos": not is_command, "slot_value": row.get("slot_value", ""),
            "transcript": resolve_transcript(row) if is_command else row.get("transcript", ""),
            "command_filename": row["filename"], "wakeword_filename": wake_row["filename"],
            "noise_filename": noise_row["filename"], "rir_index": rir_i, "snr_db": snr_db,
            "lead_s": lead_s, "gap_s": gap_s,
            "wake_start_s": round(wake_start / SAMPLE_RATE, 4),
            "wake_end_s": round((wake_start + wake.numel()) / SAMPLE_RATE, 4),
            "command_start_s": round(command_start / SAMPLE_RATE, 4),
            "speech_start_s": round(command_start / SAMPLE_RATE + start_s, 4),
            "speech_end_s": round(command_start / SAMPLE_RATE + end_s, 4),
            "aligned": bounds is not None, "duration_s": round(mixed.numel() / SAMPLE_RATE, 4),
        })
    meta = {"syntax": "lead + wakeword + gap(0.0-1.0 s, 0.1 s steps) + command + tail(3.5 s); RIR + dataset ambient noise over the whole session",
            "split": split, "seed": seed, "n": len(sessions), "sample_rate": SAMPLE_RATE, "snr_range_db": list(snr_range),
            "rir_pool_size": rir_pool_size, "vcm_manifest": str(vcm_manifest), "wakeword_manifest": str(wake_manifest),
            "wake_split": wake_split, "align_checkpoint": str(align_checkpoint), "sessions": sessions}
    (out_dir / "sessions.json").write_text(json.dumps(meta, indent=1))
    with (out_dir / "sessions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(sessions[0]))
        w.writeheader()
        w.writerows(sessions)
    return sessions


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", default="holdout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vcm-manifest", type=Path, default=DEFAULT_VCM)
    p.add_argument("--wakeword-manifest", type=Path, default=DEFAULT_WAKE)
    p.add_argument("--wake-split", default="test", help="wake-word positives are drawn from this split of the wake-word manifest")
    p.add_argument("--align-checkpoint", type=Path, default=DEFAULT_ALIGN)
    p.add_argument("--rir-pool-size", type=int, default=200)
    p.add_argument("--snr-min", type=float, default=10.0)
    p.add_argument("--snr-max", type=float, default=25.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample", type=int, default=None, help="use a seeded random subset of this many rows (for a tuning soak from val)")
    a = p.parse_args(argv)
    s = build(a.out_dir, a.split, a.seed, a.vcm_manifest, a.wakeword_manifest, a.wake_split, a.align_checkpoint,
              a.rir_pool_size, (a.snr_min, a.snr_max), a.limit, a.sample)
    gaps = sorted({x["gap_s"] for x in s})
    print(f"wrote {len(s)} sessions to {a.out_dir}; gaps used {gaps[0]}-{gaps[-1]} s ({len(gaps)} distinct); "
          f"{sum(x['is_oos'] for x in s)} out-of-scope; {sum(not x['aligned'] for x in s if not x['is_oos'])} commands not aligned")


if __name__ == "__main__":
    main()
