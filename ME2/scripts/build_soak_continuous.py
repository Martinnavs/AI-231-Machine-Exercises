"""Concatenate soak sessions (`scripts/build_soak_audio.py`) into ONE continuous wav with an arbitrary distance between units.

    uv run python scripts/build_soak_continuous.py --sessions out/soak/holdout-wake-gap-v1 [--seed 0] [--min-distance 1 --max-distance 8]

A unit is `wake word + gap + command` plus the first 0.5 s after the command (its reverb tail), cut from the session. Units are placed
in a seeded shuffled order and separated by a random distance of 1.0-8.0 s (steps of 0.1 s) of ambient noise: the previous session's own
noise clip, rolled to a new offset and scaled to the noise level measured in that session's noise-only tail. The file starts with 1 s of
ambient and ends with a 3.5 s ambient tail so the last period can time out.

Writes `continuous.wav` (16 kHz mono) and `continuous.json` (order, distances, every unit's offset in the file and its truth, with times
relative to the unit's start). `scripts/soak_run.py --continuous` streams the file once and scores each unit over its own time range
(unit start up to the next unit's start).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 16000
TAIL_S = 3.5            # tail of each session (build_soak_audio.TAIL_S)
KEEP_AFTER_COMMAND_S = 0.5
LEAD_S, END_TAIL_S = 1.0, 3.5


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt().clamp_min(1e-8))


def _ambient(noise: torch.Tensor, n: int, level: float, rng: random.Random) -> torch.Tensor:
    noise = torch.roll(noise, -int(rng.random() * noise.numel()))
    noise = noise.repeat(-(-n // noise.numel()))[:n]
    return noise * (level / _rms(noise))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--vcm-manifest", type=Path, default=None, help="manifest the noise clips come from (default: the one in sessions.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-distance", type=float, default=1.0)
    ap.add_argument("--max-distance", type=float, default=8.0)
    a = ap.parse_args()
    meta = json.loads((a.sessions / "sessions.json").read_text())
    man = Path(a.vcm_manifest or meta["vcm_manifest"])
    path_of = {r["filename"]: man.parent / r["path"] for r in csv.DictReader(man.open(newline="", encoding="utf-8"))}
    rng = random.Random(f"soak-order-{a.seed}")
    order = list(range(len(meta["sessions"])))
    rng.shuffle(order)
    sr = lambda s: int(round(s * SAMPLE_RATE))  # noqa: E731

    def noise_for(s: dict) -> torch.Tensor:
        w, r = torchaudio.load(str(path_of[s["noise_filename"]]))
        w = w.mean(0)
        return torchaudio.functional.resample(w, r, SAMPLE_RATE) if r != SAMPLE_RATE else w

    pieces, entries, pos = [], [], 0
    first = meta["sessions"][order[0]]
    w0, _ = torchaudio.load(str(a.sessions / first["file"]))
    lead_level = _rms(w0[0][-sr(TAIL_S) + sr(KEEP_AFTER_COMMAND_S):])   # the first session's own noise-only level
    pieces.append(_ambient(noise_for(first), sr(LEAD_S), lead_level, rng))
    pos += sr(LEAD_S)
    prev_noise, prev_level = None, None
    for k, i in enumerate(order):
        s = meta["sessions"][i]
        wave, rate = torchaudio.load(str(a.sessions / s["file"]))
        assert rate == SAMPLE_RATE, rate
        wave = wave[0]
        if k > 0:  # distance of ambient between the previous unit and this one
            dist = rng.randint(round(a.min_distance * 10), round(a.max_distance * 10)) / 10
            pieces.append(_ambient(prev_noise, sr(dist), prev_level, rng))
            pos += sr(dist)
            entries[-1]["distance_to_next_s"] = dist
        else:
            dist = None
        cmd_end = wave.numel() - sr(TAIL_S)
        start, end = sr(s["wake_start_s"]), cmd_end + sr(KEEP_AFTER_COMMAND_S)
        unit = wave[start:end]
        tail_noise_only = wave[end:]                                 # noise-only remainder of the session tail
        prev_level, prev_noise = _rms(tail_noise_only), noise_for(s)
        rel = s["wake_start_s"]
        entries.append({**s, "session_index": i, "order": k, "offset_s": round(pos / SAMPLE_RATE, 4), "n_samples": unit.numel(),
                        "wake_start_s": 0.0, "wake_end_s": round(s["wake_end_s"] - rel, 4),
                        "command_start_s": round(s["command_start_s"] - rel, 4), "speech_start_s": round(s["speech_start_s"] - rel, 4),
                        "speech_end_s": round(s["speech_end_s"] - rel, 4), "unit_end_s": round(unit.numel() / SAMPLE_RATE, 4),
                        "session_speech_end_s": s["speech_end_s"], "distance_from_previous_s": dist, "distance_to_next_s": None})
        pieces.append(unit)
        pos += unit.numel()
    pieces.append(_ambient(prev_noise, sr(END_TAIL_S), prev_level, rng))
    pos += sr(END_TAIL_S)
    audio = torch.cat(pieces)
    for j, e in enumerate(entries):
        e["range_end_s"] = entries[j + 1]["offset_s"] if j + 1 < len(entries) else round(audio.numel() / SAMPLE_RATE, 4)
    torchaudio.save(str(a.sessions / "continuous.wav"), audio.unsqueeze(0), SAMPLE_RATE)
    dists = [e["distance_to_next_s"] for e in entries if e["distance_to_next_s"] is not None]
    out = {k: v for k, v in meta.items() if k != "sessions"} | {
        "order_seed": a.seed, "distance_s_range": [a.min_distance, a.max_distance], "total_s": round(audio.numel() / SAMPLE_RATE, 3),
        "n": len(entries), "unit": "wake word + gap + command + 0.5 s reverb tail; times in each entry are relative to the unit start",
        "sessions": entries}
    (a.sessions / "continuous.json").write_text(json.dumps(out, indent=1))
    print(f"continuous.wav: {len(entries)} units, {audio.numel() / SAMPLE_RATE / 60:.1f} min, {audio.numel() * 2 / 1e6:.0f} MB; "
          f"distances {min(dists)}-{max(dists)} s (mean {sum(dists) / len(dists):.2f})")


if __name__ == "__main__":
    main()
