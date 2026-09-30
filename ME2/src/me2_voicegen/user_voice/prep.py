"""Stage prep: the user's mp3 recordings -> 16 kHz mono PCM16 wavs + clips.csv.

Expected text and label come from the file name (`"Shut off the lights 2.mp3"` ->
text `"Shut off the lights"`, take 2) and are validated against OPTIONB_GRAMMAR; an
unparseable or out-of-grammar file fails loudly instead of being guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

CLIP_FIELDS = ["clip_id", "text", "label", "slots", "take", "path", "duration"]
_TAKE_RE = re.compile(r"^(?P<text>.*?)(?:\s+(?P<take>\d+))?$")
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


class PrepError(ValueError):
    pass


def parse_clip_name(filename: str) -> tuple[str, int]:
    """'Shut off the lights 2.mp3' -> ('Shut off the lights', 2); no number -> take 1."""
    stem = Path(filename).stem.strip()
    m = _TAKE_RE.match(stem)
    text = m.group("text").strip()
    if not text:
        raise PrepError(f"cannot derive text from file name {filename!r}")
    return text, int(m.group("take") or 1)


def clip_id_for(text: str, take: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"uv_{slug}_{take}"


def label_for(text: str) -> tuple[str, dict]:
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

    hit = OPTIONB_GRAMMAR.accepts(text.lower())
    if not hit:
        raise PrepError(f"{text!r} is not accepted by OPTIONB_GRAMMAR")
    if len({intent for intent, _ in hit}) != 1:
        raise PrepError(f"{text!r} is ambiguous across intents: {hit}")
    return hit[0]


def prepare(raw_dir: Path, out_dir: Path) -> Path:
    import torch
    import torchaudio

    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise PrepError(f"no audio files in {raw_dir}")
    src_dir = out_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for f in files:
        text, take = parse_clip_name(f.name)
        clip_id = clip_id_for(text, take)
        if clip_id in seen:
            raise PrepError(f"duplicate clip id {clip_id!r} (from {f.name!r})")
        seen.add(clip_id)
        label, slots = label_for(text)
        wav, sr = torchaudio.load(str(f))
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = torch.clamp(wav, -1.0, 1.0)
        dest = src_dir / f"{clip_id}.wav"
        torchaudio.save(str(dest), wav, 16000, encoding="PCM_S", bits_per_sample=16)
        rows.append({"clip_id": clip_id, "text": text, "label": label, "slots": json.dumps(slots, sort_keys=True),
                     "take": take, "path": str(dest), "duration": f"{wav.shape[1] / 16000:.3f}"})
    dest_csv = out_dir / "clips.csv"
    with dest_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CLIP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"prepared {len(rows)} clips ({len({r['label'] for r in rows})} intents) -> {dest_csv}")
    return dest_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        prepare(a.raw_dir, a.out_dir)
    except PrepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
