"""Plan persona-TTS jobs that top every ai231 variation up to a per-variation target.

The fil50 persona clips already on disk cover most of the 93 variations unevenly (2 to 796 clips per phrase), and none
cover the phrasings the updated grammar introduced ("Make a call", "Pause song"). For each variation this counts the
usable existing clips (train split, transcript equal to the variation's phrase) and plans `ceil((target - have) *
overgen)` new jobs, round-robin over the `references` voices. The output is a `jobs.csv` for `accent_balance.generate`.

    uv run python -m me2_voicegen.accent_balance.plan_gap_jobs \
        --variations raw_datasets/ai231-me2-voice-commands-v2/variations.csv \
        --persona-manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
        --voices out/conversions/v2/fil50/refs/voices.csv --target 106 --out out/conversions/v2/ai231-fil50/jobs/jobs.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from random import Random

from me2_voicegen.vcm.optionb.transcript import prepare_ctc_transcript
from me2_voicegen.vcm.text import normalize_text

JOB_FIELDS = ["job_id", "model", "split", "voice_id", "text", "label", "source_row_ref", "noisy_target"]


def spoken(phrase: str) -> str:
    """The phrase as it is spoken ("6:00 AM" is read the same as "6 AM")."""
    return re.sub(r":00\b", "", phrase)


def phrase_key(label: str, phrase: str) -> tuple[str, str]:
    return label, normalize_text(prepare_ctc_transcript(spoken(phrase)))


def load_variations(path: Path) -> dict[tuple[str, str], dict]:
    rows = csv.DictReader(path.open(newline="", encoding="utf-8"))
    return {phrase_key(r["label"], r["phrase"]): {**r, "text": spoken(r["phrase"])} for r in rows}


def count_existing(persona_manifest: Path, variations: dict) -> Counter:
    """Existing clean fil50_persona train clips per variation key (the `_noisy` siblings are not separate clips)."""
    counts: Counter = Counter()
    for r in csv.DictReader(persona_manifest.open(newline="", encoding="utf-8")):
        if r["source_dataset"] != "fil50_persona" or r["split"] != "train" or "_noisy" in r["filename"]:
            continue
        key = phrase_key(r["label"], r["transcript"])
        if key in variations:
            counts[key] += 1
    return counts


def plan(variations: dict, have: Counter, voices: list[str], target: int, overgen: float, seed: int) -> list[dict]:
    rng = Random(seed)
    cycle = list(voices)
    rng.shuffle(cycle)
    jobs: list[dict] = []
    for key, v in variations.items():
        need = max(0, target - have[key])
        for _ in range(math.ceil(need * overgen)):
            n = len(jobs)
            jobs.append({
                "job_id": f"vcm_train_gap{n:06d}", "model": "vcm", "split": "train", "voice_id": cycle[n % len(cycle)],
                "text": v["text"], "label": v["label"], "source_row_ref": "", "noisy_target": "0",
            })
    return jobs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variations", type=Path, required=True)
    p.add_argument("--persona-manifest", type=Path, required=True)
    p.add_argument("--voices", type=Path, required=True, help="build_refs voices.csv; `references` voices are used")
    p.add_argument("--target", type=int, default=106)
    p.add_argument("--overgen", type=float, default=1.3, help="extra jobs to absorb QA rejects")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    variations = load_variations(a.variations)
    have = count_existing(a.persona_manifest, variations)
    voices = [r["voice_id"] for r in csv.DictReader(a.voices.open(newline="", encoding="utf-8"))
              if r["prompt_source"] == "references"]
    jobs = plan(variations, have, voices, a.target, a.overgen, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, JOB_FIELDS)
        w.writeheader()
        w.writerows(jobs)
    short = sum(max(0, a.target - have[k]) for k in variations)
    print(f"{len(variations)} variations, existing usable {sum(min(have[k], a.target) for k in variations)}, "
          f"shortfall {short}, planned {len(jobs)} jobs over {len(voices)} voices -> {a.out}")


if __name__ == "__main__":
    main()
