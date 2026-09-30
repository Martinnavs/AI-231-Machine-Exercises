"""Stage plan: persona sample + job list with a per-intent row cap.

Personas: a seeded random sample of the fil50 persona voices, restricted to voices whose
own split is `train` (the converted clips are train-only, and a voice must never appear in
two splits), stratified proportionally by prompt_source (sapinsapin / references) so the
sample is representative of the eligible pool. Jobs: per intent, at most CAP_PER_INTENT
(clip, persona) pairs, spread evenly across that intent's clips and across personas, so
the added rows do not worsen the VCM class imbalance (COLOR has 8 clips and would
otherwise dominate). `noisy_target` is drawn at the fil50 rate (4,435 noisy siblings per
9,958 clean fil50_persona rows) so collate adds ESC-50 noisy siblings at the same rate.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from random import Random

CAP_PER_INTENT = 150
N_PERSONAS = 50
NOISY_RATE = 4435 / 9958  # measured on out/conversions/v2/optionb-v3-vcmx-fil50 (fil50_persona rows)
JOB_FIELDS = [
    "job_id", "model", "split", "voice_id", "text", "label", "source_row_ref", "noisy_target",
    "path", "duration", "seconds_elapsed", "status", "speech_start_s", "speech_end_s", "source_wav",
]


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    out = {k: math.floor(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def sample_personas(voices: list[dict], n: int = N_PERSONAS, seed: int = 0) -> list[dict]:
    pool = [v for v in voices if v["split"] == "train"]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for v in pool:
        by_src[v["prompt_source"]].append(v)
    if n > len(pool):
        raise ValueError(f"asked for {n} personas but only {len(pool)} train-split voices exist")
    quota = _largest_remainder(n, {k: len(v) for k, v in by_src.items()})
    rng = Random(seed)
    chosen: list[dict] = []
    for src in sorted(by_src):
        chosen += rng.sample(sorted(by_src[src], key=lambda v: v["voice_id"]), quota[src])
    return sorted(chosen, key=lambda v: v["voice_id"])


def plan_jobs(clips: list[dict], personas: list[dict], cap: int = CAP_PER_INTENT, seed: int = 0) -> list[dict]:
    rng = Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_label[c["label"]].append(c)
    jobs: list[dict] = []
    for label in sorted(by_label):
        cl = sorted(by_label[label], key=lambda c: c["clip_id"])
        target = min(cap, len(cl) * len(personas))
        orders = {c["clip_id"]: rng.sample(personas, len(personas)) for c in cl}
        picked: list[tuple[dict, dict]] = []
        i = 0
        while len(picked) < target:
            c = cl[i % len(cl)]
            k = sum(1 for p in picked if p[0] is c)
            picked.append((c, orders[c["clip_id"]][k]))
            i += 1
        for c, p in picked:
            jobs.append({
                "job_id": f"uv_{c['clip_id']}__{p['voice_id']}", "model": "vcm", "split": "train",
                "voice_id": p["voice_id"], "text": c["text"], "label": label, "source_row_ref": c["clip_id"],
                "noisy_target": "1" if rng.random() < NOISY_RATE else "0", "path": "", "duration": "",
                "seconds_elapsed": "", "status": "", "speech_start_s": "", "speech_end_s": "",
                "source_wav": c["path"],
            })
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--voices", type=Path, required=True, help="fil50 refs/voices.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-personas", type=int, default=N_PERSONAS)
    ap.add_argument("--cap", type=int, default=CAP_PER_INTENT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    clips = list(csv.DictReader(a.clips.open(newline="", encoding="utf-8")))
    voices = list(csv.DictReader(a.voices.open(newline="", encoding="utf-8")))
    personas = sample_personas(voices, a.n_personas, a.seed)
    jobs = plan_jobs(clips, personas, a.cap, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "personas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(voices[0].keys()))
        w.writeheader(); w.writerows(personas)
    with (a.out_dir / "jobs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        w.writeheader(); w.writerows(jobs)
    per = defaultdict(int)
    for j in jobs:
        per[j["label"]] += 1
    print(f"personas: {len(personas)} ({sum(p['prompt_source']=='references' for p in personas)} references); "
          f"jobs: {len(jobs)} {dict(per)}; noisy_target=1: {sum(j['noisy_target']=='1' for j in jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
