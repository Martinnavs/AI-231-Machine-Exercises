"""Pad and equalize the ai231 training split with fil50 persona clips (the ai231 analog of fil50).

Starts from the converted ai231 manifest (every row of every split is kept) and adds clean persona clips to train, val
and test. Candidates are all clean persona clips on disk (old fil50 manifest, matched to a variation by phrase) plus the
QA-passed clips from `plan_gap_jobs` / `generate`. The persona voices are split between train, val and test (no voice
appears in two splits, as with the real ai231 speakers); within a split each variation gets at most `--cap-<split>`
persona clips, taken round-robin over that split's voices. A variation with fewer clips keeps what it has.

    uv run python -m me2_voicegen.accent_balance.build_ai231_fil50 --base out/conversions/v2/ai231-v2/manifest.csv \
        --persona-manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
        --gen-manifest out/conversions/v2/ai231-fil50/gen_manifest.csv --qa-pass out/conversions/v2/ai231-fil50/qa/qa_pass.csv \
        --variations raw_datasets/ai231-me2-voice-commands-v2/variations.csv --out out/conversions/v2/ai231-fil50/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from me2_voicegen.accent_balance.plan_gap_jobs import load_variations, phrase_key, spoken


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rebased(path: str, base_dir: Path, out_dir: Path) -> str:
    return os.path.relpath((base_dir / path).resolve(), out_dir.resolve())


def rescued_job_ids(reports_dir: Path, shim_dir: Path) -> set[str]:
    """Jobs the QA flagged only because Whisper wrote "100%" where the text says "100 percent".

    The QA scores raw string similarity (0.78 for "Brightness 100%." against "Brightness 100 percent", below the 0.80
    bar), but the audio says the right words. A flagged clip is rescued when its transcript equals the expected text
    after "%" is read as "percent", digits are spelled out and punctuation and case are dropped. Nothing looser.
    """
    def norm(text: str) -> tuple[str, str]:
        return phrase_key("", text.replace("%", " percent"))

    job_by_filename = {}
    for index in shim_dir.glob("*/index.csv"):
        job_by_filename.update({r["filename"]: r["job_id"] for r in _read(index)})
    rescued: set[str] = set()
    for report in reports_dir.glob("*.md"):
        for line in report.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and cells[0] in job_by_filename and cells[4] == "no" and norm(cells[1]) == norm(cells[2]):
                rescued.add(job_by_filename[cells[0]])
    return rescued


def candidates(persona_manifest: Path, gen_manifest: Path | None, qa_pass: Path | None, variations: dict,
               out_dir: Path, rescue: set[str] = frozenset()) -> dict[tuple[str, str], list[dict]]:
    """{variation key: [candidate persona rows]} over every old split, `path` already relative to `out_dir`."""
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in _read(persona_manifest):
        key = phrase_key(r["label"], r["transcript"])
        if r["source_dataset"] != "fil50_persona" or "_noisy" in r["filename"] or key not in variations:
            continue
        pool[key].append({"filename": r["filename"], "path": _rebased(r["path"], persona_manifest.parent, out_dir),
                          "label": r["label"], "duration": r["duration"], "voice": r["group_id"],
                          "source_relpath": r["source_relpath"], "text": r["transcript"]})
    if gen_manifest is not None and qa_pass is not None:
        passed = {r["job_id"] for r in _read(qa_pass) if r["passed"] == "True"} | set(rescue)
        for r in _read(gen_manifest):
            key = phrase_key(r["label"], r["text"])
            if r["status"] == "ok" and r["job_id"] in passed and key in variations:
                pool[key].append({"filename": f"{r['job_id']}.wav", "path": _rebased(r["path"], Path.cwd(), out_dir),
                                  "label": r["label"], "duration": r["duration"], "voice": r["voice_id"],
                                  "source_relpath": r["job_id"], "text": spoken(r["text"])})
    return pool


def assign_voices(voice_sizes: dict[str, int], n_test: int, n_val: int, test_share: float = 0.30,
                  val_share: float = 0.08) -> dict[str, str]:
    """{voice: split}. Picks the n_test / n_val voices whose clip counts come closest to the test / val share of all
    clips (exhaustive search, ties broken by voice name), so the split is a pure function of the clip counts."""
    voices = sorted(voice_sizes)
    total = sum(voice_sizes.values())
    best = None
    for test in itertools.combinations(voices, n_test):
        rest = [v for v in voices if v not in test]
        t = sum(voice_sizes[v] for v in test)
        for val in itertools.combinations(rest, n_val):
            err = abs(t - test_share * total) + abs(sum(voice_sizes[v] for v in val) - val_share * total)
            if best is None or err < best[0]:
                best = (err, test, val)
    _, test, val = best
    return {v: "test" if v in test else "val" if v in val else "train" for v in voices}


def pick(pool: list[dict], target: int, rng: Random) -> list[dict]:
    """Up to `target` rows, round-robin over voices (voice order and in-voice order shuffled by `rng`)."""
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["filename"]):
        by_voice[r["voice"]].append(r)
    voices = sorted(by_voice)
    rng.shuffle(voices)
    for v in voices:
        rng.shuffle(by_voice[v])
    chosen: list[dict] = []
    while len(chosen) < target and any(by_voice.values()):
        for v in voices:
            if by_voice[v] and len(chosen) < target:
                chosen.append(by_voice[v].pop())
    return chosen


def build(base: Path, persona_manifest: Path, gen_manifest: Path | None, qa_pass: Path | None, variations_csv: Path,
          out: Path, caps: dict[str, int], seed: int, rescue: set[str] = frozenset(), n_test_voices: int = 5,
          n_val_voices: int = 2) -> dict:
    variations = load_variations(variations_csv)
    out_dir = out.parent
    pool = candidates(persona_manifest, gen_manifest, qa_pass, variations, out_dir, rescue)
    sizes = Counter(c["voice"] for rows in pool.values() for c in rows)
    split_of = assign_voices(sizes, n_test_voices, n_val_voices)
    added: list[dict] = []
    for split, cap in caps.items():
        for key, v in variations.items():
            rows = pick([c for c in pool[key] if split_of[c["voice"]] == split], cap, Random(f"{seed}-{split}-{key}"))
            for c in rows:
                added.append({
                    "filename": f"persona_{split}_{c['filename']}", "path": c["path"], "bucket": "target_commands",
                    "label": c["label"], "duration": c["duration"], "sample_rate": 16000, "resampled": False,
                    "source_dataset": "fil50_persona", "source_relpath": c["source_relpath"], "group_id": c["voice"],
                    "split": split, "transcript": c["text"], "variation": v["phrase"], "slot_value": spoken(v["value"]),
                    "variation_match": "exact",
                })
    base_rows = _read(base)
    for r in base_rows:
        r["path"] = _rebased(r["path"], base.parent, out_dir)
    fields = list(base_rows[0].keys()) + [k for k in added[0] if k not in base_rows[0]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(base_rows + added)
    return {"voice_split": {s: sorted(v for v, sp in split_of.items() if sp == s) for s in caps},
            "persona_added": dict(Counter(r["split"] for r in added)),
            "real_in_scope": dict(Counter(r["split"] for r in base_rows if r["bucket"] == "target_commands"))}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--persona-manifest", type=Path, required=True)
    p.add_argument("--gen-manifest", type=Path)
    p.add_argument("--qa-pass", type=Path)
    p.add_argument("--variations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cap-train", type=int, default=50, help="max persona clips per variation in train")
    p.add_argument("--cap-val", type=int, default=12)
    p.add_argument("--cap-test", type=int, default=35)
    p.add_argument("--test-voices", type=int, default=5)
    p.add_argument("--val-voices", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qa-reports", type=Path, help="QA reports dir; with --qa-shim, rescues percent/digit-format false flags")
    p.add_argument("--qa-shim", type=Path)
    a = p.parse_args(argv)
    rescue = rescued_job_ids(a.qa_reports, a.qa_shim) if a.qa_reports and a.qa_shim else set()
    print(f"rescued {len(rescue)} false QA flags")
    print(build(a.base, a.persona_manifest, a.gen_manifest, a.qa_pass, a.variations, a.out,
                {"train": a.cap_train, "val": a.cap_val, "test": a.cap_test}, a.seed, rescue, a.test_voices, a.val_voices))


if __name__ == "__main__":
    main()
