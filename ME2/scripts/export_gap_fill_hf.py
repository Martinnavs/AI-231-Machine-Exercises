"""Export the 1,849 gap-fill persona clips (all QA outcomes) as the `gap_fill` config of the persona supplement on Hugging Face.

The default `train` split of `martinnavs/ai231-fil-supplemental-data` (`data/train-*`) is left untouched; the clips go to
`gap_fill/train-00000-of-00001.parquet`, which a second config of the card points at. Same first 18 columns as the
published shards, plus `reference_voice`, `voice_accent`, and four columns that make the training selection reproducible:
`job_id`, `qa_passed`, `qa_rescued`, `manifest_split` (the split in `ai231-fil50-supp/manifest.csv`, empty when the clip
was not used).

    PYTHONPATH=src uv run python scripts/export_gap_fill_hf.py                 # dry run: writes the parquet and the card, uploads nothing
    PYTHONPATH=src uv run python scripts/export_gap_fill_hf.py --push          # one commit: parquet + card
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from me2_voicegen.accent_balance.build_ai231_fil50 import rescued_job_ids
from me2_voicegen.accent_balance.plan_gap_jobs import load_variations, phrase_key, spoken

REPO = "martinnavs/ai231-fil-supplemental-data"
PARQUET = "gap_fill/train-00000-of-00001.parquet"
V2 = Path("out/conversions/v2")
CONFIG_BLOCK = "- config_name: gap_fill\n  data_files:\n  - split: train\n    path: gap_fill/train-*\n"
SECTION = """## Gap-fill clips (config `gap_fill`)

```
gap_fill/train-00000-of-00001.parquet   1,849 clips, 17 voices, all QA outcomes
```

The 1,849 clips generated to top thin wordings up to a per-wording target (`accent_balance/plan_gap_jobs.py`, then
`accent_balance/generate.py`, same CosyVoice2 zero-shot cloning from the same 17 references), **including the clips that failed the
transcription check**. They are not in the default `train` split. Extra columns: `job_id`, `qa_passed` (passed the Whisper check),
`qa_rescued` (failed it only because Whisper wrote "100%" for "100 percent"), `manifest_split` (the split the clip got in the
project's training manifest; empty if it was not used). Filter on `qa_passed or qa_rescued` for the clips the project trained on
(`manifest_split` shows the ones it actually picked, capped per wording).

```python
gap = load_dataset("martinnavs/ai231-fil-supplemental-data", "gap_fill", split="train")
```

Sampling is not seeded, so regenerating the same jobs gives the same wording, voice and label but different audio.

"""


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(out: Path) -> Path:
    base = V2 / "ai231-fil50"
    variations = load_variations(Path("raw_datasets/ai231-me2-voice-commands-v2/variations.csv"))
    passed = {r["job_id"] for r in read(base / "qa/qa_pass.csv") if r["passed"] == "True"}
    rescued = rescued_job_ids(base / "qa/reports", base / "qa/shim")
    used = {r["source_relpath"]: r["split"] for r in read(V2 / "ai231-fil50-supp/manifest.csv") if r["source_relpath"].startswith("vcm_train_gap")}
    rows = []
    for g in read(base / "gen_manifest.csv"):
        assert g["status"] == "ok", g["job_id"]
        v = variations.get(phrase_key(g["label"], g["text"]))
        wav = Path(g["path"])
        is_pass, is_rescue = g["job_id"] in passed, g["job_id"] in rescued
        rows.append({
            "audio": {"bytes": wav.read_bytes(), "path": wav.name}, "file": f"audio/{wav.name}", "transcript": spoken(g["text"]),
            "command": g["label"], "variation": v["phrase"] if v else "", "slot_value": (v or {}).get("value", ""), "out_of_scope": 0,
            "bucket": v["phrase"] if v else "", "speaker_id": g["voice_id"], "source": "fil50_persona", "is_synthetic": 1,
            "accent_group": "Synthetic", "numerals": "", "duration_s": float(g["duration"]), "transcript_source": "label",
            "variation_match": "exact" if v else "none",
            "whisper_check": "persona QA pass" if is_pass else "persona QA rescue" if is_rescue else "persona QA fail",
            "whisper_transcript": "", "note": f"fil50 persona gap-fill: zero-shot TTS cloned from references:{g['voice_id'][4:]}.mp3",
            "reference_voice": f"references:{g['voice_id'][4:]}.mp3", "voice_accent": "Filipino",
            "job_id": g["job_id"], "qa_passed": is_pass, "qa_rescued": is_rescue, "manifest_split": used.get(g["job_id"], ""),
        })
    assert len(rows) == 1849, len(rows)
    assert set(used) <= {r["job_id"] for r in rows}, "manifest uses a gap clip that is not in gen_manifest"
    assert all(r["qa_passed"] or r["qa_rescued"] for r in rows if r["manifest_split"]), "a used clip did not pass QA"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out, row_group_size=500)
    n_used, n_ok = sum(bool(r["manifest_split"]) for r in rows), sum(r["qa_passed"] or r["qa_rescued"] for r in rows)
    print(f"{len(rows)} clips: {sum(r['qa_passed'] for r in rows)} passed, {len(rescued)} rescued flags, {n_ok} passed-or-rescued, {n_used} used in the manifest; {out.stat().st_size / 1e6:.1f} MB")
    return out


def card() -> str:
    from huggingface_hub import hf_hub_download
    text = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text(encoding="utf-8")
    assert CONFIG_BLOCK not in text and "config_name: gap_fill" not in text, "card already has the gap_fill config"
    anchor_cfg, anchor_sec = "    path: data/train-*\n", "## Loading\n"
    assert text.count(anchor_cfg) == 1 and text.count(anchor_sec) == 1
    return text.replace(anchor_cfg, anchor_cfg + CONFIG_BLOCK).replace(anchor_sec, SECTION + anchor_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="where the parquet and the new card are written first")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    parquet = build(a.out_dir / PARQUET)
    readme = a.out_dir / "README.md"
    readme.write_text(card(), encoding="utf-8")
    print(f"wrote {parquet} and {readme}")
    if a.push:
        from huggingface_hub import CommitOperationAdd, HfApi
        info = HfApi().create_commit(REPO, repo_type="dataset", operations=[
            CommitOperationAdd(PARQUET, str(parquet)), CommitOperationAdd("README.md", str(readme))],
            commit_message="Add the gap_fill config: 1,849 gap-fill persona clips with QA flags (default split unchanged)")
        print("pushed", info.commit_url)


if __name__ == "__main__":
    main()
