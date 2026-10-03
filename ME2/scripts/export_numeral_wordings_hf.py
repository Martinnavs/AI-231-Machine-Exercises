"""Publish the four persona clips with spelled-out numbers that the training manifest uses but the persona dataset left out.

`martinnavs/ai231-fil-supplemental-data` left out the pool's spelled-out-number wordings, yet four of them were selected into `ai231-fil50-supp`
("wake me up at six am", "set an alarm for eight am", two "...twenty two degrees"). They go in a separate config, `numeral_wordings`
(`numeral_wordings/train-00000-of-00001.parquet`), so the default `train` split and the `gap_fill` config are unchanged. Same first 18 columns as the
published shards, plus `reference_voice`, `voice_accent`, `source_filename` (the pool's original file name, which `published_file_map.csv` points at).

    PYTHONPATH=src uv run python scripts/export_numeral_wordings_hf.py --pool-manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
        --qa-pass recipes/ai231-fil50-supp/persona-pool/qa_pass.csv --variations raw_datasets/ai231-me2-voice-commands-v2/variations.csv --out-dir <scratch>   # dry run
    ... --push                                                                                                                                              # one commit
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from me2_voicegen.accent_balance.plan_gap_jobs import load_variations, phrase_key, spoken

REPO = "martinnavs/ai231-fil-supplemental-data"
PARQUET = "numeral_wordings/train-00000-of-00001.parquet"
CLIPS = ["vcm_train_010782.wav", "vcm_test_016649.wav", "vcm_val_014572.wav", "vcm_train_002637.wav"]
CONFIG_BLOCK = "- config_name: numeral_wordings\n  data_files:\n  - split: train\n    path: numeral_wordings/train-*\n"
SECTION = """## Numeral wordings (config `numeral_wordings`)

```
numeral_wordings/train-00000-of-00001.parquet   4 clips, 4 voices
```

Four persona clips whose wording has a spelled-out number ("wake me up at six am", "set an alarm for eight am", "set the temperature to twenty
two degrees", "change the temperature to twenty two degrees"). The default `train` split leaves this kind of wording out, but the project's
training manifest uses these four, so they are published here to make that manifest rebuildable. They passed the same transcription check and are
not in the default `train` split. `source_filename` is the clip's name in the project's persona pool.

```python
num = load_dataset("martinnavs/ai231-fil-supplemental-data", "numeral_wordings", split="train")
```

"""


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(a: argparse.Namespace, out: Path) -> Path:
    variations = load_variations(a.variations)
    qa = {r["job_id"]: r["passed"] == "True" for r in read(a.qa_pass)}
    pool = {r["filename"]: r for r in read(a.pool_manifest) if r["filename"] in CLIPS}
    assert set(pool) == set(CLIPS), set(CLIPS) - set(pool)
    rows = []
    for name in CLIPS:
        r = pool[name]
        v = variations[phrase_key(r["label"], r["transcript"])]
        assert qa[Path(name).stem], f"{name} did not pass QA"
        wav = a.pool_manifest.parent / r["path"]
        ref = f"references:{r['group_id'][4:]}.mp3"
        rows.append({
            "audio": {"bytes": wav.read_bytes(), "path": name}, "file": f"audio/{name}", "transcript": spoken(r["transcript"]),
            "command": r["label"], "variation": v["phrase"], "slot_value": v.get("value", ""), "out_of_scope": 0, "bucket": v["phrase"],
            "speaker_id": r["group_id"], "source": "fil50_persona", "is_synthetic": 1, "accent_group": "Synthetic", "numerals": "",
            "duration_s": float(r["duration"]), "transcript_source": "label", "variation_match": "exact", "whisper_check": "persona QA pass",
            "whisper_transcript": "", "note": f"fil50 persona: zero-shot TTS cloned from {ref}, passed transcription QA",
            "reference_voice": ref, "voice_accent": "Filipino", "source_filename": name,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"{len(rows)} clips, {out.stat().st_size / 1e3:.0f} kB: " + "; ".join(f"{r['source_filename']} {r['transcript']!r} {r['speaker_id']}" for r in rows))
    return out


def card() -> str:
    from huggingface_hub import hf_hub_download
    text = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text(encoding="utf-8")
    assert "config_name: numeral_wordings" not in text, "card already has the numeral_wordings config"
    anchor_cfg, anchor_sec = "    path: gap_fill/train-*\n", "## Loading\n"
    assert text.count(anchor_cfg) == 1 and text.count(anchor_sec) == 1
    return text.replace(anchor_cfg, anchor_cfg + CONFIG_BLOCK).replace(anchor_sec, SECTION + anchor_sec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-manifest", type=Path, required=True)
    ap.add_argument("--qa-pass", type=Path, required=True)
    ap.add_argument("--variations", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    parquet = build(a, a.out_dir / PARQUET)
    readme = a.out_dir / "README.md"
    readme.write_text(card(), encoding="utf-8")
    if a.push:
        from huggingface_hub import CommitOperationAdd, HfApi
        info = HfApi().create_commit(REPO, repo_type="dataset", operations=[
            CommitOperationAdd(PARQUET, str(parquet)), CommitOperationAdd("README.md", str(readme))],
            commit_message="Add the numeral_wordings config: 4 persona clips with spelled-out numbers used by the training manifest (default split unchanged)")
        print("pushed", info.commit_url)


if __name__ == "__main__":
    main()
