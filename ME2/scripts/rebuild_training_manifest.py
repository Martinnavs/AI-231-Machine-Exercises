"""Rebuild the `ai231-fil50-supp` training manifest from the public data and the committed recipe, and check it against the committed manifest.

Inputs: the ai231 dataset (`--ai231`, downloaded from Hugging Face; holds `variations.csv` and `supplemental_synth/`), and the persona
supplement `martinnavs/ai231-fil-supplemental-data` (default train split for the persona pool, `numeral_wordings` config for four spelled-out-number clips, `gap_fill` config for the gap-fill clips;
`--persona-dir` / `--numerals-dir` / `--gap-fill-dir` use local copies instead of downloading). Everything else is in `recipes/ai231-fil50-supp/`.

    PYTHONPATH=src uv run python scripts/rebuild_training_manifest.py --ai231 raw_datasets/ai231-me2-voice-commands-v2 --out out/conversions/v2/rebuilt

The selection is a pure function of the recipe CSVs and was verified identical to the committed manifest; every selected clip is public, so the rebuilt
manifest must equal the committed one. (A selected clip that were unpublished would be dropped after selection, not before, because dropping it first
would change the voice split and every pick; the check below would then report the difference.) Audio is extracted from the parquet shards into `<out>/audio/` for the clips the manifest uses.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from me2_voicegen.accent_balance.build_ai231_fil50 import main as build_base

REPO = "martinnavs/ai231-fil-supplemental-data"
RECIPE = Path(__file__).resolve().parents[1] / "recipes/ai231-fil50-supp"
COMPARE = ["filename", "bucket", "label", "duration", "split", "source_dataset", "source_relpath", "group_id", "transcript", "variation", "slot_value"]


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def shards(local: Path | None, pattern: str, repo_glob: str) -> list[Path]:
    if local is not None:
        found = sorted(local.glob(pattern))
        assert found, f"no {pattern} under {local}"
        return found
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[repo_glob]))
    return sorted(root.glob(repo_glob))


def extract(parquets: list[Path], wanted: dict[str, Path]) -> set[str]:
    """Write the clips whose `file` basename is in `wanted` (published name -> destination); returns the names written."""
    done: set[str] = set()
    for shard in parquets:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=["audio", "file"]):
            for r in batch.to_pylist():
                name = Path(r["file"]).name
                if name in wanted and name not in done:
                    wanted[name].parent.mkdir(parents=True, exist_ok=True)
                    wanted[name].write_bytes(r["audio"]["bytes"])
                    done.add(name)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai231", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ai231-v2-manifest", type=Path, help="an existing import_ai231 output; default: run the importer into <out>/ai231-v2")
    ap.add_argument("--persona-dir", type=Path, help="local copy of the persona dataset's train-*.parquet shards")
    ap.add_argument("--numerals-dir", type=Path, help="local copy of the numeral_wordings train-*.parquet shard")
    ap.add_argument("--gap-fill-dir", type=Path, help="local copy of the gap_fill train-*.parquet shard")
    ap.add_argument("--no-audio", action="store_true", help="manifests only: do not extract audio")
    a = ap.parse_args()
    out = a.out.resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    base = a.ai231_v2_manifest
    if base is None:
        subprocess.run([sys.executable, "-m", "me2_voicegen.vcm.optionb.import_ai231", "--src", str(a.ai231), "--out", str(out / "ai231-v2")], check=True)
        base = out / "ai231-v2/manifest.csv"

    # point the persona pool and the gap-fill jobs at the public audio (absolute paths; the builders rebase them)
    fmap = {r["filename"]: r["published_file"] for r in read(RECIPE / "persona-pool/published_file_map.csv")}
    pool = read(RECIPE / "manifest.optionb-v3-vcmx-fil50.csv")
    for r in pool:
        pub = fmap.get(r["filename"])
        r["path"] = str(out / "audio/persona" / pub) if pub else str(out / "audio/persona/UNPUBLISHED" / r["filename"])
    write(work / "persona_manifest.csv", pool, list(pool[0].keys()))
    gen = read(RECIPE / "gap-fill/gen_manifest.csv")
    for g in gen:
        g["path"] = str(out / "audio/gap_fill" / f"{g['job_id']}.wav")
    write(work / "gen_manifest.csv", gen, list(gen[0].keys()))

    build_base(["--base", str(base), "--persona-manifest", str(work / "persona_manifest.csv"), "--gen-manifest", str(work / "gen_manifest.csv"),
                "--qa-pass", str(RECIPE / "gap-fill/qa_pass.csv"), "--qa-reports", str(RECIPE / "gap-fill/qa-reports"),
                "--qa-shim", str(RECIPE / "gap-fill/qa-shim"), "--variations", str(a.ai231 / "variations.csv"), "--out", str(out / "ai231-fil50/manifest.csv")])

    # drop the selected clips that were never published
    rows = read(out / "ai231-fil50/manifest.csv")
    dropped = [r for r in rows if "UNPUBLISHED" in r["path"]]
    kept = [r for r in rows if "UNPUBLISHED" not in r["path"]]
    write(out / "ai231-fil50/manifest.csv", kept, list(rows[0].keys()))
    print(f"dropped {len(dropped)} selected clips that were never published" + (f": {[r['filename'] for r in dropped]}" if dropped else ""))

    if not a.no_audio:
        base_dir = out / "ai231-fil50"
        used = {Path((base_dir / r["path"]).resolve()) for r in kept if r["source_dataset"] == "fil50_persona"}
        persona_want = {p.name: p for p in used if p.parent.name == "persona"}
        gap_want = {p.name: p for p in used if p.parent.name == "gap_fill"}
        got = extract(shards(a.persona_dir, "train-*.parquet", "data/train-*.parquet") + shards(a.numerals_dir, "train-*.parquet", "numeral_wordings/train-*.parquet"), persona_want)
        got |= extract(shards(a.gap_fill_dir, "train-*.parquet", "gap_fill/train-*.parquet"), gap_want)
        missing = (set(persona_want) | set(gap_want)) - got
        assert not missing, f"{len(missing)} clips not found in the published shards, e.g. {sorted(missing)[:3]}"
        print(f"extracted {len(got)} clips to {out / 'audio'}")

    subprocess.run([sys.executable, str(Path(__file__).with_name("build_ai231_fil50_supp.py")), "--base", str(out / "ai231-fil50/manifest.csv"),
                    "--supplemental", str(a.ai231 / "supplemental_synth"), "--out", str(out / "ai231-fil50-supp")], check=True)

    # compare with the committed manifest: identical except for the dropped clips
    rebuilt = {tuple(r[c] for c in COMPARE) for r in read(out / "ai231-fil50-supp/manifest.csv")}
    committed = {tuple(r[c] for c in COMPARE) for r in read(RECIPE / "manifest.ai231-fil50-supp.csv")}
    gone = {tuple(r[c] for c in COMPARE) for r in dropped}
    assert rebuilt | gone == committed and not rebuilt & gone and not rebuilt - committed, (
        f"rebuilt differs from the committed manifest beyond the dropped clips: {len(rebuilt - committed)} extra, {len(committed - rebuilt - gone)} missing")
    print(f"OK: {len(rebuilt)} rows = the committed manifest ({len(committed)}) minus the {len(gone)} unpublished clips")


if __name__ == "__main__":
    main()
