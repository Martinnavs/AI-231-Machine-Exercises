"""The committed training-manifest recipe (`recipes/ai231-fil50-supp/`) is internally consistent and has the documented row counts."""

import csv
from collections import Counter
from pathlib import Path

from me2_voicegen.accent_balance.build_ai231_fil50 import rescued_job_ids

R = Path(__file__).resolve().parents[1] / "recipes/ai231-fil50-supp"


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_final_manifest_row_counts_match_the_docs():
    m = rows(R / "manifest.ai231-fil50-supp.csv")
    assert Counter(r["split"] for r in m) == {"train": 18300, "val": 2717, "test": 7639, "holdout": 202}
    persona = Counter(r["split"] for r in m if r["source_dataset"] == "fil50_persona")
    assert persona == {"train": 4535, "val": 924, "test": 2946}


def test_gap_fill_jobs_generated_and_used_clips_are_consistent():
    jobs, gen = rows(R / "gap-fill/jobs.csv"), rows(R / "gap-fill/gen_manifest.csv")
    # gen_manifest.csv is the two GPU shards concatenated, so compare as sets, not in order
    assert {j["job_id"] for j in jobs} == {g["job_id"] for g in gen} and len(jobs) == len(gen) == 1849
    assert all(g["status"] == "ok" for g in gen)
    passed = {r["job_id"] for r in rows(R / "gap-fill/qa_pass.csv") if r["passed"] == "True"}
    rescued = rescued_job_ids(R / "gap-fill/qa-reports", R / "gap-fill/qa-shim")
    assert len(passed) == 1042 and len(rescued) == 278 and not passed & rescued
    used = {r["source_relpath"] for r in rows(R / "manifest.ai231-fil50-supp.csv") if r["source_relpath"].startswith("vcm_train_gap")}
    assert len(used) == 1157 and used <= passed | rescued


def test_persona_file_map_covers_the_selection_except_four_unpublished_clips():
    fmap = rows(R / "persona-pool/published_file_map.csv")
    assert len(fmap) == 9958 and sum(bool(r["published_file"]) for r in fmap) == 9770
    published = [r["published_file"] for r in fmap if r["published_file"]]
    assert len(set(published)) == len(published) and len({r["sha256"] for r in fmap}) == len(fmap)
    by_name = {r["filename"]: r["published_file"] for r in fmap}
    selected = [r for r in rows(R / "manifest.ai231-fil50.csv") if r["source_dataset"] == "fil50_persona" and not r["source_relpath"].startswith("vcm_train_gap")]
    unpublished = sorted(r["filename"] for r in selected if not by_name[r["filename"].split("_", 2)[2]])
    assert unpublished == ["persona_train_vcm_test_016649.wav", "persona_train_vcm_train_010782.wav", "persona_train_vcm_val_014572.wav", "persona_val_vcm_train_002637.wav"]
