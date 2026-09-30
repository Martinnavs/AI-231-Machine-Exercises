"""Stages collate + assemble for the user-voice augmentation.

`build`   gen_manifest + qa_pass -> new `user_voice_persona` rows (clean, plus an ESC-50
          noisy sibling for jobs flagged noisy_target, via the SAME `mix_vcm_noise` the
          fil50 collate uses), a `new_rows.csv`, and a small subset manifest
          (`ambient_subset/manifest.csv`) for `dataset_tools.mix_ambient_noise` so the
          ambient overlay runs ONLY on the new rows with the established parameters
          (existing ambient siblings stay byte-identical).
`assemble` existing ambient manifest (all rows preserved, val/test untouched) + new rows
          + the new ambient siblings -> the final manifest, with the repo's fail-loud
          manifest checks and a before/after class-balance report.
Train split only, by construction; every added row is asserted `split == train`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from random import Random

from me2_voicegen.accent_balance.collate import (
    FIL50_SOURCE,
    collate_job_path,
    load_csv,
    load_passing_jobs,
    mix_vcm_noise,
    rebase_path,
    write_csv,
)
from me2_voicegen.vcm.vcmx_merge import (
    MANIFEST_FIELDS as VCM_MANIFEST_FIELDS,
    assign_esc50_splits,
    freesound_id,
    load_esc50_rows,
    run_fail_loud_checks,
)
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError

USER_SOURCE = "user_voice_persona"


def build_new_row(job: dict, out_dir: Path) -> dict:
    from me2_voicegen.accent_balance.qa import prompt_source_of

    if job["split"] != "train":
        raise ManifestValidationError(f"job {job['job_id']}: user-voice rows are train-only, got split={job['split']!r}")
    row = {f: "" for f in VCM_MANIFEST_FIELDS}
    row.update(
        filename=Path(job["path"]).name, path=collate_job_path(job["path"], out_dir), bucket="target_commands",
        label=job["label"], duration=job["duration"], sample_rate="16000", resampled="True",
        source_dataset=USER_SOURCE, source_relpath=job["source_row_ref"], group_id=job["voice_id"],
        split="train", transcript=job["text"], original_dataset=prompt_source_of(job["voice_id"]),
    )
    return row


def build(gen_manifest: Path, qa_pass: Path, noise_root: Path, out_dir: Path, subset_dir: Path, seed: int) -> list[dict]:
    jobs = [j for j in load_passing_jobs(gen_manifest, qa_pass) if j["status"] == "ok"]
    esc = load_esc50_rows(noise_root / "manifest.csv")
    split_by_fsid = assign_esc50_splits(esc)
    train_noise = [r for r in esc if split_by_fsid.get(freesound_id(r["source_file"])) == "train"]
    rng = Random(seed)
    rows: list[dict] = []
    for job in sorted(jobs, key=lambda j: j["job_id"]):
        clean = build_new_row(job, out_dir)
        rows.append(clean)
        if job["noisy_target"] == "1":
            rows.append(mix_vcm_noise(clean, train_noise, noise_root, out_dir, rng))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "new_rows.csv", VCM_MANIFEST_FIELDS, rows)
    sub = [{**r, "path": rebase_path(r["path"], out_dir, subset_dir)} for r in rows]
    write_csv(subset_dir / "manifest.csv", VCM_MANIFEST_FIELDS, sub)
    by = Counter(r["label"] for r in rows if not r["filename"].endswith("_noisy.wav"))
    print(f"new rows: {len(rows)} ({len(jobs)} clean QA-passed jobs, {len(rows) - len(jobs)} noisy siblings); clean per intent {dict(by)}")
    return rows


def _class_counts(rows: list[dict], split: str = "train") -> Counter:
    return Counter(r["label"] for r in rows if r["bucket"] == "target_commands" and r["split"] == split)


def assemble(base_manifest: Path, new_rows_csv: Path, ambient_manifest: Path | None, out_dir: Path) -> Path:
    base_rows = load_csv(base_manifest)
    fields = list(base_rows[0].keys())
    base_dir = base_manifest.parent
    rebased = [{**r, "path": rebase_path(r["path"], base_dir, out_dir)} for r in base_rows]
    new_rows = load_csv(new_rows_csv)  # paths already relative to out_dir
    added = list(new_rows)
    if ambient_manifest is not None:
        amb_dir = ambient_manifest.parent
        for r in load_csv(ambient_manifest):
            if r["source_dataset"] == USER_SOURCE + "_ambient":
                added.append({**r, "path": rebase_path(r["path"], amb_dir, out_dir)})
    for r in added:
        if r["split"] != "train":
            raise ManifestValidationError(f"added row {r['filename']!r} is split {r['split']!r}, must be train")
    final = rebased + [{f: r.get(f, "") for f in fields} for r in added]
    run_fail_loud_checks(final, out_dir)
    if [r for r in final if r["split"] in ("val", "test")] != [r for r in rebased if r["split"] in ("val", "test")]:
        raise ManifestValidationError("val/test rows changed")
    dest = write_csv(out_dir / "manifest.csv", fields, final)
    before, after = _class_counts(base_rows), _class_counts(final)
    lines = ["# user-voice assemble report", f"base rows: {len(base_rows)}; added: {len(added)}; final: {len(final)}", "",
             "| intent | train before | added | train after |", "|---|---|---|---|"]
    for k in sorted(after):
        lines.append(f"| {k} | {before[k]} | {after[k] - before[k]} | {after[k]} |")
    lo, hi = min(after.values()), max(after.values())
    lines += ["", f"train class range before: {min(before.values())}..{max(before.values())}; after: {lo}..{hi} (max/min {hi/lo:.2f} vs {max(before.values())/min(before.values()):.2f})"]
    (out_dir / "assemble_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(after) - 5:]))
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gen-manifest", type=Path, required=True); b.add_argument("--qa-pass", type=Path, required=True)
    b.add_argument("--noise-root", type=Path, required=True); b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--subset-dir", type=Path, required=True); b.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("assemble")
    a.add_argument("--base-manifest", type=Path, required=True); a.add_argument("--new-rows", type=Path, required=True)
    a.add_argument("--ambient-manifest", type=Path, default=None); a.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            build(args.gen_manifest, args.qa_pass, args.noise_root, args.out_dir, args.subset_dir, args.seed)
        else:
            assemble(args.base_manifest, args.new_rows, args.ambient_manifest, args.out_dir)
    except ManifestValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
