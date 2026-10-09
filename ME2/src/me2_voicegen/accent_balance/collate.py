"""T5 (.scratch/accent-balance-fil50/tickets/00-RECAP.md): collate the
QA-passed persona clips into the final 50/50 VCM + wakeword manifests.

**Row accounting (why `select_for_target_rows` exists).** Both base
manifests already count a clean clip and its noise-mixed sibling as two
SEPARATE `target_commands`/`_wakeword_` rows (measured: every `optionb`
noisy row has a clean sibling, 8,993/8,993). `plan_jobs.count_*_deficit`'s
deficit is therefore a ROW count, but one passing synthesis job yields one
clean row plus, for jobs T2 flagged `noisy_target=1`, one additional
noise-mixed row -- so picking exactly `deficit` jobs would overshoot by
however many of them are noisy-flagged. `select_for_target_rows` greedily
takes jobs (seeded order) until the row budget is exactly spent, dropping
the noisy pairing off the one job that would overshoot it by exactly 1 row
rather than under/over-shooting the 50/50 target.

**Path rebasing.** New VCM audio lives under `out/conversions/v2/fil50/`
(T3's output root), a sibling of the base manifest's own directory -- every
row's `path` is rewritten relative to the NEW manifest's directory
(`rebase_path`), covering both pre-existing rows (whose relative paths may
resolve differently at a different depth) and new rows.

**Fail-loud checks, not a relaxed QA pass.** If a split doesn't have enough
QA-passed candidates to reach parity, this raises with the shortfall count
instead of silently filling with fewer rows or lower-quality clips (RECAP
T5's explicit instruction) -- the fix is to rerun T3 with a larger
`--overgen`, not to loosen anything here.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import zlib
from pathlib import Path
from random import Random

import soundfile as sf
import torch
import torchaudio

from me2_voicegen.accent_balance.plan_jobs import (
    SPLITS,
    count_vcm_deficit,
    count_ww_deficit,
)
from me2_voicegen.accent_balance.qa import prompt_source_of
from me2_voicegen.common.augment import SNR_MAX_DB, SNR_MIN_DB, apply_noise
from me2_voicegen.vcm.vcmx_merge import (
    MANIFEST_FIELDS as VCM_MANIFEST_FIELDS,
    VCMXMergeError,
    assign_esc50_splits,
    freesound_id,
    load_esc50_rows,
    run_fail_loud_checks as vcm_run_fail_loud_checks,
)
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError, probe_wav, resample_to_16k_in_place
from me2_voicegen.wakeword.build_dataset import MANIFEST_FIELDS as WW_MANIFEST_FIELDS

logger = logging.getLogger(__name__)

FIL50_SOURCE = "fil50_persona"
FIL50_SOURCE_NOISY_WW = "fil50_persona_noisy"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_passing_jobs(gen_manifest_path: Path, qa_pass_path: Path) -> list[dict]:
    gen_rows = {r["job_id"]: r for r in load_csv(gen_manifest_path)}
    qa_rows = load_csv(qa_pass_path)
    passing = []
    for qa_row in qa_rows:
        if qa_row["passed"] != "True":
            continue
        gen_row = gen_rows.get(qa_row["job_id"])
        if gen_row is None:
            raise ManifestValidationError(f"qa_pass.csv references unknown job_id {qa_row['job_id']!r}, not in gen_manifest.csv")
        passing.append(gen_row)
    return passing


def rebase_path(old_path: str, old_dir: Path, new_dir: Path) -> str:
    """Recompute `old_path` (as it appears in a manifest row, relative to
    `old_dir`) as a path relative to `new_dir` instead. `Path.__truediv__`
    discards `old_dir` if `old_path` is already absolute, so this also
    correctly handles the absolute-or-CWD-relative paths T3's gen_manifest
    rows carry (both processes share the same CWD, the repo root, per the
    shell entry point's own `cd` at start -- see collate_job_path)."""
    resolved = (old_dir / old_path).resolve()
    return os.path.relpath(resolved, new_dir.resolve())


def collate_job_path(job_path: str, new_dir: Path) -> str:
    """Same as `rebase_path`, specialized for a T3 gen_manifest job's own
    `path` field, which is always either absolute or relative to the
    current process's CWD (never to some other manifest's directory)."""
    return rebase_path(job_path, Path("."), new_dir)


# ---------------------------------------------------------------------------
# Row-budget selection (see module docstring)
# ---------------------------------------------------------------------------


def select_for_target_rows(candidates: list[dict], target_rows: int, seed: int) -> list[tuple[dict, bool]]:
    """Returns `[(job, include_noisy_sibling), ...]` summing to EXACTLY
    `target_rows` total output rows (1 per job, +1 more for each
    `include_noisy_sibling=True`). Raises if `candidates` can't reach
    `target_rows` even taking every candidate clean-only."""
    if target_rows <= 0:
        return []
    if len(candidates) < target_rows:
        raise ManifestValidationError(
            f"only {len(candidates)} QA-passed candidates available but {target_rows} rows are needed -- "
            f"rerun T3/generate.py with a larger --overgen for this split"
        )

    rng = Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    selected: list[tuple[dict, bool]] = []
    total_rows = 0
    for job in shuffled:
        if total_rows >= target_rows:
            break
        wants_noisy = job.get("noisy_target") == "1"
        rows_if_taken = 2 if wants_noisy else 1
        if total_rows + rows_if_taken > target_rows:
            selected.append((job, False))
            total_rows += 1
        else:
            selected.append((job, wants_noisy))
            total_rows += rows_if_taken

    if total_rows < target_rows:
        raise ManifestValidationError(
            f"could only reach {total_rows}/{target_rows} rows from {len(candidates)} candidates -- "
            f"rerun T3/generate.py with a larger --overgen for this split"
        )
    return selected


# ---------------------------------------------------------------------------
# VCM
# ---------------------------------------------------------------------------


def build_vcm_new_row(job: dict, out_dir: Path) -> dict:
    row = {field: "" for field in VCM_MANIFEST_FIELDS}
    row.update(
        {
            "filename": Path(job["path"]).name,
            "path": collate_job_path(job["path"], out_dir),
            "bucket": "target_commands",
            "label": job["label"],
            "duration": job["duration"],
            "sample_rate": "16000",
            "resampled": "True",
            "source_dataset": FIL50_SOURCE,
            "source_relpath": job["source_row_ref"],
            "group_id": job["voice_id"],
            "split": job["split"],
            "transcript": job["text"],
            "original_dataset": prompt_source_of(job["voice_id"]),
        }
    )
    return row


def mix_vcm_noise(clean_row: dict, esc50_rows_for_split: list[dict], noise_root: Path, out_dir: Path, rng: Random) -> dict:
    if not esc50_rows_for_split:
        raise ManifestValidationError(f"no ESC-50 noise clips available for split {clean_row['split']!r}")
    noise_row = esc50_rows_for_split[rng.randrange(len(esc50_rows_for_split))]
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB)

    source_path = (out_dir / clean_row["path"]).resolve()
    noise_path = (noise_root / "audio" / noise_row["filename"]).resolve()
    if not source_path.is_file():
        raise ManifestValidationError(f"clean clip missing on disk: {source_path}")
    if not noise_path.is_file():
        raise ManifestValidationError(f"noise clip missing on disk: {noise_path}")

    source_wave, _sr = torchaudio.load(str(source_path))
    noise_wave, _sr = torchaudio.load(str(noise_path))
    mixed = torch.clamp(apply_noise(source_wave[0], noise_wave[0], snr_db), -1.0, 1.0)

    dest = out_dir / "audio_noisy" / f"{Path(clean_row['filename']).stem}_noisy.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), mixed.numpy(), 16000, subtype="PCM_16")
    probe = probe_wav(dest)
    resampled = clean_row["resampled"] == "True"
    if (probe.sample_rate, probe.channels, probe.sampwidth) != (16000, 1, 2):
        resample_to_16k_in_place(dest)
        resampled = True

    row = dict(clean_row)
    row.update(
        {
            "filename": dest.name,
            "path": str(dest.relative_to(out_dir)),
            "duration": f"{probe_wav(dest).duration:.6f}",
            "resampled": "True" if resampled else "False",
        }
    )
    return row


def collate_vcm(
    *,
    passing_jobs: list[dict],
    base_manifest_path: Path,
    noise_root: Path,
    out_manifest_path: Path,
    seed: int,
) -> Path:
    out_dir = out_manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = base_manifest_path.parent
    base_rows = load_csv(base_manifest_path)

    counts = count_vcm_deficit(base_manifest_path)
    esc50_rows = load_esc50_rows(noise_root / "manifest.csv")
    esc50_split_by_fsid = assign_esc50_splits(esc50_rows)
    esc50_by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for row in esc50_rows:
        split = esc50_split_by_fsid.get(freesound_id(row["source_file"]))
        if split in esc50_by_split:
            esc50_by_split[split].append(row)

    vcm_jobs = [j for j in passing_jobs if j["model"] == "vcm"]

    rebased_base_rows = []
    for row in base_rows:
        new_row = dict(row)
        new_row["path"] = rebase_path(row["path"], base_dir, out_dir)
        rebased_base_rows.append(new_row)

    new_rows: list[dict] = []
    rng = Random(seed)
    for split in SPLITS:
        deficit = counts[split]["deficit"]
        candidates = [j for j in vcm_jobs if j["split"] == split]
        selection = select_for_target_rows(candidates, deficit, seed + zlib.crc32(split.encode()) % 10_000)
        for job, include_noisy in selection:
            clean_row = build_vcm_new_row(job, out_dir)
            new_rows.append(clean_row)
            if include_noisy:
                new_rows.append(mix_vcm_noise(clean_row, esc50_by_split[split], noise_root, out_dir, rng))

    all_rows = rebased_base_rows + new_rows
    vcm_run_fail_loud_checks(all_rows, out_dir)
    assert_fifty_fifty(all_rows, bucket_field="bucket", bucket_value="target_commands", split_field="split", is_filipino=_is_filipino_vcm_row)
    assert_base_rows_preserved(base_rows, rebased_base_rows, ignore_fields={"path"})

    return write_csv(out_manifest_path, VCM_MANIFEST_FIELDS, all_rows)


def _is_filipino_vcm_row(row: dict) -> bool:
    from me2_voicegen.accent_balance.plan_jobs import FILIPINO_VCM_SPEAKER_IDS

    if row["source_dataset"] == FIL50_SOURCE:
        return True
    return row["source_dataset"] == "optionb" and row["group_id"] in FILIPINO_VCM_SPEAKER_IDS


# ---------------------------------------------------------------------------
# Wakeword
# ---------------------------------------------------------------------------


def build_ww_new_row(job: dict, out_dir: Path) -> dict:
    row = {field: "" for field in WW_MANIFEST_FIELDS}
    row.update(
        {
            "filename": Path(job["path"]).name,
            "path": collate_job_path(job["path"], out_dir),
            "label": "_wakeword_",
            "duration": job["duration"],
            "sample_rate": "16000",
            "resampled": "True",
            "source_dataset": FIL50_SOURCE,
            "source_relpath": job["source_row_ref"],
            "group_id": job["voice_id"],
            "split": job["split"],
            "ref_voice": job["voice_id"],
            "noise_source_file": "",
            "snr_db": "",
            "speech_start_s": job.get("speech_start_s", ""),
            "speech_end_s": job.get("speech_end_s", ""),
        }
    )
    return row


def mix_ww_noise(clean_row: dict, noise_rows: list[dict], noise_root: Path, out_dir: Path, rng: Random) -> dict:
    if not noise_rows:
        raise ManifestValidationError("no ESC-50 noise clips available")
    noise_row = noise_rows[rng.randrange(len(noise_rows))]
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB)

    source_path = (out_dir / clean_row["path"]).resolve()
    noise_path = (noise_root / "audio" / noise_row["filename"]).resolve()
    if not source_path.is_file():
        raise ManifestValidationError(f"clean clip missing on disk: {source_path}")
    if not noise_path.is_file():
        raise ManifestValidationError(f"noise clip missing on disk: {noise_path}")

    source_wave, _sr = torchaudio.load(str(source_path))
    noise_wave, _sr = torchaudio.load(str(noise_path))
    mixed = torch.clamp(apply_noise(source_wave[0], noise_wave[0], snr_db), -1.0, 1.0)

    dest = out_dir / "audio_noisy" / f"{Path(clean_row['filename']).stem}_noisy.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), mixed.numpy(), 16000, subtype="PCM_16")
    probe = probe_wav(dest)
    resampled = True
    if (probe.sample_rate, probe.channels, probe.sampwidth) != (16000, 1, 2):
        resample_to_16k_in_place(dest)
        resampled = True

    row = dict(clean_row)
    row.update(
        {
            "filename": dest.name,
            "path": str(dest.relative_to(out_dir)),
            "duration": f"{probe_wav(dest).duration:.6f}",
            "resampled": "True" if resampled else "False",
            "source_dataset": FIL50_SOURCE_NOISY_WW,
            "noise_source_file": noise_row["filename"],
            "snr_db": f"{snr_db:.2f}",
        }
    )
    return row


def collate_wakeword(
    *,
    passing_jobs: list[dict],
    base_manifest_path: Path,
    noise_root: Path,
    out_manifest_path: Path,
    seed: int,
) -> Path:
    out_dir = out_manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = base_manifest_path.parent
    base_rows = load_csv(base_manifest_path)

    counts = count_ww_deficit(base_manifest_path)
    esc50_rows = load_esc50_rows(noise_root / "manifest.csv")

    ww_jobs = [j for j in passing_jobs if j["model"] == "wakeword"]

    rebased_base_rows = []
    for row in base_rows:
        new_row = dict(row)
        new_row["path"] = rebase_path(row["path"], base_dir, out_dir)
        rebased_base_rows.append(new_row)

    new_rows: list[dict] = []
    rng = Random(seed)
    for split in SPLITS:
        deficit = counts[split]["deficit"]
        candidates = [j for j in ww_jobs if j["split"] == split]
        selection = select_for_target_rows(candidates, deficit, seed + zlib.crc32(split.encode()) % 10_000)
        for job, include_noisy in selection:
            clean_row = build_ww_new_row(job, out_dir)
            new_rows.append(clean_row)
            if include_noisy:
                new_rows.append(mix_ww_noise(clean_row, esc50_rows, noise_root, out_dir, rng))

    all_rows = rebased_base_rows + new_rows
    assert_group_id_split_disjoint(all_rows)
    assert_fifty_fifty(all_rows, bucket_field=None, bucket_value=None, split_field="split", is_filipino=_is_filipino_ww_row, label_field="label", label_value="_wakeword_")
    assert_base_rows_preserved(base_rows, rebased_base_rows, ignore_fields={"path"})

    return write_csv(out_manifest_path, WW_MANIFEST_FIELDS, all_rows)


def _is_filipino_ww_row(row: dict) -> bool:
    import re

    if row["source_dataset"] in (FIL50_SOURCE, FIL50_SOURCE_NOISY_WW):
        return True
    return bool(re.match(r"^(tagalog|ilonggo)", row.get("ref_voice") or ""))


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def assert_fifty_fifty(
    rows: list[dict],
    *,
    bucket_field: str | None,
    bucket_value: str | None,
    split_field: str,
    is_filipino,
    label_field: str | None = None,
    label_value: str | None = None,
    tolerance: float = 0.01,
) -> None:
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for row in rows:
        if bucket_field and row.get(bucket_field) != bucket_value:
            continue
        if label_field and row.get(label_field) != label_value:
            continue
        if row[split_field] in by_split:
            by_split[row[split_field]].append(row)

    report = {}
    for split, split_rows in by_split.items():
        if not split_rows:
            continue
        n_fil = sum(1 for r in split_rows if is_filipino(r))
        share = n_fil / len(split_rows)
        report[split] = (n_fil, len(split_rows), share)
        if abs(share - 0.5) > tolerance:
            raise ManifestValidationError(
                f"split {split!r}: Filipino share is {share:.1%} ({n_fil}/{len(split_rows)}), not within "
                f"{tolerance:.0%} of 50% -- {report}"
            )
    logger.info("50/50 check passed: %s", report)


def assert_group_id_split_disjoint(rows: list[dict]) -> None:
    """Cross-split speaker-leak guard. `ref_`-prefixed group ids carry the
    Phase-1 scoped exception: references voices are usable in every split
    (their timbre already spans every split via converted wakeword
    positives), so they may legitimately appear in more than one split.
    EVERY other group id (sNN, fsc_NN, anything else) still raises."""
    by_group: dict[str, set[str]] = {}
    for row in rows:
        gid = row.get("group_id")
        if not gid or gid.startswith("ref_"):
            continue
        by_group.setdefault(gid, set()).add(row["split"])
    violations = {gid: splits for gid, splits in by_group.items() if len(splits) > 1}
    if violations:
        raise ManifestValidationError(f"group_id spans more than one split: {violations}")


def assert_base_rows_preserved(original_rows: list[dict], rebased_rows: list[dict], ignore_fields: set[str]) -> None:
    if len(original_rows) != len(rebased_rows):
        raise ManifestValidationError(f"base row count changed: {len(original_rows)} -> {len(rebased_rows)}")
    for orig, rebased in zip(original_rows, rebased_rows):
        for field in orig:
            if field in ignore_fields:
                continue
            if orig[field] != rebased.get(field):
                raise ManifestValidationError(f"base row {orig.get('filename')!r} field {field!r} changed: {orig[field]!r} -> {rebased.get(field)!r}")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen-manifest", type=Path, required=True)
    parser.add_argument("--qa-pass", type=Path, required=True)
    parser.add_argument("--vcm-manifest", type=Path, required=True)
    parser.add_argument("--wakeword-manifest", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vcm-out", type=Path, required=True)
    parser.add_argument("--wakeword-out", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        passing_jobs = load_passing_jobs(args.gen_manifest, args.qa_pass)
        vcm_dest = collate_vcm(
            passing_jobs=passing_jobs, base_manifest_path=args.vcm_manifest, noise_root=args.noise_root,
            out_manifest_path=args.vcm_out, seed=args.seed,
        )
        print(f"wrote {vcm_dest}")
        ww_dest = collate_wakeword(
            passing_jobs=passing_jobs, base_manifest_path=args.wakeword_manifest, noise_root=args.noise_root,
            out_manifest_path=args.wakeword_out, seed=args.seed,
        )
        print(f"wrote {ww_dest}")
    except (ManifestValidationError, VCMXMergeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
