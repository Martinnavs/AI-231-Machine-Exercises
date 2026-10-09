"""Stage `mix`/`finalize` CLI for the ambient-noise-overlay feature
(feature-engineering/ambient-noise-overlay/SPEC.md).

Materializes ASR-gated ambient-noise siblings for one model's dataset:

  stage mix      plan jobs (common.ambient_mix.plan_jobs) over the eligible
                 rows of a base manifest, mix every planned attempt to a
                 staging dir, and stage the QA inputs: a v1-pair symlink dir
                 per gate unit ("<expected text> - <rowid>-a<attempt>.wav",
                 the naming simple-audio-transcriber understands) and a
                 free-decode copy dir for wakeword `_unknown_` rows.
                 The transcriber itself runs externally between stages (this
                 module never imports or shells out to it -- the
                 accent_balance stage-3 convention); the exact commands to
                 run are printed.

  stage finalize read the transcriber reports (v1-pair units via
                 `build_sanitized_dataset.parse_report`; `_unknown_` rows via
                 a decisions csv `filename,decoded_text` produced by the
                 plain-transcribe path) and apply the gate: v1-pair rows pass
                 iff unflagged, `_unknown_` rows pass iff the decode does NOT
                 contain the wake word (sesame/sesames/sesame's,
                 word-boundary). Missing reports/decisions never pass.
                 Kept mixes are promoted to the final audio dir (atomic
                 staged rename, the mix_background_noise pattern) and a
                 derived manifest (base rows + ambient rows) plus summary.md
                 (pass rates, realized SNR histogram, clipping rate, license
                 block) are written.

Eligibility: VCM = `bucket == "target_commands"` with a non-empty transcript;
wakeword = `label` in {_wakeword_, _unknown_}; both exclude rows whose
`source_dataset` already ends in `_noisy` or `_ambient` (existing ESC-50
mixes and prior ambient runs stay untouched).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import torchaudio

from me2_voicegen.common import ambient_mix
from me2_voicegen.common.ambient_mix import (
    AmbientNoiseError,
    ambient_filename,
    ambient_filename_from,
    job_key,
    load_chunk_pool,
    mix_one,
    plan_jobs,
    v1_pair_violations,
)
from me2_voicegen.dataset_tools.build_sanitized_dataset import parse_report

QA_REPO = Path.home() / "simple-audio-transcriber"
QA_THRESHOLD = 0.80
CLIP_LEVEL = 0.999

QA_PLAN_FIELDS = [
    "key",
    "unit",
    "group_id",
    "filename",
    "split",
    "attempt",
    "wav",
    "expected_text",
    "gate_mode",
    "chunk_id",
    "chunk_source_file",
    "snr_db",
    "clipped",
]

TRIGGER_RE = re.compile(r"\bsesames?\b")
_NOISEY_SUFFIXES = ("_noisy", "_ambient")


def _already_noised(row: dict) -> bool:
    """Rows that already carry noise are never re-mixed (SPEC edge E: the
    existing ESC-50/~30 dB rows stay untouched as the faint anchor): the
    wakeword signals them via `_noisy` source_dataset subsets, Option B via
    the `_noisy` filename stem (its source_dataset stays `optionb`), and
    prior ambient runs via the `_ambient` source_dataset suffix."""
    if row.get("source_dataset", "").endswith(_NOISEY_SUFFIXES):
        return True
    return Path(row.get("filename", "")).stem.endswith("_noisy")

_UNSAFE_SLUG_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: str) -> str:
    slug = _UNSAFE_SLUG_CHARS.sub("-", value).strip("-")
    if not slug:
        raise AmbientNoiseError(f"cannot derive a filename-safe slug from {value!r}")
    return slug


def unit_for(model: str) -> str:
    return "vcm" if model == "vcm" else "wakeword_sesame"


def row_id(row: dict) -> str:
    """(group_id, filename) is the pair both manifests already guarantee
    unique; the shim/decisions filename is built from it."""
    return _slug(f"{row['group_id']}__{Path(row['filename']).stem}")


def shims_name(expected_text: str, row: dict, attempt: int) -> str:
    return f"{expected_text} - {row_id(row)}-a{attempt}.wav"


def free_decode_name(row: dict, attempt: int) -> str:
    return f"{row_id(row)}-a{attempt}.wav"


def read_manifest(manifest_path: Path) -> tuple[list[str], list[dict]]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise AmbientNoiseError(f"manifest is empty: {manifest_path}")
    return fieldnames, rows


def filter_eligible(model: str, rows: list[dict]) -> list[dict]:
    eligible = []
    for row in rows:
        if _already_noised(row):
            continue
        if model == "vcm":
            if row.get("bucket") == "target_commands" and str(row.get("transcript", "")).strip():
                eligible.append(row)
        elif model == "wakeword":
            if row.get("label") in ("_wakeword_", "_unknown_"):
                eligible.append(row)
        else:
            raise AmbientNoiseError(f"unknown model {model!r}")
    return sorted(eligible, key=lambda r: (r.get("group_id", ""), r.get("filename", "")))


def _plan_stats(jobs, rows: list[dict]) -> None:
    selected = {(j.source_row["group_id"], j.source_row["filename"]) for j in jobs}
    snrs = [j.snr_db for j in jobs]
    print(f"eligible rows: {len(rows)}; selected: {len(selected)}; planned jobs: {len(jobs)}")
    if snrs:
        print(
            f"planned SNR: min={min(snrs):.1f} max={max(snrs):.1f} mean={sum(snrs)/len(snrs):.1f} dB"
        )
    per_split = Counter(j.source_row["split"] for j in jobs)
    print(f"planned jobs per split: {dict(sorted(per_split.items()))}")


def stage_mix(
    *,
    model: str,
    manifest_path: Path,
    corpus_root: Path,
    out_root: Path,
    p_mix: float,
    seed: int,
    snr_min_db: float = ambient_mix.SNR_MIN_DB,
    snr_max_db: float = ambient_mix.SNR_MAX_DB,
    max_rows: int | None = None,
    dry_run: bool = False,
) -> Path | None:
    _, all_rows = read_manifest(manifest_path)
    eligible = filter_eligible(model, all_rows)
    if max_rows is not None:
        eligible = eligible[:max_rows]
    if not eligible:
        raise AmbientNoiseError(f"no eligible rows in {manifest_path} for model {model!r}")

    splits = sorted({row["split"] for row in eligible})
    pools = {split: load_chunk_pool(corpus_root, split) for split in splits}
    chunk_source = _corpus_chunk_sources(corpus_root)
    jobs = plan_jobs(
        model,
        eligible,
        pools,
        p_mix=p_mix,
        snr_min_db=snr_min_db,
        snr_max_db=snr_max_db,
        seed=seed,
    )
    violations = v1_pair_violations(jobs)
    if violations:
        raise AmbientNoiseError(
            f"{len(violations)} v1-pair-unsafe transcript(s) in the plan -- not rewritten:\n  "
            + "\n  ".join(violations)
        )
    _plan_stats(jobs, eligible)
    if dry_run:
        print("dry-run: nothing written")
        return None

    staging = out_root / ".ambient.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "audio").mkdir(parents=True)
    shim_dir = staging / "qa" / "shim" / unit_for(model)
    free_dir = staging / "qa" / "free_decode"
    if any(j.expected_text for j in jobs):
        shim_dir.mkdir(parents=True)
    if any(not j.expected_text for j in jobs):
        free_dir.mkdir(parents=True)

    audio_root = manifest_path.parent
    plan_rows: list[dict] = []
    for job in jobs:
        wav_rel = f"audio/{ambient_filename(job)}"
        wav_path = staging / wav_rel
        source_path = audio_root / job.source_row["path"]
        mix_one(job, source_path, wav_path)
        wave, _ = torchaudio.load(str(wav_path))
        clipped = "True" if float(wave.abs().max()) >= CLIP_LEVEL else "False"
        key = job_key(job)
        if job.expected_text:
            link = shim_dir / shims_name(job.expected_text, job.source_row, job.attempt)
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(wav_path.resolve())
            gate_mode, unit = "v1_pair", unit_for(model)
        else:
            dest = free_dir / free_decode_name(job.source_row, job.attempt)
            shutil.copyfile(wav_path, dest)
            gate_mode, unit = "free_decode", ""
        plan_rows.append(
            {
                "key": key,
                "unit": unit,
                "group_id": job.source_row.get("group_id", ""),
                "filename": job.source_row.get("filename", ""),
                "split": job.source_row["split"],
                "attempt": str(job.attempt),
                "wav": wav_rel,
                "expected_text": job.expected_text,
                "gate_mode": gate_mode,
                "chunk_id": job.chunk.chunk_id + ".wav",
                "chunk_source_file": chunk_source.get(job.chunk.chunk_id + ".wav", ""),
                "snr_db": f"{job.snr_db:.2f}",
                "clipped": clipped,
            }
        )

    qa_plan = staging / "qa_plan.csv"
    with qa_plan.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QA_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan_rows)

    n_v1 = sum(1 for r in plan_rows if r["gate_mode"] == "v1_pair")
    n_free = sum(1 for r in plan_rows if r["gate_mode"] == "free_decode")
    print(f"staged {len(plan_rows)} mixes -> {staging / 'audio'}")
    if n_v1:
        print(f"NEXT (v1-pair, {n_v1} files) -- run from the transcriber's own repo/venv:")
        print(
            f"  cd {QA_REPO} && uv run python -m audio_transcript_parser.qa "
            f"{shim_dir} --naming v1-pair --threshold {QA_THRESHOLD} --device cuda "
            f"--out-dir <reports-dir>"
        )
    if n_free:
        print(f"NEXT (free-decode, {n_free} files) -- plain-transcribe {free_dir}, then write "
              f"a decisions csv (filename,decoded_text) for --negatives-decisions")
    return qa_plan


def _corpus_chunk_sources(corpus_root: Path) -> dict[str, str]:
    """chunk filename -> raw source_file, read once (the mix loop touches
    every job; per-job manifest re-reads would be quadratic)."""
    manifest_path = corpus_root / "manifest.csv"
    if not manifest_path.is_file():
        raise AmbientNoiseError(f"corpus manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return {row["filename"]: row["source_file"] for row in csv.DictReader(f)}


def load_qa_plan(staging: Path) -> list[dict]:
    qa_plan = staging / "qa_plan.csv"
    if not qa_plan.is_file():
        raise AmbientNoiseError(f"qa_plan.csv not found (run --stage mix first): {qa_plan}")
    with qa_plan.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def v1_pair_passed(reports_dir: Path, plan_rows: list[dict]) -> dict[str, bool]:
    """A v1-pair row passes iff its unit's report exists and its shim
    filename is not flagged. Missing report = failure (never pass)."""
    flagged_by_unit: dict[str, set[str]] = {}
    for report in sorted(reports_dir.glob("*.md")) if reports_dir.is_dir() else []:
        unit, _source, flagged = parse_report(report)
        flagged_by_unit[unit] = flagged
    passed: dict[str, bool] = {}
    for row in plan_rows:
        if row["gate_mode"] != "v1_pair":
            continue
        flagged = flagged_by_unit.get(row["unit"])
        if flagged is None:
            passed[row["key"]] = False
            continue
        source = {"group_id": row["group_id"], "filename": row["filename"]}
        name = shims_name(row["expected_text"], source, int(row["attempt"]))
        passed[row["key"]] = name not in flagged
    return passed


def decodes_as_wakeword(text: str) -> bool:
    normalized = text.lower().replace("'", "")
    return bool(TRIGGER_RE.search(normalized))


def negative_passed(decisions_path: Path | None, plan_rows: list[dict]) -> dict[str, bool]:
    """A `_unknown_` row passes iff a decode exists for it and does NOT
    contain the wake word. Missing decode = failure (never pass)."""
    decisions: dict[str, str] = {}
    if decisions_path is not None:
        if not decisions_path.is_file():
            raise AmbientNoiseError(f"negatives decisions csv not found: {decisions_path}")
        with decisions_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                decisions[row["filename"]] = row["decoded_text"]
    passed: dict[str, bool] = {}
    for row in plan_rows:
        if row["gate_mode"] != "free_decode":
            continue
        name = free_decode_name({"group_id": row["group_id"], "filename": row["filename"]}, int(row["attempt"]))
        text = decisions.get(name)
        passed[row["key"]] = text is not None and not decodes_as_wakeword(text)
    return passed


def ambient_row(base_row: dict, plan_row: dict, path_prefix: str) -> dict:
    row = dict(base_row)
    row["filename"] = ambient_filename_from(
        plan_row["group_id"], plan_row["filename"], int(plan_row["attempt"]), plan_row["chunk_id"]
    )
    row["path"] = f"{path_prefix}/{row['filename']}" if path_prefix else row["filename"]
    row["source_dataset"] = base_row["source_dataset"] + ambient_mix.AMBIENT_SUFFIX
    row["noise_source_file"] = plan_row["chunk_source_file"]
    row["noise_chunk_id"] = plan_row["chunk_id"]
    row["snr_db"] = plan_row["snr_db"]
    return row


def stage_finalize(
    *,
    model: str,
    manifest_path: Path,
    staging: Path,
    reports_dir: Path,
    negatives_decisions: Path | None,
    audio_dir: Path,
    manifest_out: Path,
    summary_out: Path,
    seed: int,
    p_mix: float,
    snr_min_db: float = ambient_mix.SNR_MIN_DB,
    snr_max_db: float = ambient_mix.SNR_MAX_DB,
) -> tuple[Path, Path]:
    plan_rows = load_qa_plan(staging)
    passed = {}
    passed.update(v1_pair_passed(reports_dir, plan_rows))
    passed.update(negative_passed(negatives_decisions, plan_rows))
    kept = [r for r in plan_rows if passed.get(r["key"], False)]
    for r in kept:
        if not (staging / r["wav"]).is_file():
            raise AmbientNoiseError(f"kept mix missing on disk: {staging / r['wav']}")

    base_fieldnames, base_rows = read_manifest(manifest_path)
    new_cols = [c for c in ("noise_source_file", "noise_chunk_id", "snr_db") if c not in base_fieldnames]
    fieldnames = base_fieldnames + new_cols
    by_key = {(r.get("group_id", ""), r.get("filename", "")): r for r in base_rows}

    base_root = manifest_path.parent.resolve()
    manifest_parent = manifest_out.parent.resolve()
    audio_dir_resolved = audio_dir.resolve()
    try:
        path_prefix = audio_dir_resolved.relative_to(manifest_parent).as_posix()
    except ValueError as exc:
        raise AmbientNoiseError(
            f"audio dir {audio_dir_resolved} must live under the manifest's parent {manifest_parent}"
        ) from exc

    def _rebase(p: str) -> str:
        # Base rows' relative paths resolve against the BASE manifest's dir
        # (e.g. vcm_balanced's local audio_noisy/ rows in the fil50 base).
        # Re-express them against the derived manifest's dir so the derived
        # manifest is self-consistent wherever it lives; paths that already
        # escape the base dir (../..) normalize to the same target.
        if not p or p.startswith("/"):
            return p
        return os.path.relpath(os.path.normpath(os.path.join(str(base_root), p)), str(manifest_parent))

    out_rows: list[dict] = []
    for base_row in base_rows:
        row = {k: base_row.get(k, "") for k in fieldnames}
        row["path"] = _rebase(row["path"])
        out_rows.append(row)
    for r in kept:
        base_row = by_key.get((r["group_id"], r["filename"]))
        if base_row is None:
            raise AmbientNoiseError(f"plan row has no matching base row: {r['key']}")
        out_rows.append(ambient_row(base_row, r, path_prefix))

    # Atomic audio promotion: stage copies, then rename (mix_background_noise pattern).
    final_staging = audio_dir_resolved.parent / f".{audio_dir_resolved.name}.final"
    if final_staging.exists():
        shutil.rmtree(final_staging)
    final_staging.mkdir(parents=True)
    for r in kept:
        dest = final_staging / Path(r["wav"]).name
        shutil.copyfile(staging / r["wav"], dest)
    if audio_dir_resolved.exists():
        shutil.rmtree(audio_dir_resolved)
    final_staging.rename(audio_dir_resolved)

    # Loud check (after promotion, before the manifest is published): every
    # derived row must resolve to a real file from the derived manifest's dir.
    # Catches base rows whose local paths break once the manifest moves to a
    # sibling dir (a silent data loss at train time otherwise -- the
    # DataLoader only hits them mid-epoch).
    missing = [r["path"] for r in out_rows if r["path"] and not os.path.exists(os.path.join(str(manifest_parent), r["path"]))]
    if missing:
        raise AmbientNoiseError(
            f"{len(missing)} derived-manifest path(s) do not resolve under {manifest_parent} (first: {missing[:5]})"
        )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(out_rows)

    summary_path = write_summary(
        summary_out,
        model=model,
        manifest_path=manifest_path,
        n_base=len(base_rows),
        plan_rows=plan_rows,
        kept=kept,
        seed=seed,
        p_mix=p_mix,
        snr_min_db=snr_min_db,
        snr_max_db=snr_max_db,
    )
    print(f"kept {len(kept)}/{len(plan_rows)} mixes -> {audio_dir_resolved}")
    print(f"derived manifest: {manifest_out} ({len(out_rows)} rows)")
    print(f"summary: {summary_path}")
    return manifest_out, summary_path


def write_summary(
    out_path: Path,
    *,
    model: str,
    manifest_path: Path,
    n_base: int,
    plan_rows: list[dict],
    kept: list[dict],
    seed: int,
    p_mix: float,
    snr_min_db: float,
    snr_max_db: float,
) -> Path:
    lines = [
        f"# Ambient-noise overlay -- {model} summary",
        "",
        f"Base manifest: `{manifest_path}` ({n_base} rows). seed={seed}, p_mix={p_mix}, "
        f"SNR uniform [{snr_min_db}, {snr_max_db}] dB, 2 attempts per selected row, "
        "every stored mix ASR-gated (simple-audio-transcriber, v1-pair threshold "
        f"{QA_THRESHOLD} / wake-word trigger check for `_unknown_`).",
        "",
        "## Gate results",
        "",
        f"- planned mixes: {len(plan_rows)}; stored: {len(kept)} "
        f"({len(kept) / len(plan_rows):.1%} pass rate)" if plan_rows else "- no mixes planned",
    ]
    kept_keys = {k["key"] for k in kept}
    per_attempt: dict[str, list[bool]] = {"1": [], "2": []}
    per_split: dict[str, list[bool]] = {}
    for r in plan_rows:
        per_attempt[r["attempt"]].append(r["key"] in kept_keys)
        per_split.setdefault(r["split"], []).append(r["key"] in kept_keys)
    for attempt in ("1", "2"):
        results = per_attempt[attempt]
        if results:
            lines.append(
                f"- attempt {attempt}: {sum(results)}/{len(results)} passed "
                f"({sum(results) / len(results):.1%})"
            )
    if per_split:
        lines.append(
            "- per split (stored/planned): "
            + ", ".join(f"{s} {sum(v)}/{len(v)}" for s, v in sorted(per_split.items()))
        )

    lines += ["", "## Realized SNR histogram (stored mixes, 5 dB bins)", ""]
    bins = Counter()
    for r in kept:
        snr = float(r["snr_db"])
        lo = int(max(snr_min_db, (snr // 5) * 5))
        bins[(lo, min(lo + 5, snr_max_db))] += 1
    if bins:
        for (lo, hi), count in sorted(bins.items()):
            bar = "#" * count
            lines.append(f"- [{lo:+.0f}, {hi:+.0f}) dB: {count:5d} {bar}")
    else:
        lines.append("- (no stored mixes)")

    clipped_total = sum(1 for r in kept if r["clipped"] == "True")
    lines += [
        "",
        "## Clipping",
        "",
        f"- stored mixes clipped to ±1.0: {clipped_total}/{len(kept)} "
        f"({clipped_total / len(kept):.1%})" if kept else "- no stored mixes",
        "",
        "## Chunk usage (source file -> stored mixes)",
        "",
    ]
    usage = Counter(r["chunk_source_file"] for r in kept)
    for source_file, count in sorted(usage.items()):
        lines.append(f"- {source_file}: {count}")

    lines += [
        "",
        "## License -- read before using this subset",
        "",
        "The ambient noise mixed into this subset is Creative Commons licensed "
        "(YouTube source). The exact per-file CC variant is to be confirmed at "
        "attribution time; until then this subset is handled with the same "
        "isolation + note discipline as the ESC-50 (CC-BY-NC-SA-4.0) rows in "
        "this tree: it lives in its own subset/derived manifest, and every "
        "checkpoint trained on a manifest containing these rows must carry a "
        "license note naming this corpus (see out/conversions/v2/README.md's "
        "background_noise section for the precedent). Any file later determined "
        "to be CC-ND (no derivatives) must be excluded from the corpus and this "
        "subset rebuilt -- mixing is a derivative work.",
        "",
        "Per-file attribution (fill at attribution time):",
        "",
    ]
    for source_file in sorted({r["chunk_source_file"] for r in kept} | _corpus_source_files()):
        lines.append(f"- {source_file}: <uploader> -- <video URL> [CC variant unverified]")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _corpus_source_files() -> set[str]:
    # Best-effort full file list for the attribution stubs; empty if the corpus
    # manifest is not at the default location relative to this module.
    candidate = Path(__file__).resolve().parents[3] / "out" / "conversions" / "v2" / "ambient_noise" / "manifest.csv"
    if not candidate.is_file():
        return set()
    with candidate.open(newline="", encoding="utf-8") as f:
        return {row["source_file"] for row in csv.DictReader(f)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=["vcm", "wakeword"])
    parser.add_argument("--stage", required=True, choices=["mix", "finalize"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path("out/conversions/v2/ambient_noise"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--negatives-decisions", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None, help="finalize: final audio destination")
    parser.add_argument("--manifest-out", type=Path, default=None, help="finalize: derived manifest path")
    parser.add_argument("--summary-out", type=Path, default=None, help="finalize: summary.md path")
    parser.add_argument("--p-mix", type=float, default=ambient_mix.P_MIX_DEFAULT)
    parser.add_argument("--snr-min-db", type=float, default=ambient_mix.SNR_MIN_DB)
    parser.add_argument("--snr-max-db", type=float, default=ambient_mix.SNR_MAX_DB)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="cap eligible rows (smoke/pilot runs)")
    parser.add_argument("--dry-run", action="store_true", help="mix stage: plan and report only, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    staging = args.out_root / ".ambient.staging"
    try:
        if args.stage == "mix":
            stage_mix(
                model=args.model,
                manifest_path=args.manifest,
                corpus_root=args.corpus_root,
                out_root=args.out_root,
                p_mix=args.p_mix,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
                seed=args.seed,
                max_rows=args.max_rows,
                dry_run=args.dry_run,
            )
        else:
            if args.reports_dir is None or args.audio_dir is None or args.manifest_out is None or args.summary_out is None:
                raise AmbientNoiseError(
                    "finalize requires --reports-dir, --audio-dir, --manifest-out, --summary-out"
                )
            stage_finalize(
                model=args.model,
                manifest_path=args.manifest,
                staging=staging,
                reports_dir=args.reports_dir,
                negatives_decisions=args.negatives_decisions,
                audio_dir=args.audio_dir,
                manifest_out=args.manifest_out,
                summary_out=args.summary_out,
                seed=args.seed,
                p_mix=args.p_mix,
                snr_min_db=args.snr_min_db,
                snr_max_db=args.snr_max_db,
            )
    except (AmbientNoiseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
