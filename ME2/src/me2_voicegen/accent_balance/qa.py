"""T4 (.scratch/accent-balance-fil50/tickets/00-RECAP.md): QA the generated
clips with `~/simple-audio-transcriber`'s `audio_transcript_parser.qa`.

Two subcommands, run either side of the actual transcriber invocation (the
shell script runs that CLI in between, from its own repo/uv env -- this
module never imports or shells out to it):

  build-shim   T3's `gen_manifest.csv` (status=ok rows) -> one flat symlink
               dir per `(model, prompt_source)` unit, named
               `"<text> - <job_id>.wav"` (the `v1-pair` naming
               `simple-audio-transcriber` understands), plus that unit's own
               `index.csv` (filename -> job_id) so `parse` can map a report's
               flagged bare filenames back to job_ids without re-parsing the
               v1-pair name (text may itself contain characters that would
               make that ambiguous, which is exactly why build-shim validates
               and rejects unsafe text up front rather than rewriting it).

  parse        Reads every `<reports-dir>/*.md` report
               (`dataset_tools.build_sanitized_dataset.parse_report`) plus
               each unit's `index.csv`, and writes `qa_pass.csv`
               (job_id,passed,flag_reason) covering every job in
               `gen_manifest.csv` -- a `status=error` (failed synthesis) row
               is `passed=False` with reason `generation_error`, never
               silently dropped. Also writes `qa_summary.md`, the per-voice
               and per-(model,prompt_source) pass-rate table the pilot report
               needs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

from me2_voicegen.dataset_tools.build_sanitized_dataset import parse_report
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError

logger = logging.getLogger(__name__)

INDEX_FIELDS = ["filename", "job_id"]
QA_PASS_FIELDS = ["job_id", "passed", "flag_reason"]

# " - " is the v1-pair separator (`_resolve_v1_pair` splits the stem on the
# FIRST occurrence); "/" and "\" would escape the symlink's own directory.
# Rejected outright, never silently rewritten (RECAP T4).
_UNSAFE_TEXT_MARKERS = (" - ", "/", "\\")


def unit_name(model: str, prompt_source: str) -> str:
    return f"{model}_{prompt_source}"


def prompt_source_of(voice_id: str) -> str:
    """voice_id -> prompt_source, from T1 build_refs.py's own naming
    convention (`fsc_<speaker_id>` / `ref_<stem>`) -- avoids threading a
    separate join against voices.csv through every downstream stage just for
    this one field."""
    if voice_id.startswith("fsc_"):
        return "sapinsapin"
    if voice_id.startswith("ref_"):
        return "references"
    raise ManifestValidationError(f"voice_id {voice_id!r} doesn't match the fsc_*/ref_* convention build_refs.py writes")


def load_gen_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_shim_text(text: str, job_id: str) -> None:
    if not text.strip():
        raise ManifestValidationError(f"job {job_id}: empty text, cannot build a v1-pair filename")
    for marker in _UNSAFE_TEXT_MARKERS:
        if marker in text:
            raise ManifestValidationError(
                f"job {job_id}: text {text!r} contains {marker!r}, unsafe for v1-pair naming "
                f"(breaks the v1-pair split or escapes the shim directory) -- fix the source text, "
                f"this is not silently rewritten"
            )


def build_shim(rows: list[dict], shim_dir: Path) -> dict[str, Path]:
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        prompt_source = prompt_source_of(row["voice_id"])
        by_unit[unit_name(row["model"], prompt_source)].append(row)

    unit_dirs: dict[str, Path] = {}
    for unit, unit_rows in sorted(by_unit.items()):
        unit_dir = shim_dir / unit
        unit_dir.mkdir(parents=True, exist_ok=True)
        index_rows = []
        for row in unit_rows:
            job_id, text = row["job_id"], row["text"]
            validate_shim_text(text, job_id)
            filename = f"{text} - {job_id}.wav"
            link_path = unit_dir / filename
            target = Path(row["path"]).resolve()
            if not target.is_file():
                raise ManifestValidationError(f"job {job_id}: generated wav missing on disk: {target}")
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(target)
            index_rows.append({"filename": filename, "job_id": job_id})

        with (unit_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(index_rows)
        unit_dirs[unit] = unit_dir

    return unit_dirs


def parse_reports(gen_manifest_path: Path, shim_dir: Path, reports_dir: Path) -> list[dict]:
    """Returns qa_pass rows covering every job in gen_manifest.csv."""
    gen_rows = load_gen_manifest(gen_manifest_path)
    reasons: dict[str, str] = {}  # job_id -> flag_reason ("" means pass)
    covered: set[str] = set()

    report_paths = sorted(reports_dir.glob("*.md")) if reports_dir.is_dir() else []
    for report_path in report_paths:
        unit, _source, flagged_filenames = parse_report(report_path)
        index_path = shim_dir / unit / "index.csv"
        if not index_path.is_file():
            raise ManifestValidationError(
                f"{report_path}: report unit {unit!r} has no matching {index_path} "
                f"(shim-dir mismatch between build-shim and parse, or a stale report)"
            )
        with index_path.open(newline="", encoding="utf-8") as f:
            for idx_row in csv.DictReader(f):
                job_id = idx_row["job_id"]
                covered.add(job_id)
                reasons[job_id] = "qa_flagged" if idx_row["filename"] in flagged_filenames else ""

    rows: list[dict] = []
    for row in gen_rows:
        job_id = row["job_id"]
        if row["status"] != "ok":
            rows.append({"job_id": job_id, "passed": "False", "flag_reason": "generation_error"})
            continue
        if job_id not in covered:
            logger.warning("job %s: status=ok but not covered by any parsed QA report", job_id)
            rows.append({"job_id": job_id, "passed": "False", "flag_reason": "missing_qa_report"})
            continue
        reason = reasons[job_id]
        rows.append({"job_id": job_id, "passed": "False" if reason else "True", "flag_reason": reason})

    return rows


def write_qa_pass_csv(out_path: Path, rows: list[dict]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QA_PASS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_qa_summary(gen_manifest_path: Path, qa_pass_rows: list[dict], out_path: Path) -> Path:
    gen_rows = {r["job_id"]: r for r in load_gen_manifest(gen_manifest_path)}
    pass_by_id = {r["job_id"]: r["passed"] == "True" for r in qa_pass_rows}

    by_unit: dict[str, list[bool]] = defaultdict(list)
    by_voice: dict[str, list[bool]] = defaultdict(list)
    for job_id, passed in pass_by_id.items():
        gen_row = gen_rows.get(job_id)
        if gen_row is None:
            continue
        by_unit[unit_name(gen_row["model"], prompt_source_of(gen_row["voice_id"]))].append(passed)
        by_voice[gen_row["voice_id"]].append(passed)

    lines = ["# QA pass-rate summary", "", "## By (model, prompt_source)", "", "| unit | n | pass rate |", "|---|---|---|"]
    for unit in sorted(by_unit):
        results = by_unit[unit]
        lines.append(f"| {unit} | {len(results)} | {sum(results) / len(results):.1%} |")

    lines += ["", "## By voice", "", "| voice_id | n | pass rate |", "|---|---|---|"]
    for voice_id in sorted(by_voice):
        results = by_voice[voice_id]
        lines.append(f"| {voice_id} | {len(results)} | {sum(results) / len(results):.1%} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_shim = sub.add_parser("build-shim")
    p_shim.add_argument("--gen-manifest", type=Path, required=True, help="T3's gen_manifest.csv")
    p_shim.add_argument("--shim-dir", type=Path, required=True)

    p_parse = sub.add_parser("parse")
    p_parse.add_argument("--gen-manifest", type=Path, required=True)
    p_parse.add_argument("--shim-dir", type=Path, required=True)
    p_parse.add_argument("--reports-dir", type=Path, required=True)
    p_parse.add_argument("--out", type=Path, required=True)

    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        if args.command == "build-shim":
            unit_dirs = build_shim(load_gen_manifest(args.gen_manifest), args.shim_dir)
            for unit, unit_dir in unit_dirs.items():
                n = len(list(unit_dir.glob("*.wav")))
                print(f"{unit}: {n} clips -> {unit_dir}")
        elif args.command == "parse":
            rows = parse_reports(args.gen_manifest, args.shim_dir, args.reports_dir)
            dest = write_qa_pass_csv(args.out, rows)
            summary_path = write_qa_summary(args.gen_manifest, rows, args.out.parent / "qa_summary.md")
            n_pass = sum(1 for r in rows if r["passed"] == "True")
            print(f"wrote {dest}: {n_pass}/{len(rows)} passed")
            print(f"wrote {summary_path}")
    except ManifestValidationError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
