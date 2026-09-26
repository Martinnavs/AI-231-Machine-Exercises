"""T7 (.scratch/accent-balance-fil50/tickets/00-RECAP.md): summarize a
pilot run (stages 0-3 only) so the user can review it before authorizing
the full ~15-20 GPU-hour run.

Reads T3's `gen_manifest.csv` (throughput) and T4's `qa_summary.md`/
`qa_pass.csv` (pass rates), recomputes the REAL (non-pilot) per-split
deficit against the actual base manifests to project full-run cost, and
copies a handful of passing/flagged clips per prompt_source into a
`listen/` directory for the user to play.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from random import Random

from me2_voicegen.accent_balance.plan_jobs import count_vcm_deficit, count_ww_deficit
from me2_voicegen.accent_balance.qa import prompt_source_of

logger = logging.getLogger(__name__)


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def throughput_by_model(gen_rows: list[dict]) -> dict[str, float]:
    """model -> mean seconds/clip over status=ok rows."""
    by_model: dict[str, list[float]] = {}
    for row in gen_rows:
        if row["status"] != "ok" or not row.get("seconds_elapsed"):
            continue
        by_model.setdefault(row["model"], []).append(float(row["seconds_elapsed"]))
    return {model: sum(v) / len(v) for model, v in by_model.items() if v}


def project_full_run(
    *,
    vcm_manifest: Path,
    wakeword_manifest: Path,
    overgen: float,
    throughput: dict[str, float],
    n_gpus: int,
) -> dict:
    vcm_counts = count_vcm_deficit(vcm_manifest)
    ww_counts = count_ww_deficit(wakeword_manifest)
    vcm_jobs = sum(int(c["deficit"] * overgen + 0.999) for c in vcm_counts.values())
    ww_jobs = sum(int(c["deficit"] * overgen + 0.999) for c in ww_counts.values())

    vcm_seconds = vcm_jobs * throughput.get("vcm", 0.0)
    ww_seconds = ww_jobs * throughput.get("wakeword", 0.0)
    total_gpu_hours = (vcm_seconds + ww_seconds) / 3600.0
    wall_clock_hours = total_gpu_hours / max(n_gpus, 1)

    return {
        "vcm_jobs": vcm_jobs, "wakeword_jobs": ww_jobs,
        "vcm_gpu_hours": vcm_seconds / 3600.0, "wakeword_gpu_hours": ww_seconds / 3600.0,
        "total_gpu_hours": total_gpu_hours, "n_gpus": n_gpus, "wall_clock_hours": wall_clock_hours,
    }


def populate_listen_dir(gen_rows: list[dict], qa_rows: list[dict], listen_dir: Path, n_per_bucket: int, seed: int) -> None:
    passed_by_id = {r["job_id"]: r["passed"] == "True" for r in qa_rows}
    gen_by_id = {r["job_id"]: r for r in gen_rows}

    by_source_status: dict[tuple[str, str], list[dict]] = {}
    for job_id, passed in passed_by_id.items():
        row = gen_by_id.get(job_id)
        if row is None or row["status"] != "ok" or not row.get("path"):
            continue
        source = prompt_source_of(row["voice_id"])
        by_source_status.setdefault((source, "pass" if passed else "flag"), []).append(row)

    rng = Random(seed)
    for (source, status), rows in sorted(by_source_status.items()):
        rng.shuffle(rows)
        dest_dir = listen_dir / source / status
        dest_dir.mkdir(parents=True, exist_ok=True)
        for row in rows[:n_per_bucket]:
            src = Path(row["path"])
            if src.is_file():
                shutil.copy2(src, dest_dir / f"{row['job_id']}_{src.name}")


def write_report(
    *,
    out_path: Path,
    throughput: dict[str, float],
    projection: dict,
    qa_summary_path: Path,
    listen_dir: Path,
) -> Path:
    lines = ["# accent-balance-fil50 pilot report", ""]

    lines += ["## Throughput (measured)", "", "| model | mean s/clip |", "|---|---|"]
    for model in sorted(throughput):
        lines.append(f"| {model} | {throughput[model]:.2f} |")

    lines += [
        "",
        "## Full-run projection",
        "",
        f"- VCM: {projection['vcm_jobs']} jobs, {projection['vcm_gpu_hours']:.1f} GPU-hours",
        f"- Wakeword: {projection['wakeword_jobs']} jobs, {projection['wakeword_gpu_hours']:.1f} GPU-hours",
        f"- **Total: {projection['total_gpu_hours']:.1f} GPU-hours**",
        f"- At {projection['n_gpus']} GPU(s): ~{projection['wall_clock_hours']:.1f} wall-clock hours",
        "",
        "## QA pass rates",
        "",
    ]
    if qa_summary_path.is_file():
        lines.append(qa_summary_path.read_text(encoding="utf-8"))
    else:
        lines.append("(qa_summary.md not found -- did stage 3 run?)")

    lines += [
        "",
        f"## Sample clips ({listen_dir})",
        "",
        "Up to 10 passing and 5 flagged clips per prompt_source, copied for review.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="the pilot's $ROOT dir")
    parser.add_argument("--vcm-manifest", type=Path, required=True)
    parser.add_argument("--wakeword-manifest", type=Path, required=True)
    parser.add_argument("--overgen", type=float, default=1.15)
    parser.add_argument("--gpus", type=int, default=1, help="GPU count to project full-run wall-clock at")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-bucket", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    gen_rows = load_csv(args.root / "gen_manifest.csv")
    qa_rows = load_csv(args.root / "qa" / "qa_pass.csv")
    throughput = throughput_by_model(gen_rows)
    projection = project_full_run(
        vcm_manifest=args.vcm_manifest, wakeword_manifest=args.wakeword_manifest,
        overgen=args.overgen, throughput=throughput, n_gpus=args.gpus,
    )
    listen_dir = args.root / "listen"
    populate_listen_dir(gen_rows, qa_rows, listen_dir, n_per_bucket=args.n_per_bucket, seed=args.seed)
    dest = write_report(
        out_path=args.root / "pilot_report.md", throughput=throughput, projection=projection,
        qa_summary_path=args.root / "qa" / "qa_summary.md", listen_dir=listen_dir,
    )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
