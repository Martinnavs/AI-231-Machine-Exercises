#!/usr/bin/env python3
"""Pilot report + `listen/` dir for the ambient-noise-overlay feature
(feature-engineering/ambient-noise-overlay/SPEC.md, Proof #5).

Reads the pilot's two derived manifests, copies up to --per-band stored
ambient mixes per model per SNR band (5 dB bins, the summary's convention;
deterministic: first rows in manifest order) into
`<out-root>/listen/<model>/<band>/<filename>`, and writes
`<out-root>/pilot_report.md` (gate results + a listen checklist) so a
human can review quality before the full run.

Standalone on purpose (csv/pathlib/shutil only -- no torch, no
me2_voicegen import): run with the ME2 venv,
`uv run python scripts/ambient_pilot_report.py ...`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Keep in sync with common.ambient_mix (inlined so this script stays import-light).
SNR_MIN_DB = -5.0
SNR_MAX_DB = 30.0


def _band_of(snr_db: float) -> str:
    lo = int(max(SNR_MIN_DB, snr_db // 5 * 5))
    hi = min(lo + 5, SNR_MAX_DB)
    return f"{lo:+.0f}to{hi:+.0f}dB"


def _collect(model: str, manifest_path: Path, out_root: Path, per_band: int) -> tuple[list[str], int]:
    """Copy per-band samples; return (pilot-report lines, n stored)."""
    audio_root = manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ambient = [r for r in rows if r.get("source_dataset", "").endswith("_ambient")]

    lines = [f"## {model}", ""]
    if not ambient:
        lines.append("- **no ambient rows stored** -- the gate dropped every mix (check the summary's pass rates)")
        return lines, 0

    by_band: dict[str, list[dict]] = {}
    for r in ambient:
        by_band.setdefault(_band_of(float(r["snr_db"])), []).append(r)

    lines += ["| SNR band | split | snr_db | source row | sample |", "|---|---|---|---|---|"]
    for band in sorted(by_band, key=lambda b: float(b.split("to")[0])):
        for r in by_band[band][:per_band]:
            src = audio_root / r["path"]
            if not src.is_file():
                lines.append(f"| {band} | {r['split']} | {r['snr_db']} | {r['filename']} | **missing on disk: {src}** |")
                continue
            dest_dir = out_root / "listen" / model / band
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / r["filename"]
            shutil.copyfile(src, dest)
            lines.append(
                f"| {band} | {r['split']} | {r['snr_db']} | {r['filename']} | `{dest.relative_to(out_root)}` |"
            )
    lines += [
        "",
        f"stored ambient rows: {len(ambient)} (table shows up to {per_band} per band; full list: {manifest_path})",
        "",
    ]
    return lines, len(ambient)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vcm-manifest", type=Path, required=True)
    parser.add_argument("--vcm-summary", type=Path, required=True)
    parser.add_argument("--ww-manifest", type=Path, required=True)
    parser.add_argument("--ww-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True, help="pilot root ($ROOT); listen/ + pilot_report.md land here")
    parser.add_argument("--per-band", type=int, default=3)
    args = parser.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Ambient-noise overlay -- pilot report",
        "",
        f"Generated {stamp}. Pilot run: first N eligible rows per model (see the mix logs in `{args.out_root}/logs/`).",
        "",
        "## Listen checklist (per sample, in `listen/`)",
        "",
        "1. babble (people talking) is present and natural-sounding -- not a tone, click, or ESC-50 texture",
        "2. at low SNR ([-5, +5] dB) the command is still recoverable by ear -- this is the target robustness condition",
        "3. at high SNR ([+25, +30] dB) the babble is faint but audible -- the anchor condition",
        "4. no clipping distortion (harsh crackle) beyond what the summary's clipping rate says is expected",
        "5. no double-noise artifact (the base row was clean/ESC-50-anchored only; a sibling must sound like ONE noise mix)",
        "",
    ]

    for model, manifest, summary in (
        ("vcm", args.vcm_manifest, args.vcm_summary),
        ("wakeword_sesame", args.ww_manifest, args.ww_summary),
    ):
        section, n = _collect(model, manifest, args.out_root, args.per_band)
        lines += section
        if summary.is_file():
            lines += [f"### {model} -- full gate summary", "", "```markdown", summary.read_text(encoding="utf-8").rstrip(), "```", ""]
        else:
            lines += [f"### {model} -- full gate summary", "", f"(missing: {summary})", ""]
        lines.append(f"stored ambient rows ({model}): {n}")
        lines.append("")

    report = args.out_root / "pilot_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report} (listen dir: {args.out_root / 'listen'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
