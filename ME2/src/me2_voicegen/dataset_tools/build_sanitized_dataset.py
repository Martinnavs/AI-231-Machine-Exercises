"""Split `make qa-batch` output into clean/ and flagged/ using the QA reports.

`generate_conversions.py` output gets QA'd by simple-audio-transcriber's
`audio_transcript_parser.qa` (see out/conversions/v2/reports/*.md): each clip is
transcribed and scored against phrase.txt's expected phrase, and anything under
threshold lands in that report's "## Flagged" table. This module doesn't re-run
QA - it trusts those reports and copies each clip into clean/<INTENT>/ or
flagged/<INTENT>/ accordingly.

Stdlib-only: this repo's real dependencies (torch, CosyVoice, ...) are for
synthesis, not for copying files that already exist on disk.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^# QA Report: (?P<name>.+)$")
SOURCE_RE = re.compile(r"^- \*\*Source\*\*: (?P<path>.+)$")
FLAGGED_FILE_RE = re.compile(r"^\|\s*([^|]+?)\s*\|")

DEFAULT_QA_ROOT = Path.home() / "simple-audio-transcriber" / ".tmp" / "qa"


def parse_report(path: Path) -> tuple[str, Path, set[str]]:
    """Returns (intent, source_dir, flagged_filenames)."""
    intent: str | None = None
    source: Path | None = None
    flagged: set[str] = set()
    in_flagged = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if heading := HEADING_RE.match(line):
            intent = heading.group("name").strip()
        elif src := SOURCE_RE.match(line):
            source = Path(src.group("path").strip())
        elif line.strip() == "## Flagged":
            in_flagged = True
        elif line.startswith("## "):
            in_flagged = False
        elif in_flagged and (row := FLAGGED_FILE_RE.match(line)):
            name = row.group(1).replace("\\|", "|")
            if name not in ("file", "---"):
                flagged.add(name)

    if intent is None or source is None:
        raise ValueError(f"{path}: missing '# QA Report: <name>' heading or '- **Source**' line")

    return intent, source, flagged


def resolve_source(source: Path, qa_root: Path) -> Path:
    """source is `.tmp/qa/<unit>/<name>` relative to the transcriber repo it was
    produced in; qa_root is that repo's `.tmp/qa` on this machine."""
    parts = source.parts
    try:
        tail = parts[parts.index("qa") + 1 :]
    except ValueError:
        tail = parts
    return qa_root.joinpath(*tail)


def split_reports(reports_dir: Path, qa_root: Path, out_dir: Path) -> tuple[int, int]:
    kept_total = 0
    flagged_total = 0

    for report_path in sorted(reports_dir.glob("*.md")):
        intent, source, flagged_names = parse_report(report_path)
        src_dir = resolve_source(source, qa_root)
        if not src_dir.is_dir():
            raise FileNotFoundError(
                f"{report_path}: resolved source {src_dir} does not exist "
                f"(pass --qa-root if simple-audio-transcriber's .tmp/qa lives elsewhere)"
            )

        clean_dir = out_dir / "clean" / intent
        flagged_dir = out_dir / "flagged" / intent
        clean_dir.mkdir(parents=True, exist_ok=True)
        flagged_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in sorted(src_dir.glob("*.wav")):
            if wav_path.name in flagged_names:
                shutil.copy2(wav_path, flagged_dir / wav_path.name)
                flagged_total += 1
            else:
                shutil.copy2(wav_path, clean_dir / wav_path.name)
                kept_total += 1

    return kept_total, flagged_total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("out/conversions/v2/reports"),
        help="directory of QA report .md files (default: out/conversions/v2/reports)",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=DEFAULT_QA_ROOT,
        help=f"simple-audio-transcriber's .tmp/qa on this machine (default: {DEFAULT_QA_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/conversions/v2/sanitized"),
        help="output directory: <out-dir>/clean/<INTENT>/*.wav + <out-dir>/flagged/<INTENT>/*.wav",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.reports_dir.is_dir():
        print(f"error: reports dir not found: {args.reports_dir}", file=sys.stderr)
        return 1

    kept, flagged = split_reports(args.reports_dir, args.qa_root, args.out_dir)
    print(f"{kept} clean, {flagged} flagged -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
