"""Merge every subset this feature has generated so far -- `positives_real`,
`adversaries`, `adversaries_noisy`, `positives_converted`,
`positives_converted_noisy`, `silence_synthetic` -- into one flat manifest,
for convenience.

This is deliberately NOT ticket 04's job (`build_dataset.py`, not yet
written): no external corpora, no "computer"-token scrub, no class-ratio
balancing, no train/val/test split assignment, no physical audio copying.
Output is written under distinct names (`generated_manifest.csv` /
`generated_summary.md`) so it can never collide with ticket 04's eventual
`out/conversions/v2/wakeword/{manifest.csv,summary.md}`.

Each row's `path` is rewritten relative to `--wakeword-root` (prefixed with
its own subset dir name) rather than relative to that subset's own
directory, per docs/WAKEWORD-DATASET-CONTRACT.md section 6's note that a
merged manifest's path convention differs from a per-subset one.

**Comparability with the `simple-audio-transcriber` QA check** (see
`.scratch/wakeword-computer-dataset/HANDOFF.md` section 3): that tool's
reports key clips by bare `filename`, scoped to one source directory per
report -- i.e. one subset at a time, not this flattened view. Bare
`filename` is therefore NOT a reliable global join key across subsets on
its own (`positives_converted`'s own filenames repeat once per `group_id`,
by design); the authoritative unique key here is `path`, and this module
also reports (not fails on) how many bare-`filename` collisions exist, so
a future QA pass knows to join scoped by `subset` first.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from me2_voicegen.wakeword.fetch_positives import MANIFEST_FIELDS as BASE_MANIFEST_FIELDS, ManifestValidationError

DEFAULT_WAKEWORD_ROOT = Path("out/conversions/v2/wakeword")

# Fixed, known set -- not autodetected -- so an unrelated future directory
# under wakeword_root is never silently swept in.
KNOWN_SUBSETS = (
    "positives_real",
    "adversaries",
    "adversaries_noisy",
    "positives_converted",
    "positives_converted_noisy",
    "silence_synthetic",
)

# Extension columns any known subset may append after the mandatory 10,
# in this fixed order; "" where a subset/row doesn't have one.
EXTENSION_FIELDS = ["ref_voice", "noise_source_file", "snr_db"]
CONSOLIDATED_FIELDS = ["subset"] + BASE_MANIFEST_FIELDS + EXTENSION_FIELDS


def load_subset_rows(wakeword_root: Path, subset: str) -> list[dict]:
    manifest_path = wakeword_root / subset / "manifest.csv"
    if not manifest_path.is_file():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def consolidate(wakeword_root: Path, subsets: list[str]) -> tuple[list[dict], dict[str, int]]:
    """Returns (consolidated rows, per-subset row counts including 0 for
    any known subset that isn't present yet). Fails loudly on a duplicate
    consolidated `path` or a row whose file doesn't exist on disk -- both
    would mean a real bug, never an expected condition."""
    consolidated: list[dict] = []
    counts: dict[str, int] = {}
    seen_paths: dict[str, str] = {}

    for subset in subsets:
        rows = load_subset_rows(wakeword_root, subset)
        counts[subset] = len(rows)

        for row in rows:
            consolidated_path = f"{subset}/{row['path']}"
            if consolidated_path in seen_paths:
                raise ManifestValidationError(
                    f"duplicate consolidated path {consolidated_path!r} "
                    f"(subsets {seen_paths[consolidated_path]!r} and {subset!r})"
                )
            seen_paths[consolidated_path] = subset

            resolved = (wakeword_root / consolidated_path).resolve()
            if not resolved.is_file():
                raise ManifestValidationError(f"manifest row path does not exist on disk: {consolidated_path}")

            out_row = {field: "" for field in CONSOLIDATED_FIELDS}
            out_row.update({k: v for k, v in row.items() if k in CONSOLIDATED_FIELDS})
            out_row["subset"] = subset
            out_row["path"] = consolidated_path
            consolidated.append(out_row)

    return consolidated, counts


def write_manifest(out_path: Path, rows: list[dict]) -> Path:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONSOLIDATED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_summary(out_path: Path, *, subsets: list[str], counts: dict[str, int], rows: list[dict]) -> None:
    present = [s for s in subsets if counts.get(s, 0) > 0]
    absent = [s for s in subsets if counts.get(s, 0) == 0]

    label_counts: dict[str, int] = {}
    source_dataset_counts: dict[str, int] = {}
    filename_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
        source_dataset_counts[row["source_dataset"]] = source_dataset_counts.get(row["source_dataset"], 0) + 1
        filename_counts[row["filename"]] = filename_counts.get(row["filename"], 0) + 1
    filename_collisions = sum(1 for c in filename_counts.values() if c > 1)

    lines = [
        "# Wakeword generated-subsets consolidated manifest summary",
        "",
        "NOT the final ticket-04-assembled dataset -- no external corpora, no "
        '"computer"-token scrub, no class-ratio balancing, no train/val/test split. '
        "Just a flat merge of this feature's own locally-generated subsets, for "
        "convenience.",
        "",
        f"Total rows: **{len(rows)}**.",
        "",
        "## Per-subset row counts",
        "",
    ]
    for subset in subsets:
        lines.append(f"- `{subset}`: {counts.get(subset, 0)}" + (" (absent, skipped)" if subset in absent else ""))
    lines += [
        "",
        "## Per-label totals",
        "",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"- `{label}`: {count}")
    lines += [
        "",
        "## Per-source_dataset totals",
        "",
    ]
    for source_dataset, count in sorted(source_dataset_counts.items()):
        lines.append(f"- `{source_dataset}`: {count}")
    lines += [
        "",
        "## Join-key note",
        "",
        f"`path` is the unique key ({len(rows)} rows, {len(rows)} distinct paths by "
        "construction -- enforced, not assumed). Bare `filename` is NOT globally "
        f"unique: {filename_collisions} filenames repeat across rows (expected for "
        "`positives_converted`/`positives_converted_noisy`, whose filenames are only "
        "unique within one `group_id` subdirectory). A tool joining a per-subset QA "
        "report back to this manifest should filter by `subset` first, then match "
        "`filename` within that subset.",
        "",
        "## License",
        "",
    ]
    if any(s.endswith("_noisy") and s in present for s in subsets):
        lines += [
            "This manifest includes rows from a `*_noisy` subset, which mixes in "
            "`out/conversions/v2/background_noise` (ESC-50, CC-BY-NC-SA-4.0). Those "
            "rows -- and therefore this consolidated file as a whole -- are "
            "CC-BY-NC-SA-4.0-encumbered: non-commercial use only, share-alike on "
            "redistribution. See `docs/WAKEWORD-DATASET-CONTRACT.md` section 7.",
        ]
    else:
        lines += [
            "No `*_noisy` subset present in this consolidation -- no CC-BY-NC-SA-4.0 "
            "encumbrance from background_noise applies to the rows included here.",
        ]
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wakeword-root", type=Path, default=DEFAULT_WAKEWORD_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wakeword_root = args.wakeword_root.resolve()

    try:
        rows, counts = consolidate(wakeword_root, list(KNOWN_SUBSETS))
    except ManifestValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manifest_path = write_manifest(wakeword_root / "generated_manifest.csv", rows)
    write_summary(
        wakeword_root / "generated_summary.md",
        subsets=list(KNOWN_SUBSETS),
        counts=counts,
        rows=rows,
    )

    present = [s for s in KNOWN_SUBSETS if counts.get(s, 0) > 0]
    print(f"{len(rows)} rows from {len(present)} present subsets ({', '.join(present)}) -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
