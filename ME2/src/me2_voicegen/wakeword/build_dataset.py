"""Assemble the final wakeword dataset: merge this feature's subsets (each
already sized to the intended class composition by whichever step produced
it -- this module does no further sampling of its own) into
`out/conversions/v2/wakeword/{manifest.csv,summary.md}`, and assign a
GROUP-DISJOINT, per-label-stratified 70/20/10 train/val/test split.

Subset -> label mapping (fixed, not autodetected, so an unrelated future
directory is never silently swept in):

    _wakeword_: positives_real, positives_converted, positives_converted_noisy
    _unknown_:  adversaries, adversaries_noisy, common_voice_negative_sample
    _silence_:  silence_synthetic

Split assignment is GROUP-aware and greedy: within each label independently,
groups (by `group_id`) are sorted largest-first and each assigned to
whichever split is currently furthest below its 70/20/10 target share of
that label's rows (a standard greedy balanced-partition heuristic -- exact
70.0/20.0/10.0 is not achievable when a label's groups are lumpy, e.g.
`silence_synthetic`'s ~9-12 color/band groups, so this gets as close as
whole-group assignment allows and reports the realized split honestly
rather than clipping/rebalancing across a group boundary).

A `group_id` that (coincidentally) appears under more than one label reuses
whichever split it was already assigned under the first label processed,
rather than re-deciding independently -- this guarantees the "never split
a group across train/val/test" invariant by construction rather than by
after-the-fact luck, and is then verified with an explicit post-assignment
assertion per docs/WAKEWORD-DATASET-CONTRACT.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from me2_voicegen.wakeword.consolidate_manifest import load_subset_rows
from me2_voicegen.wakeword.fetch_positives import MANIFEST_FIELDS as BASE_MANIFEST_FIELDS, ManifestValidationError

DEFAULT_WAKEWORD_ROOT = Path("out/conversions/v2/wakeword")

SUBSETS_BY_LABEL = {
    "_wakeword_": ("positives_real", "positives_converted", "positives_converted_noisy"),
    "_unknown_": ("adversaries", "adversaries_noisy", "common_voice_negative_sample"),
    "_silence_": ("silence_synthetic",),
}
# Fixed processing order -- also the tie-break order for the rare
# cross-label group_id collision case described in the module docstring.
LABELS = ("_wakeword_", "_unknown_", "_silence_")

# speech_start_s/speech_end_s: feature wakeword-dscnn's derive_speech_spans.py
# extension columns (WAKEWORD-DATASET-CONTRACT.md section 8 addendum) --
# must stay whitelisted here or they're silently dropped by the `k in
# MANIFEST_FIELDS` filter below when this final manifest is assembled.
EXTENSION_FIELDS = ["ref_voice", "noise_source_file", "snr_db", "speech_start_s", "speech_end_s"]
MANIFEST_FIELDS = BASE_MANIFEST_FIELDS + EXTENSION_FIELDS

SPLIT_TARGETS = {"train": 0.70, "val": 0.20, "test": 0.10}


def load_label_rows(wakeword_root: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for subset in SUBSETS_BY_LABEL[label]:
        for row in load_subset_rows(wakeword_root, subset):
            out_row = {field: "" for field in MANIFEST_FIELDS}
            out_row.update({k: v for k, v in row.items() if k in MANIFEST_FIELDS})
            out_row["path"] = f"{subset}/{row['path']}"
            if out_row["label"] != label:
                raise ManifestValidationError(
                    f"{subset}/{row['path']}: manifest label {out_row['label']!r} != "
                    f"expected {label!r} for this subset mapping"
                )
            out_row["_source_subset"] = subset  # internal only, stripped before writing
            rows.append(out_row)
    return rows


def assign_group_disjoint_splits(rows_by_label: dict[str, list[dict]], seed: int) -> dict[str, dict]:
    """Mutates every row's `split` in place. Returns per-label realized
    counts per split, for reporting."""
    assigned: dict[str, str] = {}
    stats: dict[str, dict] = {}

    for label in LABELS:
        rows = rows_by_label[label]
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)

        # Pre-count rows already pinned by an earlier label's assignment
        # (the rare cross-label group_id collision case) so the greedy
        # balancer below compensates for them correctly.
        running = {split: 0 for split in SPLIT_TARGETS}
        undecided: list[tuple[str, list[dict]]] = []
        for group_id, group_rows in groups.items():
            if group_id in assigned:
                running[assigned[group_id]] += len(group_rows)
            else:
                undecided.append((group_id, group_rows))

        total = len(rows)
        undecided.sort(key=lambda item: (-len(item[1]), item[0]))  # largest group first, deterministic tiebreak
        for group_id, group_rows in undecided:
            def deficit(split: str) -> float:
                target = SPLIT_TARGETS[split] * total
                return target - running[split]

            chosen = max(SPLIT_TARGETS, key=deficit)
            assigned[group_id] = chosen
            running[chosen] += len(group_rows)

        for row in rows:
            row["split"] = assigned[row["group_id"]]

        stats[label] = {
            "total": total,
            "n_groups": len(groups),
            **{split: running[split] for split in SPLIT_TARGETS},
        }

    return stats


def assert_group_disjointness(rows: list[dict]) -> None:
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])
    violations = {gid: splits for gid, splits in splits_by_group.items() if len(splits) > 1}
    if violations:
        sample = dict(list(violations.items())[:5])
        raise ManifestValidationError(f"{len(violations)} group_id(s) span more than one split: {sample}")


def group_concentration_notes(rows_by_label: dict[str, list[dict]]) -> list[str]:
    notes: list[str] = []
    for label, rows in rows_by_label.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["group_id"]] = counts.get(row["group_id"], 0) + 1
        if not counts:
            continue
        top_group, top_count = max(counts.items(), key=lambda kv: kv[1])
        share = top_count / len(rows)
        if share >= 0.05:
            notes.append(
                f"- `{label}`: largest single `group_id` (`{top_group}`) is {top_count}/{len(rows)} "
                f"rows ({share:.1%}) of this class."
            )
    return notes or ["- No single `group_id` reaches 5% of any class."]


def verify_paths_exist(wakeword_root: Path, rows: list[dict]) -> None:
    for row in rows:
        resolved = (wakeword_root / row["path"]).resolve()
        if not resolved.is_file():
            raise ManifestValidationError(f"manifest row path does not exist on disk: {row['path']}")


def write_manifest(out_path: Path, rows: list[dict]) -> Path:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    return out_path


def write_summary(out_path: Path, *, seed: int, stats: dict[str, dict], rows_by_label: dict[str, list[dict]]) -> None:
    total = sum(s["total"] for s in stats.values())
    lines = [
        "# Wakeword final dataset assembly summary",
        "",
        f"seed={seed}. Total rows: **{total}**.",
        "",
        "## Class composition",
        "",
    ]
    for label in LABELS:
        s = stats[label]
        pct = s["total"] / total * 100 if total else 0.0
        lines.append(f"- `{label}`: {s['total']} rows ({pct:.1f}% of total), {s['n_groups']} groups.")
    lines += [
        "",
        "## Realized train/val/test split (group-disjoint, per-label greedy balance vs. 70/20/10 target)",
        "",
        "| label | train | val | test | train% | val% | test% |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        s = stats[label]
        t = s["total"] or 1
        lines.append(
            f"| `{label}` | {s['train']} | {s['val']} | {s['test']} | "
            f"{s['train'] / t:.1%} | {s['val'] / t:.1%} | {s['test'] / t:.1%} |"
        )
    lines += [
        "",
        "Exact 70.0/20.0/10.0 is not always achievable -- assignment is by whole "
        "`group_id`, never split, so a label with few/lumpy groups (e.g. "
        "`_silence_`'s small number of color/band groups) lands close to, not "
        "exactly on, the target.",
        "",
        "## Group-concentration notes",
        "",
        *group_concentration_notes(rows_by_label),
        "",
        "## Confounds (real, not papered over)",
        "",
        "- Picovoice real positives have no speaker labels (per "
        "`docs/WAKEWORD-DATASET-CONTRACT.md` section 5) -- splits are "
        "source-clip-disjoint, NOT speaker-disjoint. Do not claim speaker-disjointness.",
        "- `_wakeword_` mixes real, voice-converted, and noise-augmented-converted "
        "clips; `_unknown_` mixes narrow phonetic-adversary phrases (7 groups) with "
        "broad general speech (`common_voice_negative_sample`, near-singleton groups) "
        "-- very different diversity profiles within one class.",
        "- Duration profiles differ by source (real/converted positives ~1-3s, "
        "`common_voice_negative_sample` chunks ~0.3-2.3s, synthetic silence per "
        "`generate_silence.py`'s own bands).",
        "",
        "## License",
        "",
        "This dataset includes rows from `adversaries_noisy`/`positives_converted_noisy` "
        "(background_noise/ESC-50, CC-BY-NC-SA-4.0) -- see "
        "`docs/WAKEWORD-DATASET-CONTRACT.md` section 7. As a whole, this dataset is "
        "CC-BY-NC-SA-4.0-encumbered: non-commercial use only, share-alike on "
        "redistribution of the dataset or of anything trained on it. "
        "`common_voice_negative_sample` itself is separately CC0-1.0 (no encumbrance "
        "of its own), and the remaining sources are Apache-2.0/public-domain/"
        "no-external-restriction -- it is the `*_noisy` rows specifically that govern "
        "the whole.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wakeword-root", type=Path, default=DEFAULT_WAKEWORD_ROOT)
    parser.add_argument("--seed", type=int, required=True, help="seed for deterministic split-assignment tiebreaks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wakeword_root = args.wakeword_root.resolve()

    try:
        rows_by_label = {label: load_label_rows(wakeword_root, label) for label in LABELS}
        for label, rows in rows_by_label.items():
            if not rows:
                raise ManifestValidationError(
                    f"no rows found for label {label!r} across subsets {SUBSETS_BY_LABEL[label]} "
                    f"under {wakeword_root} -- build them first"
                )

        stats = assign_group_disjoint_splits(rows_by_label, args.seed)

        all_rows = [row for rows in rows_by_label.values() for row in rows]
        for row in all_rows:
            row.pop("_source_subset", None)

        assert_group_disjointness(all_rows)
        verify_paths_exist(wakeword_root, all_rows)

        seen_paths: dict[str, bool] = {}
        for row in all_rows:
            if row["path"] in seen_paths:
                raise ManifestValidationError(f"duplicate path across subsets: {row['path']}")
            seen_paths[row["path"]] = True
    except ManifestValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_manifest(wakeword_root / "manifest.csv", all_rows)
    write_summary(wakeword_root / "summary.md", seed=args.seed, stats=stats, rows_by_label=rows_by_label)

    print(f"{len(all_rows)} rows -> {wakeword_root / 'manifest.csv'}")
    for label in LABELS:
        s = stats[label]
        print(f"  {label}: {s['total']} (train={s['train']} val={s['val']} test={s['test']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
