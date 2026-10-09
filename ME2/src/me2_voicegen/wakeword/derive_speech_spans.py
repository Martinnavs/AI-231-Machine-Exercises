"""One-time offline pass that derives `speech_start_s`/`speech_end_s`
extension columns for a subset of `out/conversions/v2/wakeword/`'s
manifests (feature `wakeword-dscnn`, feature-engineering/wakeword-dscnn/
SPEC.md -- see that doc's RECAP for the full rationale and the measured
numbers backing every design choice here; only a summary is repeated
below).

VAD (`torchaudio.functional.vad`, forward + reversed-waveform pass) is run
**once per clip's cleanest ancestor**, never on noisy or voice-converted
audio directly, then propagated onto derived/noisy siblings via the
dataset's own provenance columns (`group_id`, `ref_voice`):

    positives_real  --(group_id)-->            positives_converted
    positives_converted --(group_id,ref_voice)--> positives_converted_noisy
    common_voice_negative_sample (no derived sibling -- cached directly)

`adversaries`/`adversaries_noisy` are deliberately **not** touched by this
module: measured VAD-empty rates on the real dataset are ~54%/~45%
respectively (SPEC.md round 3) -- noise is not the dominant failure mode
there, precomputing would not have bought materially cleaner anchoring,
and a misfire on `_unknown_` content is low-risk (it still isn't the
wakeword) unlike on `_wakeword_` content. Those two subsets are left on
`WakewordDataset`'s live-VAD-with-fallback path.

`silence_synthetic` is never VAD'd (synthetic silence by construction).

A row whose join fails, or whose mapped span's end exceeds its own
measured `duration` by more than `SPAN_MARGIN_SECONDS` (voice conversion's
measured duration drift is a few tens of ms at most, so this is a real
sanity gate, not a formality), is left with empty span columns rather than
a wrong value -- `WakewordDataset` falls back to live per-sample VAD for
exactly those rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

SPAN_MARGIN_SECONDS: float = 0.15
SPAN_COLUMNS: tuple[str, str] = ("speech_start_s", "speech_end_s")

# Subsets that are their own ancestor: VAD runs on them directly, once,
# cached into their own manifest (no join needed).
ANCESTOR_SUBSETS: tuple[str, ...] = ("positives_real", "common_voice_negative_sample")

# (child_subset, ancestor_subset, join_keys) -- processed in this order, so
# an ancestor earlier in this list (positives_converted) is already
# populated by the time a later entry (positives_converted_noisy) joins
# against it.
DERIVED_SUBSETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("positives_converted", "positives_real", ("group_id",)),
    ("positives_converted_noisy", "positives_converted", ("group_id", "ref_voice")),
)


def detect_speech_span(waveform: torch.Tensor, sample_rate: int) -> tuple[float, float] | None:
    """`(start_s, end_s)` of detected speech via a forward pass (trims
    leading silence) then a reversed-waveform pass on the survivor (trims
    trailing silence). `None` if either pass collapses to an empty
    waveform -- callers must treat that as "no span available", not as
    "span covers the whole clip"."""
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    forward_trimmed = AF.vad(waveform.unsqueeze(0), sample_rate).squeeze(0)
    if forward_trimmed.numel() == 0:
        return None

    start_sample = waveform.numel() - forward_trimmed.numel()
    reversed_forward = torch.flip(forward_trimmed, dims=[0])
    reversed_trimmed = AF.vad(reversed_forward.unsqueeze(0), sample_rate).squeeze(0)
    if reversed_trimmed.numel() == 0:
        return None

    end_sample = start_sample + reversed_trimmed.numel()
    return start_sample / sample_rate, end_sample / sample_rate


def _read_manifest(manifest_path: Path) -> tuple[list[dict], list[str]]:
    import csv

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_manifest(manifest_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    import csv

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _with_span_columns(fieldnames: list[str]) -> list[str]:
    out = list(fieldnames)
    for col in SPAN_COLUMNS:
        if col not in out:
            out.append(col)
    return out


def derive_for_ancestor(wakeword_root: Path, subset_name: str) -> dict:
    """Run `detect_speech_span` on every row of `subset_name`'s own
    manifest and write `speech_start_s`/`speech_end_s` into that manifest
    in place. Returns a small stats dict (also this function's proof of
    work for `main`'s printed summary)."""
    manifest_path = wakeword_root / subset_name / "manifest.csv"
    audio_root = manifest_path.parent
    rows, fieldnames = _read_manifest(manifest_path)
    fieldnames = _with_span_columns(fieldnames)

    n_empty = 0
    for row in rows:
        wav_path = audio_root / row["path"]
        waveform, sample_rate = torchaudio.load(str(wav_path))
        span = detect_speech_span(waveform, sample_rate)
        if span is None:
            n_empty += 1
            row[SPAN_COLUMNS[0]] = ""
            row[SPAN_COLUMNS[1]] = ""
        else:
            row[SPAN_COLUMNS[0]] = f"{span[0]:.6f}"
            row[SPAN_COLUMNS[1]] = f"{span[1]:.6f}"

    _write_manifest(manifest_path, rows, fieldnames)
    return {"subset": subset_name, "rows": len(rows), "empty": n_empty}


def derive_for_join(
    wakeword_root: Path,
    child_subset: str,
    ancestor_subset: str,
    join_keys: tuple[str, ...],
) -> dict:
    """Propagate `speech_start_s`/`speech_end_s` from `ancestor_subset`'s
    manifest (already carrying those columns) onto `child_subset`'s
    manifest via `join_keys`, after a sanity check (mapped end <= the
    child row's own measured `duration` + `SPAN_MARGIN_SECONDS`). Rows
    that fail the join or the sanity check are left with empty span
    columns -- `WakewordDataset`'s live-VAD fallback territory, not a
    fatal error here."""
    ancestor_path = wakeword_root / ancestor_subset / "manifest.csv"
    ancestor_rows, _ = _read_manifest(ancestor_path)

    ancestor_index: dict[tuple, dict] = {}
    n_duplicate_keys = 0
    for row in ancestor_rows:
        key = tuple(row.get(k, "") for k in join_keys)
        if key in ancestor_index:
            n_duplicate_keys += 1
            continue
        ancestor_index[key] = row
    if n_duplicate_keys:
        print(
            f"WARNING: {ancestor_subset} has {n_duplicate_keys} duplicate "
            f"{join_keys} key(s) -- first match wins per key, per-row "
            "provenance may be ambiguous for those rows"
        )

    child_path = wakeword_root / child_subset / "manifest.csv"
    child_rows, fieldnames = _read_manifest(child_path)
    fieldnames = _with_span_columns(fieldnames)

    n_mapped = 0
    n_missing_join = 0
    n_failed_sanity = 0
    for row in child_rows:
        key = tuple(row.get(k, "") for k in join_keys)
        ancestor_row = ancestor_index.get(key)
        if (
            ancestor_row is None
            or not ancestor_row.get(SPAN_COLUMNS[0])
            or not ancestor_row.get(SPAN_COLUMNS[1])
        ):
            n_missing_join += 1
            row[SPAN_COLUMNS[0]] = ""
            row[SPAN_COLUMNS[1]] = ""
            continue

        start_s = float(ancestor_row[SPAN_COLUMNS[0]])
        end_s = float(ancestor_row[SPAN_COLUMNS[1]])
        child_duration = float(row["duration"])
        if end_s > child_duration + SPAN_MARGIN_SECONDS:
            n_failed_sanity += 1
            row[SPAN_COLUMNS[0]] = ""
            row[SPAN_COLUMNS[1]] = ""
            continue

        row[SPAN_COLUMNS[0]] = f"{start_s:.6f}"
        row[SPAN_COLUMNS[1]] = f"{end_s:.6f}"
        n_mapped += 1

    _write_manifest(child_path, child_rows, fieldnames)
    return {
        "subset": child_subset,
        "rows": len(child_rows),
        "mapped": n_mapped,
        "missing_join": n_missing_join,
        "failed_sanity": n_failed_sanity,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wakeword-root",
        type=Path,
        default=Path("out/conversions/v2/wakeword"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root: Path = args.wakeword_root

    for subset in ANCESTOR_SUBSETS:
        stats = derive_for_ancestor(root, subset)
        empty_pct = stats["empty"] / max(stats["rows"], 1) * 100
        print(f"{stats['subset']}: {stats['rows']} rows, {stats['empty']} empty-span ({empty_pct:.1f}%)")

    for child, ancestor, keys in DERIVED_SUBSETS:
        stats = derive_for_join(root, child, ancestor, keys)
        print(
            f"{stats['subset']}: {stats['rows']} rows, {stats['mapped']} mapped, "
            f"{stats['missing_join']} missing-join, {stats['failed_sanity']} failed-sanity"
        )


if __name__ == "__main__":
    main()
