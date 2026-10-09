"""Merge Option B (M0, `out/conversions/v2/optionb-v3/manifest.csv`) with
VCM Dataset B (balanced, `raw_datasets/VCM_BALANCED*`) and an ESC-50
silence pool rebuilt by fold into one speaker-disjoint train/val/test
manifest pool. See `.scratch/optionb-v3-vcmx/tickets/00-RECAP.md` (Phase 2)
for the full design rationale -- summarized here only where it isn't
obvious from the code.

Lives at the `vcm` package level (not under `vcm.optionb`), so it may import
`vcm.text` and `vcm.optionb`. The one-directional rule in
`docs/VCM-CONTRACT.md` section 4 forbids only `vcm.optiona`/`vcm.optionb`
importing `vcm`'s own generic machinery -- this module is exactly the
reverse direction (a `vcm`-level consumer of `vcm.optionb`'s grammar), so it
is not affected by that rule.

Zero-copy by design: no audio bytes are ever copied or resampled here.
Every row's `path` points at an existing file under `raw_datasets/`,
`out/conversions/v2/optionb-v3/`, or `out/conversions/v2/background_noise/`,
rewritten relative to this module's own output directory -- the same
"embed by reference" precedent `optionb-v3/manifest.csv` itself uses for its
`test_set` babble/silence probes.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.text import normalize_text, resolve_transcript

# ---------------------------------------------------------------------------
# Schema / constants.
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "filename",
    "path",
    "bucket",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
    "transcript",
    "original_dataset",
]

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2

# B `class` -> OPTIONB_GRAMMAR intent, for classes whose name differs from
# the grammar's own intent name. Classes absent here keep their name
# unchanged (e.g. PLAY_MUSIC, WEATHER, TIME already match).
LABEL_MAP: dict[str, str] = {
    "MEDIA_NEXT": "NEXT",
    "MEDIA_PAUSE": "PAUSE",
    "MEDIA_STOP": "STOP",
    "SET_TIMER": "TIMER",
    "SET_ALARM": "ALARM",
    "SET_TEMPERATURE": "TEMPERATURE",
    "LIGHT_DIM": "BRIGHTNESS",
}

_CURLY_APOSTROPHES = "‘’ʼ"
_MAX_UNKNOWN_DIGIT_VALUE = 100  # spell_integer's own ceiling (numbers.py)

_PCT_RE = re.compile(r"(\d+)\s*%")
_HHMM00_RE = re.compile(r"(\d+):00")
_DIGIT_AMPM_RE = re.compile(r"(\d+)\s*(am|pm)")
_NON_ALIAS_RE = re.compile(r"[^a-z0-9' ]")
_DIGIT_RUN_RE = re.compile(r"\d+")

FOLD_TO_SPLIT: dict[str, str] = {"1": "train", "2": "train", "3": "train", "4": "val", "5": "test"}


class VCMXMergeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Text normalization (D-alias, ticket Phase 2 Step 1).
# ---------------------------------------------------------------------------


def _base_alias_normalize(text: str) -> str:
    """Lowercase + numeric/apostrophe aliasing, *before* the final
    non-`[a-z0-9' ]` strip -- split out so `normalize_unknown_text` can
    insert its own `@`/`&` substitutions between this step and the strip
    (those characters would otherwise be silently dropped by the strip
    before ever becoming " at "/" and ")."""
    text = text.lower()
    for ch in _CURLY_APOSTROPHES:
        text = text.replace(ch, "'")
    text = _PCT_RE.sub(r"\1 percent", text)
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = _HHMM00_RE.sub(r"\1", text)
    text = _DIGIT_AMPM_RE.sub(r"\1 \2", text)
    return text


def _strip_and_collapse(text: str) -> str:
    text = _NON_ALIAS_RE.sub(" ", text)
    return " ".join(text.split())


def alias_normalize(text: str) -> str:
    """Full Step-1 alias normalization for B command-row transcripts
    (joined from A). Digits are kept (unlike `vcm.text.normalize_text`) so
    `OPTIONB_GRAMMAR.accepts` can match digit-form slot phrases directly."""
    return _strip_and_collapse(_base_alias_normalize(text))


def normalize_unknown_text(text: str) -> str | None:
    """B UNKNOWN-row normalization: `@` -> " at ", `&` -> " and "; returns
    `None` (caller drops the row) if the text contains `#` or a digit run
    whose integer value exceeds 100 (`spell_integer`'s own ceiling -- a
    value it would raise on rather than silently mis-spell)."""
    base = _base_alias_normalize(text)
    if "#" in base:
        return None
    for m in _DIGIT_RUN_RE.finditer(base):
        if int(m.group()) > _MAX_UNKNOWN_DIGIT_VALUE:
            return None
    base = base.replace("@", " at ").replace("&", " and ")
    return _strip_and_collapse(base)


# ---------------------------------------------------------------------------
# Step 1: select and label B rows.
# ---------------------------------------------------------------------------


@dataclass
class SelectedBRow:
    b_row: dict
    bucket: str  # "target_commands" | "babble" | "silence"
    label: str  # intent name | "unknown" | "silence"
    transcript: str


@dataclass
class SelectionResult:
    kept: list[SelectedBRow] = field(default_factory=list)
    drop_counts: Counter = field(default_factory=Counter)


def load_a_transcripts(a_metadata_root: Path) -> dict[str, str]:
    path = a_metadata_root / "manifests" / "all.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["filepath"]: row["transcript"] for row in csv.DictReader(f)}


def load_b_rows(b_metadata_root: Path) -> list[dict]:
    path = b_metadata_root / "manifests" / "train.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_b_provenance(b_metadata_root: Path) -> dict[str, dict]:
    path = b_metadata_root / "manifests" / "audio_provenance.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["final_filename"]: row for row in csv.DictReader(f)}


def select_b_rows(b_rows: list[dict], a_transcripts: dict[str, str]) -> SelectionResult:
    """Step 1: label + filter every B row. Command rows are kept only when
    their alias-normalized, A-joined transcript is *exactly* in-grammar for
    the label-mapped intent (D-2) -- non-matching positives are dropped,
    never kept as loss-bearing rows under the wrong label."""
    result = SelectionResult()

    for row in b_rows:
        a_text = a_transcripts.get(row["original_source"])
        if a_text is None:
            result.drop_counts["no_a_join"] += 1
            continue

        cls = row["class"]

        if cls == "SILENCE":
            result.kept.append(SelectedBRow(b_row=row, bucket="silence", label="silence", transcript=""))
            continue

        if cls == "UNKNOWN":
            normalized = normalize_unknown_text(a_text)
            if normalized is None:
                if "#" in _base_alias_normalize(a_text):
                    result.drop_counts["unknown_contains_hash"] += 1
                else:
                    result.drop_counts["unknown_digit_overflow"] += 1
                continue
            result.kept.append(
                SelectedBRow(b_row=row, bucket="babble", label="unknown", transcript=normalized)
            )
            continue

        # Command row.
        intent = LABEL_MAP.get(cls, cls)
        normalized = alias_normalize(a_text)
        accepted = OPTIONB_GRAMMAR.accepts(normalized)
        if accepted is None:
            result.drop_counts["out_of_grammar"] += 1
            continue
        got_intent, _slots = accepted[0]
        if got_intent != intent:
            result.drop_counts["accepted_other_intent"] += 1
            continue
        result.kept.append(
            SelectedBRow(b_row=row, bucket="target_commands", label=intent, transcript=normalized)
        )

    return result


# ---------------------------------------------------------------------------
# Step 2: B speaker split.
# ---------------------------------------------------------------------------


def assign_b_speaker_splits(b_rows: list[dict], seed: int) -> dict[tuple[str, str], str]:
    """Deterministic (seeded) per-`(source_dataset, speaker_id)` split
    assignment, targeting ~80/10/10 by row count *within each B
    `source_dataset` family* -- a naive `hash(speaker_id)` split was
    measured against the real data and is too lumpy per intent (NEXT came
    out 0 val / 60 test, since Multi-Sensor's 16 speakers are far too few
    for a per-speaker hash to land close to 80/10/10 by chance). Weighting
    is by each speaker's *original*-row count only: augmented rows always
    follow their speaker's split (Step 2's own rule), so weighting by them
    too would double-count the same underlying recording.

    Greedy weighted assignment: shuffle each family's speakers
    deterministically, then assign each speaker (in shuffled order) to
    whichever split is currently furthest below its target share of that
    family's total original-row count.
    """
    original_counts: Counter[tuple[str, str]] = Counter()
    for row in b_rows:
        if row["original_or_augmented"] == "original":
            original_counts[(row["source_dataset"], row["speaker_id"])] += 1

    families: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (source_dataset, speaker_id), count in original_counts.items():
        families[source_dataset].append((speaker_id, count))

    targets = {"train": 0.8, "val": 0.1, "test": 0.1}
    assignment: dict[tuple[str, str], str] = {}

    for source_dataset in sorted(families):
        speakers = sorted(families[source_dataset])
        rng = Random(f"{seed}:{source_dataset}")
        rng.shuffle(speakers)
        total_rows = sum(count for _, count in speakers)
        totals = {"train": 0, "val": 0, "test": 0}
        for speaker_id, count in speakers:
            split = max(
                targets,
                key=lambda s: (targets[s] * total_rows - totals[s], s),
            )
            totals[split] += count
            assignment[(source_dataset, speaker_id)] = split

    return assignment


# ---------------------------------------------------------------------------
# Step 2b: ESC-50 silence rebuilt by fold (D-6).
# ---------------------------------------------------------------------------


def load_esc50_rows(esc50_manifest: Path) -> list[dict]:
    with esc50_manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def freesound_id(source_file: str) -> str:
    parts = source_file.split("-")
    if len(parts) < 2:
        raise VCMXMergeError(f"ESC-50 source_file {source_file!r} has no Freesound-ID field")
    return parts[1]


def assign_esc50_splits(esc50_rows: list[dict]) -> dict[str, str]:
    """Step 2b: fold 1-3 -> train, 4 -> val, 5 -> test, resolved at the
    *Freesound-recording* level, not the individual-chunk level.

    Verified finding against the real pool (deviation from a literal
    per-chunk fold mapping, flagged per this ticket's must-verify rule):
    2 of 431 Freesound recordings have chunks in more than one official
    ESC-50 fold (`131943`: folds 2+3, both -> train, so harmless; `209698`:
    folds 4+5 -> val and test, a real train/val/test-disjointness
    violation if split per chunk). Resolving each whole Freesound ID to
    the majority fold-derived split of its own chunks (ties broken toward
    the lexicographically larger split name, deterministic) fixes this by
    construction -- `assign_esc50_splits` is per-recording, so the D-6
    disjointness guard in `_check_group_id_disjoint` cannot fire for
    background_noise as a result of this. Affects 2 of 1,633 chunks
    (`209698`'s 2 val-fold chunks move to test)."""
    by_fid: dict[str, list[str]] = defaultdict(list)
    for row in esc50_rows:
        by_fid[freesound_id(row["source_file"])].append(FOLD_TO_SPLIT[row["fold"]])

    resolved: dict[str, str] = {}
    for fid, splits in by_fid.items():
        votes = Counter(splits)
        best_count = max(votes.values())
        winners = sorted(s for s, c in votes.items() if c == best_count)
        resolved[fid] = winners[-1]
    return resolved


def probe_wav_header(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth()


# ---------------------------------------------------------------------------
# Step 3: rows in the pipeline's schema.
# ---------------------------------------------------------------------------


def _rel_path(target: Path, out_dir: Path) -> str:
    return os.path.relpath(str(target), str(out_dir))


def build_b_schema_rows(
    selection: SelectionResult,
    b_root: Path,
    b_provenance: dict[str, dict],
    speaker_splits: dict[tuple[str, str], str],
    out_dir: Path,
) -> tuple[list[dict], Counter]:
    """B rows -> final schema rows. Augmented rows whose speaker landed in
    val/test are dropped (Step 2: val/test hold originals only); SILENCE is
    forced to train regardless of its (GSC_background_noise) speaker's
    assigned split."""
    rows: list[dict] = []
    drop_counts: Counter = Counter()
    audio_dir = (b_root / "audio").resolve()

    for sel in selection.kept:
        b_row = sel.b_row
        filename = Path(b_row["filepath"]).name
        provenance = b_provenance.get(filename)
        if provenance is None:
            raise VCMXMergeError(f"no audio_provenance.csv row for B file {filename!r}")

        sr, ch, bits = int(provenance["sample_rate"]), int(provenance["channels"]), int(provenance["bit_depth"])
        if (sr, ch, bits) != (REQUIRED_SR, REQUIRED_CHANNELS, REQUIRED_SAMPWIDTH * 8):
            raise VCMXMergeError(
                f"B file {filename!r}: provenance format {sr}Hz/{ch}ch/{bits}bit != required "
                f"{REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/{REQUIRED_SAMPWIDTH * 8}bit"
            )

        family = b_row["source_dataset"]
        speaker_split = speaker_splits.get((family, b_row["speaker_id"]))
        if speaker_split is None:
            raise VCMXMergeError(f"no split assigned for B speaker {(family, b_row['speaker_id'])!r}")

        if sel.bucket == "silence":
            split = "train"
        elif b_row["original_or_augmented"] == "augmented" and speaker_split != "train":
            drop_counts["augmented_dropped_val_test"] += 1
            continue
        else:
            split = speaker_split

        target = (audio_dir / filename).resolve()
        rows.append(
            {
                "filename": filename,
                "path": _rel_path(target, out_dir),
                "bucket": sel.bucket,
                "label": sel.label,
                "duration": f"{float(provenance['duration']):.6f}",
                "sample_rate": str(REQUIRED_SR),
                "resampled": "False",
                "source_dataset": "vcm_balanced",
                "source_relpath": b_row["original_source"],
                "group_id": b_row["speaker_id"],
                "split": split,
                "transcript": sel.transcript,
                "original_dataset": family,
            }
        )

    return rows, drop_counts


def build_esc50_schema_rows(
    esc50_rows: list[dict],
    esc50_manifest_path: Path,
    fid_splits: dict[str, str],
    out_dir: Path,
) -> list[dict]:
    audio_dir = (esc50_manifest_path.parent / "audio").resolve()
    rows: list[dict] = []
    for row in esc50_rows:
        filename = row["filename"]
        wav_path = audio_dir / filename
        sr, ch, sampwidth = probe_wav_header(wav_path)
        if (sr, ch, sampwidth) != (REQUIRED_SR, REQUIRED_CHANNELS, REQUIRED_SAMPWIDTH):
            raise VCMXMergeError(
                f"ESC-50 chunk {filename!r}: WAV header {sr}Hz/{ch}ch/{sampwidth * 8}bit != required "
                f"{REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/{REQUIRED_SAMPWIDTH * 8}bit"
            )

        fid = freesound_id(row["source_file"])
        rows.append(
            {
                "filename": filename,
                "path": _rel_path((audio_dir / filename).resolve(), out_dir),
                "bucket": "silence",
                "label": "silence",
                "duration": f"{float(row['duration']):.6f}",
                "sample_rate": str(REQUIRED_SR),
                "resampled": "False",
                "source_dataset": "background_noise",
                "source_relpath": row["source_file"],
                "group_id": fid,
                "split": fid_splits[fid],
                "transcript": "",
                "original_dataset": row["category"],
            }
        )
    return rows


def build_m0_schema_rows(m0_rows: list[dict], m0_dir: Path, out_dir: Path) -> list[dict]:
    """M0 (`optionb-v3/manifest.csv`) rows, `path` rewritten relative to
    the new output dir. Step 2b: every M0 row with
    `source_dataset == "background_noise"` is dropped here -- it is
    replaced wholesale by the fold-rebuilt ESC-50 pool
    (`build_esc50_schema_rows`), never merged with it."""
    rows: list[dict] = []
    for row in m0_rows:
        if row["source_dataset"] == "background_noise":
            continue
        new_row = dict(row)
        abs_path = os.path.normpath(os.path.join(str(m0_dir), row["path"]))
        new_row["path"] = _rel_path(Path(abs_path), out_dir)
        new_row["original_dataset"] = ""
        rows.append(new_row)
    return rows


# M0-carried reject-probe families whose group_id is a real per-speaker/
# per-session id (unlike `common_voice_negative`, whose group_id is not a
# real speaker id at all -- see `resolve_m0_group_splits`'s docstring).
_M0_GROUP_RESOLVE_FAMILIES = frozenset({"filipino_speech_corpus", "youtube_institutional"})

# Tie-break order when a group_id's rows are evenly split across splits
# (lower = preferred): val first, since these are reject probes and val is
# what `vcm.evaluate`'s threshold sweep actually reads; then test; train
# last (train is exactly the split most immune to biasing an operating
# threshold, so it absorbs ties least deserving of "wins").
_SPLIT_TIEBREAK_ORDER = {"val": 0, "test": 1, "train": 2}


def resolve_m0_group_splits(rows: list[dict]) -> list[dict]:
    """Fixes real M0-carried group_id-spans-split leakage found against the
    refreshed data: 38 `filipino_speech_corpus` group_ids and 6
    `youtube_institutional` group_ids span more than one split (inherited
    unmodified from `test_set/`, predating this ticket). Every leaking
    `(source_dataset, group_id)` is reassigned, in full, to the split
    holding the most of its own rows -- ties (all 38 `filipino_speech_corpus`
    cases are exact 1-vs-1 ties) broken toward val over test over train
    (`_SPLIT_TIEBREAK_ORDER`).

    Applied exactly once, to the shared M0 row set `build()` passes to both
    treatment and control before they ever diverge, so both manifests see
    the identical reassignment and stay val/test-identical (D-7).

    `common_voice_negative` is deliberately excluded: its group_id is not a
    real per-speaker id (`docs/VCM-CONTRACT.md` section 4's `source_relpath`-
    keyed join has no per-speaker concept for it), the same convention
    `vcm.evaluate.classify_speaker_group` already encodes by excluding a
    falsy group_id from grouping rather than treating it as one giant
    group -- grouping `common_voice_negative` rows by a shared/blank
    group_id here would conflate unrelated rows under one "group" and
    reassign far more rows than the real leak actually touches.
    """
    by_key: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        gid = row["group_id"]
        if not gid or row["source_dataset"] not in _M0_GROUP_RESOLVE_FAMILIES:
            continue
        by_key[(row["source_dataset"], gid)][row["split"]] += 1

    resolved_split: dict[tuple[str, str], str] = {}
    for key, counts in by_key.items():
        if len(counts) <= 1:
            continue  # already single-split; nothing to resolve
        best_count = max(counts.values())
        tied = [split for split, count in counts.items() if count == best_count]
        resolved_split[key] = min(tied, key=lambda s: _SPLIT_TIEBREAK_ORDER[s])

    if not resolved_split:
        return rows

    out = []
    for row in rows:
        key = (row["source_dataset"], row["group_id"])
        if key in resolved_split and row["split"] != resolved_split[key]:
            row = dict(row)
            row["split"] = resolved_split[key]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Step 4: fail-loud checks.
# ---------------------------------------------------------------------------


def _check_files_exist(rows: list[dict], out_dir: Path) -> None:
    missing = []
    for row in rows:
        resolved = (out_dir / row["path"]).resolve()
        if not resolved.is_file():
            missing.append((row["filename"], row["path"]))
        if len(missing) >= 20:
            break
    if missing:
        raise VCMXMergeError(f"{len(missing)}+ manifest rows reference missing files, e.g. {missing[:5]}")


def _check_group_id_disjoint(rows: list[dict]) -> None:
    """Whole-manifest, every source_dataset family with a non-empty
    group_id. Previously scoped to `vcm_balanced`/`background_noise` only,
    to work around 44 pre-existing `filipino_speech_corpus`/
    `youtube_institutional` violations inherited from `test_set/` --
    `resolve_m0_group_splits` now fixes those before this check ever runs
    (called once in `build()`, ahead of treatment/control diverging), so
    the check no longer needs to be scoped around them. A falsy group_id
    (e.g. `common_voice_negative`'s, not a real per-speaker id) is still
    excluded, matching `vcm.evaluate.classify_speaker_group`'s own
    convention of excluding falsy group_id from grouping rather than
    treating it as one giant group.

    `ref_`-prefixed group ids carry the Phase-1 scoped exception
    (accent-balance references voices, usable in every split because their
    timbre already spans every split via converted wakeword positives);
    every other group id stays strictly enforced."""
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        gid = row["group_id"]
        if not gid or gid.startswith("ref_"):
            continue
        by_key[(row["source_dataset"], gid)].add(row["split"])
    violations = {k: v for k, v in by_key.items() if len(v) > 1}
    if violations:
        raise VCMXMergeError(f"group_id spans more than one split within a source_dataset family: {violations}")


def _check_freesound_disjoint(rows: list[dict]) -> None:
    by_fid: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["source_dataset"] != "background_noise":
            continue
        by_fid[row["group_id"]].add(row["split"])
    violations = {k: v for k, v in by_fid.items() if len(v) > 1}
    if violations:
        raise VCMXMergeError(f"Freesound ID spans more than one split: {violations}")


def _check_transcripts_encode(rows: list[dict]) -> None:
    failures = []
    for row in rows:
        text = resolve_transcript(row)
        if text is None:
            continue
        try:
            alphabet.encode(normalize_text(text))
        except ValueError as exc:
            failures.append((row["filename"], text, str(exc)))
        if len(failures) >= 20:
            break
    if failures:
        raise VCMXMergeError(f"{len(failures)}+ rows failed normalize_text/alphabet.encode, e.g. {failures[:5]}")


def _ctc_feasible(duration: float, target_ids: list[int]) -> bool:
    frames = math.floor(duration * REQUIRED_SR / 160) + 1
    repeats = sum(1 for i in range(1, len(target_ids)) if target_ids[i] == target_ids[i - 1])
    return frames >= len(target_ids) + repeats


def _check_ctc_feasible(rows: list[dict]) -> None:
    failures = []
    for row in rows:
        text = resolve_transcript(row)
        if text is None:
            continue
        ids = alphabet.encode(normalize_text(text))
        if not _ctc_feasible(float(row["duration"]), ids):
            failures.append((row["filename"], row["duration"], text))
        if len(failures) >= 20:
            break
    if failures:
        raise VCMXMergeError(f"{len(failures)}+ rows are CTC-infeasible (frames < target+repeats), e.g. {failures[:5]}")


def run_fail_loud_checks(rows: list[dict], out_dir: Path) -> None:
    _check_files_exist(rows, out_dir)
    _check_group_id_disjoint(rows)
    _check_freesound_disjoint(rows)
    _check_transcripts_encode(rows)
    _check_ctc_feasible(rows)


# ---------------------------------------------------------------------------
# Step 5: write outputs.
# ---------------------------------------------------------------------------


def write_manifest(out_dir: Path, rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def build_control_rows(treatment_rows: list[dict]) -> list[dict]:
    return [
        row
        for row in treatment_rows
        if not (row["source_dataset"] == "vcm_balanced" and row["split"] == "train")
    ]


def _val_test_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["split"] in ("val", "test")]


def assert_val_test_identical(treatment_rows: list[dict], control_rows: list[dict]) -> None:
    t = _val_test_rows(treatment_rows)
    c = _val_test_rows(control_rows)
    if t != c:
        raise VCMXMergeError("val/test rows differ between treatment and control manifests")


def build_report(
    treatment_rows: list[dict],
    control_rows: list[dict],
    drop_counts: Counter,
    seed: int,
) -> str:
    lines = ["# vcmx_merge build report", "", f"Seed: `{seed}`", ""]

    lines += ["## Bucket x split (treatment)", ""]
    lines += ["| bucket | split | rows |", "|---|---|---:|"]
    bucket_split = Counter((r["bucket"], r["split"]) for r in treatment_rows)
    for (bucket, split), count in sorted(bucket_split.items()):
        lines.append(f"| {bucket} | {split} | {count} |")
    lines.append("")

    lines += ["## Intent x split x source_dataset (target_commands only)", ""]
    lines += ["| intent | split | source_dataset | rows |", "|---|---|---|---:|"]
    intent_split_source = Counter(
        (r["label"], r["split"], r["source_dataset"])
        for r in treatment_rows
        if r["bucket"] == "target_commands"
    )
    for (intent, split, source), count in sorted(intent_split_source.items()):
        lines.append(f"| {intent} | {split} | {source} | {count} |")
    lines.append("")

    lines += ["## Drop reasons (B row selection, Step 1/3)", ""]
    lines += ["| reason | rows |", "|---|---:|"]
    for reason, count in sorted(drop_counts.items()):
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    b_next = [
        r
        for r in treatment_rows
        if r["source_dataset"] == "vcm_balanced" and r["label"] == "NEXT" and r["bucket"] == "target_commands"
    ]
    b_next_song = [r for r in b_next if r["transcript"] == "next song"]
    b_commands_total = sum(
        1 for r in treatment_rows if r["source_dataset"] == "vcm_balanced" and r["bucket"] == "target_commands"
    )
    lines += ["## \"next song\" share (D: not capped, flagged per ticket non-goal)", ""]
    if b_commands_total:
        share_of_commands = len(b_next_song) / b_commands_total
        lines.append(
            f"- B's NEXT/\"next song\" (Multi-Sensor) contributes {len(b_next_song)} of "
            f"{b_commands_total} B in-grammar commands ({share_of_commands:.1%}). "
            "Not capped -- this is expected, since B has zero coverage for most other "
            "intents; kept as-is per this ticket's non-goals."
        )
    lines.append("")

    lines += ["## Row counts", ""]
    lines.append(f"- Treatment manifest: {len(treatment_rows)} rows.")
    lines.append(f"- Control manifest: {len(control_rows)} rows.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build(
    *,
    optionb_manifest: Path,
    b_root: Path,
    b_metadata_root: Path,
    a_metadata_root: Path,
    esc50_manifest: Path,
    out_treatment: Path,
    out_control: Path,
    seed: int,
    dry_run: bool,
) -> None:
    a_transcripts = load_a_transcripts(a_metadata_root)
    b_rows = load_b_rows(b_metadata_root)
    b_provenance = load_b_provenance(b_metadata_root)

    selection = select_b_rows(b_rows, a_transcripts)
    speaker_splits = assign_b_speaker_splits(b_rows, seed)

    esc50_rows = load_esc50_rows(esc50_manifest)
    fid_splits = assign_esc50_splits(esc50_rows)

    with optionb_manifest.open(newline="", encoding="utf-8") as f:
        m0_rows = list(csv.DictReader(f))

    b_schema_rows, b_drop_counts = build_b_schema_rows(
        selection, b_root, b_provenance, speaker_splits, out_treatment
    )
    esc50_schema_rows = build_esc50_schema_rows(esc50_rows, esc50_manifest, fid_splits, out_treatment)
    m0_schema_rows = build_m0_schema_rows(m0_rows, optionb_manifest.parent, out_treatment)
    m0_schema_rows = resolve_m0_group_splits(m0_schema_rows)

    treatment_rows = m0_schema_rows + b_schema_rows + esc50_schema_rows

    drop_counts = Counter(selection.drop_counts)
    drop_counts.update(b_drop_counts)

    control_rows = build_control_rows(treatment_rows)
    # Paths embedded in each row are computed relative to `out_treatment`
    # above; recompute them relative to `out_control` too, unless the two
    # output dirs are the same depth (the default case, and this ticket's
    # only supported configuration) in which case the strings are already
    # identical -- verified by `assert_val_test_identical` below rather
    # than assumed.
    if out_treatment.resolve() != out_control.resolve():
        control_out_rows: list[dict] = []
        for row in control_rows:
            abs_path = Path(os.path.normpath(os.path.join(str(out_treatment), row["path"])))
            new_row = dict(row)
            new_row["path"] = _rel_path(abs_path, out_control)
            control_out_rows.append(new_row)
        control_rows = control_out_rows

    if dry_run:
        print(
            f"dry-run: would write {len(treatment_rows)} treatment rows -> {out_treatment}, "
            f"{len(control_rows)} control rows -> {out_control}"
        )
        return

    run_fail_loud_checks(treatment_rows, out_treatment)
    run_fail_loud_checks(control_rows, out_control)
    assert_val_test_identical(treatment_rows, control_rows)

    write_manifest(out_treatment, treatment_rows)
    write_manifest(out_control, control_rows)

    report = build_report(treatment_rows, control_rows, drop_counts, seed)
    (out_treatment / "report.md").write_text(report, encoding="utf-8")
    (out_control / "report.md").write_text(report, encoding="utf-8")

    print(f"{len(treatment_rows)} treatment rows -> {out_treatment}")
    print(f"{len(control_rows)} control rows -> {out_control}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build the merged treatment/control manifests")
    build_parser.add_argument(
        "--optionb-manifest",
        type=Path,
        default=PROJECT_ROOT / "out" / "conversions" / "v2" / "optionb-v3" / "manifest.csv",
    )
    build_parser.add_argument("--b-root", type=Path, default=PROJECT_ROOT / "raw_datasets" / "VCM_BALANCED")
    build_parser.add_argument(
        "--b-metadata-root", type=Path, default=PROJECT_ROOT / "raw_datasets" / "VCM_BALANCED_METADATA"
    )
    build_parser.add_argument(
        "--a-metadata-root", type=Path, default=PROJECT_ROOT / "raw_datasets" / "VCM_MASTER_METADATA"
    )
    build_parser.add_argument(
        "--esc50-manifest",
        type=Path,
        default=PROJECT_ROOT / "out" / "conversions" / "v2" / "background_noise" / "manifest.csv",
    )
    build_parser.add_argument(
        "--out-treatment",
        type=Path,
        default=PROJECT_ROOT / "out" / "conversions" / "v2" / "optionb-v3-vcmx",
    )
    build_parser.add_argument(
        "--out-control",
        type=Path,
        default=PROJECT_ROOT / "out" / "conversions" / "v2" / "optionb-v3-vcmx-control",
    )
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        try:
            build(
                optionb_manifest=args.optionb_manifest,
                b_root=args.b_root,
                b_metadata_root=args.b_metadata_root,
                a_metadata_root=args.a_metadata_root,
                esc50_manifest=args.esc50_manifest,
                out_treatment=args.out_treatment,
                out_control=args.out_control,
                seed=args.seed,
                dry_run=args.dry_run,
            )
        except VCMXMergeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    raise VCMXMergeError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
