"""Assemble a standalone KWS (keyword-spotting) test set from the five
out/conversions/v2 corpora: sanitized clean commands (20 intents), two babble
speech sources (common_voice_negative, filipino_speech_corpus), a third babble
+ ambient-silence source (youtube_institutional), and an ESC-50-derived
silence/noise source (background_noise).

Allocation is computed, not hardcoded: `commands_per_intent` is the minimum
per-intent pool across the 20 clean-command intents (73 on the real corpus,
CALL's pool), so `target_commands_total = commands_per_intent * n_intents`.
The remaining ~35% (babble + silence) is derived from that via
`total = round(target_commands_total / 0.65)`, then split babble:silence =
25:10 and further down to individual sources/classes, all via largest-
remainder allocation - see compute_allocation().

Every source has its own manifest schema (or, for sanitized/clean, no
manifest at all); adapters normalize each into a common `Candidate` shape
before sampling and emission. csv.DictReader/DictWriter is mandatory for any
manifest with free-text columns (sentence/transcript carry embedded commas -
naive splitting silently corrupts counts).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import statistics
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import librosa
import soundfile as sf

REQUIRED_SR = 16000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPWIDTH = 2

BABBLE_SILENCE_WEIGHTS = {"babble": 25, "silence": 10}
BABBLE_SOURCE_WEIGHTS = {
    "common_voice_negative": 1,
    "filipino_speech_corpus": 1,
    "youtube_institutional": 1,
}

CLEAN_FILENAME_RE = re.compile(r"^(?P<intent>[A-Z_]+)_s(?P<s>\d+)_p(?P<p>\d+)_v(?P<v>\d+)\.wav$")

LICENSES = {
    "sanitized/clean": "synthesized in-repo (CosyVoice2 TTS output) - no external license restriction",
    "common_voice_negative": "CC-0 (Mozilla Common Voice)",
    "filipino_speech_corpus": "see filipino_speech_corpus source README for exact terms",
    "youtube_institutional": "institutional YouTube uploads - review per-video terms before redistribution",
    "background_noise": "CC-BY-NC-SA-4.0 (ESC-50) - GOVERNS THE WHOLE ASSEMBLED SET as the most restrictive license present",
}


@dataclass(frozen=True)
class Candidate:
    filename: str
    audio_path: Path
    duration: float
    orig_sample_rate: int
    group_id: str
    source_dataset: str
    source_relpath: str
    needs_resample: bool


class AllocationError(RuntimeError):
    pass


class PoolExhaustedError(RuntimeError):
    pass


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Apportion an integer `total` across `weights.keys()` proportional to
    weight, using largest-remainder rounding. Deterministic tie-break: ties in
    remainder go to the alphabetically earlier key first."""
    if total < 0:
        raise AllocationError(f"cannot allocate negative total {total}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise AllocationError(f"weights must sum to a positive number, got {weights}")

    exact = {k: total * w / weight_sum for k, w in weights.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = total - sum(floors.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - floors[k]), k))
    for k in order[:remainder]:
        floors[k] += 1
    assert sum(floors.values()) == total
    return floors


def compute_allocation(intent_pools: dict[str, int], bg_class_pools: dict[str, int], yt_ambient_pool: int) -> dict:
    n_intents = len(intent_pools)
    if n_intents == 0:
        raise AllocationError("no clean-command intents found")
    commands_per_intent = min(intent_pools.values())
    target_commands_total = commands_per_intent * n_intents

    total = round(target_commands_total / 0.65)
    residual = total - target_commands_total
    babble_silence = largest_remainder(BABBLE_SILENCE_WEIGHTS, residual)
    babble_total = babble_silence["babble"]
    silence_total = babble_silence["silence"]

    babble_split = largest_remainder(BABBLE_SOURCE_WEIGHTS, babble_total)

    silence_weights = {"background_noise": sum(bg_class_pools.values()), "youtube_institutional": yt_ambient_pool}
    silence_split = largest_remainder(silence_weights, silence_total)

    esc_weights = {cls: 1 for cls in bg_class_pools}
    esc_split = largest_remainder(esc_weights, silence_split["background_noise"])
    for cls, target in esc_split.items():
        if target > bg_class_pools[cls]:
            raise PoolExhaustedError(
                f"background_noise class {cls!r}: requested {target} exceeds pool {bg_class_pools[cls]} "
                f"(short by {target - bg_class_pools[cls]})"
            )

    allocation = {
        "commands_per_intent": commands_per_intent,
        "n_intents": n_intents,
        "target_commands_total": target_commands_total,
        "total": total,
        "babble_total": babble_total,
        "silence_total": silence_total,
        "babble_split": babble_split,
        "silence_split": silence_split,
        "esc_split": esc_split,
    }

    if commands_per_intent == 73 and n_intents == 20:
        expected = {
            "total": 2246,
            "target_commands_total": 1460,
            "babble_total": 561,
            "silence_total": 225,
            "babble_split": {
                "common_voice_negative": 187,
                "filipino_speech_corpus": 187,
                "youtube_institutional": 187,
            },
            "silence_split": {"background_noise": 194, "youtube_institutional": 31},
        }
        for key, want in expected.items():
            got = allocation[key]
            if got != want:
                raise AllocationError(
                    f"allocator self-check failed for {key!r}: computed {got!r}, expected {want!r} "
                    f"(this indicates a bug in the allocator, not a data change)"
                )
        if sum(esc_split.values()) != 194:
            raise AllocationError(f"ESC-50 class split does not sum to 194: {esc_split}")

    return allocation


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_intents(clean_dir: Path) -> list[str]:
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"sanitized clean dir not found: {clean_dir}")
    return sorted(p.name for p in clean_dir.iterdir() if p.is_dir())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def build_clean_candidates(clean_dir: Path, intent: str) -> list[Candidate]:
    intent_dir = clean_dir / intent
    candidates = []
    for wav_path in sorted(intent_dir.glob("*.wav")):
        m = CLEAN_FILENAME_RE.match(wav_path.name)
        if not m or m.group("intent") != intent:
            raise ValueError(f"{wav_path}: filename does not match <INTENT>_s<N>_p1_v<N>.wav pattern")
        candidates.append(
            Candidate(
                filename=wav_path.name,
                audio_path=wav_path,
                duration=wav_duration(wav_path),
                orig_sample_rate=REQUIRED_SR,
                group_id=f"s{m.group('s')}",
                source_dataset="sanitized_clean",
                source_relpath=f"clean/{intent}/{wav_path.name}",
                needs_resample=False,
            )
        )
    return candidates


def _resolve_manifest(source_dir: Path) -> Path:
    for candidate in (source_dir / "manifest.csv", source_dir / "audio" / "manifest.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no manifest.csv found under {source_dir} or {source_dir / 'audio'}")


def build_cv_candidates(source_dir: Path) -> list[Candidate]:
    manifest = _resolve_manifest(source_dir)
    audio_dir = source_dir / "audio"
    candidates = []
    for row in read_manifest_rows(manifest):
        candidates.append(
            Candidate(
                filename=row["filename"],
                audio_path=audio_dir / row["filename"],
                duration=float(row["duration"]),
                orig_sample_rate=48000,
                group_id=row["client_id"],
                source_dataset="common_voice_negative",
                source_relpath=f"audio/{row['filename']}",
                needs_resample=True,
            )
        )
    return candidates


def build_fsc_candidates(source_dir: Path) -> list[Candidate]:
    manifest = _resolve_manifest(source_dir)
    audio_dir = source_dir / "audio"
    candidates = []
    for row in read_manifest_rows(manifest):
        candidates.append(
            Candidate(
                filename=row["filename"],
                audio_path=audio_dir / row["filename"],
                duration=float(row["duration"]),
                orig_sample_rate=REQUIRED_SR,
                group_id=row["speaker_id"],
                source_dataset="filipino_speech_corpus",
                source_relpath=f"audio/{row['filename']}",
                needs_resample=False,
            )
        )
    return candidates


def build_yt_candidates(source_dir: Path, bucket: str) -> list[Candidate]:
    manifest = _resolve_manifest(source_dir)
    clips_dir = source_dir / "clips" / bucket
    candidates = []
    for row in read_manifest_rows(manifest):
        if row["bucket"] != bucket:
            continue
        candidates.append(
            Candidate(
                filename=row["filename"],
                audio_path=clips_dir / row["filename"],
                duration=float(row["duration"]),
                orig_sample_rate=REQUIRED_SR,
                group_id=row["video_id"],
                source_dataset="youtube_institutional",
                source_relpath=f"clips/{bucket}/{row['filename']}",
                needs_resample=False,
            )
        )
    return candidates


def build_bg_candidates(source_dir: Path) -> dict[str, list[Candidate]]:
    manifest = _resolve_manifest(source_dir)
    audio_dir = source_dir / "audio"
    by_class: dict[str, list[Candidate]] = {}
    for row in read_manifest_rows(manifest):
        cls = row["category"]
        by_class.setdefault(cls, []).append(
            Candidate(
                filename=row["filename"],
                audio_path=audio_dir / row["filename"],
                duration=float(row["duration"]),
                orig_sample_rate=REQUIRED_SR,
                group_id=row["source_file"],
                source_dataset="background_noise",
                source_relpath=f"audio/{row['filename']}",
                needs_resample=False,
            )
        )
    return by_class


def grouped_sample(rng: random.Random, items: list[Candidate], n: int) -> tuple[list[Candidate], int]:
    """Draw n items, capping per-group_id draws at ceil(n/n_groups) where
    feasible; relaxes the cap (flat fallback) only if capped capacity can't
    reach n. Returns (selected, max_drawn_from_any_one_group)."""
    if n > len(items):
        raise PoolExhaustedError(f"requested {n} items but pool has only {len(items)}")
    if n == 0:
        return [], 0

    by_group: dict[str, list[Candidate]] = {}
    for item in items:
        by_group.setdefault(item.group_id, []).append(item)
    groups = sorted(by_group)
    for g in groups:
        rng.shuffle(by_group[g])

    n_groups = len(groups)
    cap = math.ceil(n / n_groups)
    alloc = {g: 0 for g in groups}
    remaining = n
    for g in groups:
        take = min(cap, len(by_group[g]), remaining)
        alloc[g] = take
        remaining -= take

    while remaining > 0:
        progressed = False
        for g in groups:
            if remaining <= 0:
                break
            if alloc[g] < len(by_group[g]):
                alloc[g] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    if remaining > 0:
        raise PoolExhaustedError(f"could not satisfy request of {n} items even relaxing per-group cap (pool={len(items)})")

    selected: list[Candidate] = []
    for g in groups:
        selected.extend(by_group[g][: alloc[g]])
    rng.shuffle(selected)
    return selected, max(alloc.values())


def write_clip(candidate: Candidate, dest_path: Path) -> tuple[int, int, int]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate.needs_resample:
        data, sr = librosa.load(str(candidate.audio_path), sr=REQUIRED_SR, mono=True)
        sf.write(str(dest_path), data, sr, subtype="PCM_16")
    else:
        shutil.copy2(candidate.audio_path, dest_path)

    with wave.open(str(dest_path), "rb") as w:
        sr_w, ch_w, sw_w = w.getframerate(), w.getnchannels(), w.getsampwidth()
    if (sr_w, ch_w, sw_w) != (REQUIRED_SR, REQUIRED_CHANNELS, REQUIRED_SAMPWIDTH):
        raise RuntimeError(
            f"{dest_path}: written format {sr_w}Hz/{ch_w}ch/{sw_w * 8}bit != "
            f"required {REQUIRED_SR}Hz/{REQUIRED_CHANNELS}ch/{REQUIRED_SAMPWIDTH * 8}bit"
        )
    return sr_w, ch_w, sw_w


def emit_stratum(
    rng: random.Random,
    candidates: list[Candidate],
    n: int,
    out_root: Path,
    bucket: str,
    label: str,
    stratum_key: str,
    used_filenames: set[str],
    manifest_rows: list[dict],
    group_concentration: dict[str, int],
    realized_counts: dict[str, int],
    pool_group_counts: dict[str, int],
) -> None:
    selected, max_group = grouped_sample(rng, candidates, n)
    group_concentration[stratum_key] = max_group
    pool_group_counts[stratum_key] = len({c.group_id for c in candidates})

    dest_dir = out_root / "audio" / bucket if label in ("unknown", "silence") else out_root / "audio" / "target_commands" / label
    for cand in selected:
        if cand.filename in used_filenames:
            raise ValueError(f"duplicate clip filename across emitted set: {cand.filename}")
        used_filenames.add(cand.filename)

        dest_path = dest_dir / cand.filename
        write_clip(cand, dest_path)
        duration = wav_duration(dest_path)

        manifest_rows.append(
            {
                "filename": cand.filename,
                "path": str(dest_path.relative_to(out_root)),
                "bucket": bucket,
                "label": label,
                "duration": f"{duration:.6f}",
                "sample_rate": str(cand.orig_sample_rate),
                "resampled": str(cand.needs_resample),
                "source_dataset": cand.source_dataset,
                "source_relpath": cand.source_relpath,
                "group_id": cand.group_id,
            }
        )
        realized_counts[stratum_key] = realized_counts.get(stratum_key, 0) + 1


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
]


def write_manifest(out_root: Path, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["bucket"], r["label"], r["filename"]))
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows_sorted)


def write_summary(
    out_root: Path,
    seed: int,
    allocation: dict,
    realized_counts: dict[str, int],
    group_concentration: dict[str, int],
    duration_stats: dict[str, float],
    pool_group_counts: dict[str, int],
) -> None:
    lines = ["# KWS test set summary", "", f"Seed: `{seed}`", ""]

    lines += ["## Realized counts", ""]
    lines += ["| Stratum | Count |", "|---|---|"]
    for key in sorted(realized_counts):
        lines.append(f"| {key} | {realized_counts[key]} |")
    total_realized = sum(realized_counts.values())
    lines += [f"| **total** | **{total_realized}** |", ""]

    lines += ["## Per-source licenses", ""]
    for source, lic in LICENSES.items():
        lines.append(f"- `{source}`: {lic}")
    lines += [
        "",
        f"**Governing restriction for the whole assembled set: {LICENSES['background_noise']}**",
        "",
    ]

    lines += [
        "## Confounds (documented, not corrected)",
        "",
        "- **Single-persona confound**: all target-command clips come from a single "
        "synthetic TTS voice (`p1` token universal across the 2347-clip sanitized/clean "
        "corpus) — this test set cannot measure cross-speaker generalization for the "
        "command class, only babble/silence speaker/source diversity.",
        f"- **Duration confound**: target-command clips (median "
        f"{duration_stats.get('target_commands', float('nan')):.3f}s) are markedly shorter "
        f"than babble (median {duration_stats.get('babble', float('nan')):.3f}s) and silence "
        f"(median {duration_stats.get('silence', float('nan')):.3f}s) clips; this length "
        "difference is a potential shortcut feature for a KWS classifier and is not corrected "
        "here (no padding/cropping - see Non-Goals).",
        "",
    ]

    lines += ["## Per-group_id concentration (max clips drawn from any one group)", ""]
    lines += ["| Stratum | Max from one group |", "|---|---|"]
    for key in sorted(group_concentration):
        n_groups = pool_group_counts.get(key)
        flag = ""
        if key == "babble/youtube_institutional":
            flag = f" (draws from only {n_groups} distinct video_ids - structurally low diversity)"
        lines.append(f"| {key} | {group_concentration[key]}{flag} |")
    lines.append("")

    lines += ["## Allocation", ""]
    lines.append(f"- commands_per_intent (anchor) = {allocation['commands_per_intent']}, n_intents = {allocation['n_intents']}")
    lines.append(f"- target_commands_total = {allocation['target_commands_total']}")
    lines.append(f"- total = {allocation['total']} (target_commands / 0.65)")
    lines.append(f"- babble_total = {allocation['babble_total']}, silence_total = {allocation['silence_total']}")
    lines.append(f"- babble split: {allocation['babble_split']}")
    lines.append(f"- silence split: {allocation['silence_split']}")
    lines.append(f"- ESC-50 class split (background_noise): {allocation['esc_split']}")
    lines.append("")

    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="random seed (required for reproducibility)")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("out/conversions/v2"),
        help="root containing sanitized/, common_voice_negative/, filipino_speech_corpus/, "
        "youtube_institutional/, background_noise/ (default: out/conversions/v2)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir (default: <corpus-root>/test_set); removed and recreated if it exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus_root: Path = args.corpus_root
    out_root: Path = args.out_dir if args.out_dir is not None else corpus_root / "test_set"

    clean_dir = corpus_root / "sanitized" / "clean"
    cv_dir = corpus_root / "common_voice_negative"
    fsc_dir = corpus_root / "filipino_speech_corpus"
    yt_dir = corpus_root / "youtube_institutional"
    bg_dir = corpus_root / "background_noise"

    try:
        intents = discover_intents(clean_dir)
        intent_candidates = {intent: build_clean_candidates(clean_dir, intent) for intent in intents}
        intent_pools = {intent: len(cands) for intent, cands in intent_candidates.items()}

        cv_candidates = build_cv_candidates(cv_dir)
        fsc_candidates = build_fsc_candidates(fsc_dir)
        yt_babble_candidates = build_yt_candidates(yt_dir, "babble")
        yt_ambient_candidates = build_yt_candidates(yt_dir, "ambient")
        bg_candidates_by_class = build_bg_candidates(bg_dir)
        bg_class_pools = {cls: len(cands) for cls, cands in bg_candidates_by_class.items()}

        allocation = compute_allocation(intent_pools, bg_class_pools, len(yt_ambient_candidates))
    except (FileNotFoundError, ValueError, AllocationError, PoolExhaustedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    babble_pools = {
        "common_voice_negative": len(cv_candidates),
        "filipino_speech_corpus": len(fsc_candidates),
        "youtube_institutional": len(yt_babble_candidates),
    }
    for source, requested in allocation["babble_split"].items():
        if requested > babble_pools[source]:
            print(
                f"error: babble/{source}: requested {requested} exceeds pool {babble_pools[source]} "
                f"(short by {requested - babble_pools[source]})",
                file=sys.stderr,
            )
            return 1
    if allocation["silence_split"]["youtube_institutional"] > len(yt_ambient_candidates):
        requested = allocation["silence_split"]["youtube_institutional"]
        print(
            f"error: silence/youtube_institutional: requested {requested} exceeds pool "
            f"{len(yt_ambient_candidates)} (short by {requested - len(yt_ambient_candidates)})",
            file=sys.stderr,
        )
        return 1
    for intent, requested in ((i, allocation["commands_per_intent"]) for i in intents):
        if requested > intent_pools[intent]:
            print(
                f"error: target_commands/{intent}: requested {requested} exceeds pool {intent_pools[intent]}",
                file=sys.stderr,
            )
            return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    rng = random.Random(args.seed)
    used_filenames: set[str] = set()
    manifest_rows: list[dict] = []
    group_concentration: dict[str, int] = {}
    realized_counts: dict[str, int] = {}
    pool_group_counts: dict[str, int] = {}

    for intent in intents:
        emit_stratum(
            rng,
            intent_candidates[intent],
            allocation["commands_per_intent"],
            out_root,
            bucket="target_commands",
            label=intent,
            stratum_key=f"target_commands/{intent}",
            used_filenames=used_filenames,
            manifest_rows=manifest_rows,
            group_concentration=group_concentration,
            realized_counts=realized_counts,
            pool_group_counts=pool_group_counts,
        )

    babble_sources = {
        "common_voice_negative": cv_candidates,
        "filipino_speech_corpus": fsc_candidates,
        "youtube_institutional": yt_babble_candidates,
    }
    for source, cands in babble_sources.items():
        emit_stratum(
            rng,
            cands,
            allocation["babble_split"][source],
            out_root,
            bucket="babble",
            label="unknown",
            stratum_key=f"babble/{source}",
            used_filenames=used_filenames,
            manifest_rows=manifest_rows,
            group_concentration=group_concentration,
            realized_counts=realized_counts,
            pool_group_counts=pool_group_counts,
        )

    for cls, n in allocation["esc_split"].items():
        emit_stratum(
            rng,
            bg_candidates_by_class[cls],
            n,
            out_root,
            bucket="silence",
            label="silence",
            stratum_key=f"silence/background_noise/{cls}",
            used_filenames=used_filenames,
            manifest_rows=manifest_rows,
            group_concentration=group_concentration,
            realized_counts=realized_counts,
            pool_group_counts=pool_group_counts,
        )
    emit_stratum(
        rng,
        yt_ambient_candidates,
        allocation["silence_split"]["youtube_institutional"],
        out_root,
        bucket="silence",
        label="silence",
        stratum_key="silence/youtube_institutional",
        used_filenames=used_filenames,
        manifest_rows=manifest_rows,
        group_concentration=group_concentration,
        realized_counts=realized_counts,
        pool_group_counts=pool_group_counts,
    )

    write_manifest(out_root, manifest_rows)

    duration_stats = {}
    for bucket_group, rows in (
        ("target_commands", [r for r in manifest_rows if r["bucket"] == "target_commands"]),
        ("babble", [r for r in manifest_rows if r["bucket"] == "babble"]),
        ("silence", [r for r in manifest_rows if r["bucket"] == "silence"]),
    ):
        if rows:
            duration_stats[bucket_group] = statistics.median(float(r["duration"]) for r in rows)

    write_summary(
        out_root, args.seed, allocation, realized_counts, group_concentration, duration_stats, pool_group_counts
    )

    print(f"{len(manifest_rows)} clips -> {out_root} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
