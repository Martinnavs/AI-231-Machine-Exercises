"""T3 (.scratch/accent-balance-fil50/tickets/00-RECAP.md): sharded persona
(zero-shot) synthesis of T2's job list.

One process per GPU (`--shard i/N`, `index % N == i`). Builds the synthesizer
once per process -- same pattern as `generate_personas.py`'s `main()`. Each
job's `VoicePrompt` comes from T1's `<voice_id>.wav`/`.txt` pair in
`--refs-dir`.

Resumable: a job already `status=ok` in that shard's own prior manifest,
whose wav still exists on disk, is skipped, not re-synthesized. A failing job
is logged and marked `status=error` -- one bad clip must never abort the
shard.

Wakeword jobs additionally get `speech_start_s`/`speech_end_s` computed via
`wakeword.derive_speech_spans.detect_speech_span` -- `wakeword/dataset.py`'s
own windowing already crops+fits variable-length clips at train/eval time
from these columns (or falls back to live VAD/whole-clip when they're
empty), so no fixed-length pad/crop is needed here.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from me2_voicegen.generation.cli_common import DEFAULT_BACKEND, build_config
from me2_voicegen.synthesis.base import SynthesisResult, VoicePrompt, save_wav
from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError, probe_wav, resample_to_16k_in_place

logger = logging.getLogger(__name__)

GEN_FIELDS = [
    "job_id", "model", "split", "voice_id", "text", "label", "source_row_ref", "noisy_target",
    "path", "duration", "seconds_elapsed", "status", "speech_start_s", "speech_end_s",
]


def load_jobs(jobs_csv: Path) -> list[dict]:
    with jobs_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        i_str, n_str = spec.split("/", 1)
        i, n = int(i_str), int(n_str)
    except ValueError as exc:
        raise ValueError(f"--shard must be 'i/N' (e.g. '0/4'), got {spec!r}") from exc
    if n <= 0 or not (0 <= i < n):
        raise ValueError(f"--shard {spec!r}: need 0 <= i < N and N > 0")
    return i, n


def load_prior_shard_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["job_id"]: row for row in csv.DictReader(f)}


def load_voice_prompt(refs_dir: Path, voice_id: str) -> VoicePrompt:
    wav_path = refs_dir / f"{voice_id}.wav"
    txt_path = refs_dir / f"{voice_id}.txt"
    if not wav_path.is_file() or not txt_path.is_file():
        raise ManifestValidationError(f"voice {voice_id!r}: missing {wav_path.name}/{txt_path.name} under {refs_dir}")
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ManifestValidationError(f"voice {voice_id!r}: empty prompt text at {txt_path}")
    return VoicePrompt(wav_path=wav_path, text=text)


def compute_wakeword_span(wav_path: Path) -> tuple[str, str]:
    import torchaudio

    from me2_voicegen.wakeword.derive_speech_spans import detect_speech_span

    waveform, sr = torchaudio.load(str(wav_path))
    span = detect_speech_span(waveform, sr)
    if span is None:
        return "", ""
    return f"{span[0]:.3f}", f"{span[1]:.3f}"


def synthesize_job(
    job: dict, synthesizer, refs_dir: Path, out_dir: Path
) -> dict:
    out_wav = out_dir / "audio" / job["model"] / f"{job['job_id']}.wav"
    prompt = load_voice_prompt(refs_dir, job["voice_id"])

    start = time.perf_counter()
    result: SynthesisResult = synthesizer.synthesize(job["text"], prompt=prompt)
    save_wav(result, out_wav)
    if result.sample_rate != 16000:
        resample_to_16k_in_place(out_wav)
    elapsed = time.perf_counter() - start

    probe = probe_wav(out_wav)
    row = dict(job)
    row["path"] = str(out_wav)
    row["duration"] = f"{probe.duration:.3f}"
    row["seconds_elapsed"] = f"{elapsed:.3f}"
    row["status"] = "ok"
    row["speech_start_s"] = ""
    row["speech_end_s"] = ""
    if job["model"] == "wakeword":
        row["speech_start_s"], row["speech_end_s"] = compute_wakeword_span(out_wav)
    return row


def run_shard(jobs_csv: Path, out_dir: Path, shard_spec: str, refs_dir: Path, device: str, backend: str) -> Path:
    i, n = parse_shard(shard_spec)
    jobs = [j for idx, j in enumerate(load_jobs(jobs_csv)) if idx % n == i]

    shard_manifest_path = out_dir / f"gen_manifest.shard{i}.csv"
    prior = load_prior_shard_manifest(shard_manifest_path)

    backend_cls = get_backend_class(backend)
    config = build_config(backend_cls, backend, device, [])
    synthesizer = create_synthesizer(backend, **config)

    results: list[dict] = []
    n_ok = n_skipped = n_error = 0
    for job in jobs:
        prior_row = prior.get(job["job_id"])
        if prior_row and prior_row.get("status") == "ok" and prior_row.get("path") and Path(prior_row["path"]).is_file():
            results.append(prior_row)
            n_skipped += 1
            continue
        try:
            row = synthesize_job(job, synthesizer, refs_dir, out_dir)
            results.append(row)
            n_ok += 1
        except Exception as exc:
            logger.error("job %s failed: %s", job["job_id"], exc)
            row = dict(job)
            for field in ("path", "duration", "seconds_elapsed", "speech_start_s", "speech_end_s"):
                row[field] = ""
            row["status"] = "error"
            results.append(row)
            n_error += 1

    shard_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GEN_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"shard {i}/{n}: {n_ok} synthesized, {n_skipped} skipped (already ok), {n_error} failed -> {shard_manifest_path}")
    return shard_manifest_path


def merge_shards(out_dir: Path) -> Path:
    shard_paths = sorted(out_dir.glob("gen_manifest.shard*.csv"))
    if not shard_paths:
        raise ManifestValidationError(f"no gen_manifest.shard*.csv found under {out_dir}")

    seen: dict[str, dict] = {}
    for shard_path in shard_paths:
        with shard_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["job_id"] in seen:
                    raise ManifestValidationError(f"duplicate job_id {row['job_id']!r} across shards (sharding must be disjoint)")
                seen[row["job_id"]] = row

    dest = out_dir / "gen_manifest.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GEN_FIELDS)
        writer.writeheader()
        writer.writerows(seen.values())

    n_ok = sum(1 for r in seen.values() if r["status"] == "ok")
    n_error = sum(1 for r in seen.values() if r["status"] == "error")
    print(f"merged {len(shard_paths)} shards -> {dest}: {len(seen)} jobs ({n_ok} ok, {n_error} error)")
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=Path, help="T2's jobs.csv (required unless --merge-shards)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", default=None, help="'i/N', e.g. '0/4' (required unless --merge-shards)")
    parser.add_argument("--refs-dir", type=Path, default=None, help="T1's refs dir, voice_id.wav/.txt pairs (required unless --merge-shards)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        if args.merge_shards:
            merge_shards(args.out_dir)
        else:
            if not args.jobs or not args.shard or not args.refs_dir:
                raise ManifestValidationError("--jobs, --shard, and --refs-dir are required unless --merge-shards")
            run_shard(args.jobs, args.out_dir, args.shard, args.refs_dir, args.device, args.backend)
    except ManifestValidationError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
