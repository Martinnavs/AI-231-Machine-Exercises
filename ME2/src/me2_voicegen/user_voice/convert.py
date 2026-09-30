"""Stage convert: sharded CosyVoice2 voice conversion of the user's clips into persona voices.

One process per GPU (`--shard i/N`, `index % N == i`), resumable (a job already `status=ok`
in the shard's own manifest whose wav still exists is skipped), one failing job never
aborts the shard. Content and prosody come from the user's clip (`convert_voice`, no
transcript needed); timbre comes from the persona's `<voice_id>.wav` prompt in --refs-dir.
Outputs 16 kHz mono PCM16 wavs under <out-dir>/audio/vcm/<job_id>.wav and a
GEN_FIELDS-compatible `gen_manifest.shard{i}.csv` (+ `source_wav`) so the existing
`accent_balance.qa` shim/parse tooling applies unchanged.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from me2_voicegen.accent_balance.generate import parse_shard
from me2_voicegen.user_voice.plan import JOB_FIELDS
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError, probe_wav, resample_to_16k_in_place

logger = logging.getLogger(__name__)


def _load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def convert_job(job: dict, synthesizer, refs_dir: Path, out_dir: Path, trim_cache: Path) -> dict:
    from me2_voicegen.generation.generate_conversions import _resolve_prompt_wav
    from me2_voicegen.synthesis.base import save_wav

    ref = refs_dir / f"{job['voice_id']}.wav"
    if not ref.is_file():
        raise ManifestValidationError(f"persona prompt missing: {ref}")
    src = Path(job["source_wav"])
    if not src.is_file():
        raise ManifestValidationError(f"source clip missing: {src}")
    out_wav = out_dir / "audio" / "vcm" / f"{job['job_id']}.wav"
    t0 = time.perf_counter()
    result = synthesizer.convert_voice(str(src), str(_resolve_prompt_wav(ref, trim_cache)))
    save_wav(result, out_wav)
    if result.sample_rate != 16000:
        resample_to_16k_in_place(out_wav)
    row = dict(job)
    probe = probe_wav(out_wav)
    row.update(path=str(out_wav), duration=f"{probe.duration:.3f}",
               seconds_elapsed=f"{time.perf_counter() - t0:.3f}", status="ok")
    return row


def run_shard(jobs_csv: Path, out_dir: Path, shard: str, refs_dir: Path, device: str, backend: str) -> Path:
    from me2_voicegen.generation.cli_common import build_config
    from me2_voicegen.synthesis.factory import create_synthesizer, get_backend_class

    i, n = parse_shard(shard)
    jobs = [j for idx, j in enumerate(_load(jobs_csv)) if idx % n == i]
    shard_path = out_dir / f"gen_manifest.shard{i}.csv"
    prior = {r["job_id"]: r for r in _load(shard_path)} if shard_path.is_file() else {}
    cfg = build_config(get_backend_class(backend), backend, device, [])
    synth = create_synthesizer(backend, **cfg)
    if not hasattr(synth, "convert_voice"):
        raise ManifestValidationError(f"backend {backend!r} has no convert_voice")
    out_dir.mkdir(parents=True, exist_ok=True)
    results, ok, skipped, err = [], 0, 0, 0
    for job in jobs:
        p = prior.get(job["job_id"])
        if p and p.get("status") == "ok" and p.get("path") and Path(p["path"]).is_file():
            results.append(p); skipped += 1
            continue
        try:
            results.append(convert_job(job, synth, refs_dir, out_dir, out_dir / "_trimmed_refs")); ok += 1
        except Exception as exc:  # one bad clip must never abort the shard
            logger.error("job %s failed: %s", job["job_id"], exc)
            row = dict(job)
            for f in ("path", "duration", "seconds_elapsed"):
                row[f] = ""
            row["status"] = "error"
            results.append(row); err += 1
    with shard_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        w.writeheader(); w.writerows(results)
    print(f"shard {i}/{n}: {ok} converted, {skipped} skipped, {err} failed -> {shard_path}")
    return shard_path


def merge_shards(out_dir: Path) -> Path:
    paths = sorted(out_dir.glob("gen_manifest.shard*.csv"))
    if not paths:
        raise ManifestValidationError(f"no gen_manifest.shard*.csv under {out_dir}")
    seen: dict[str, dict] = {}
    for p in paths:
        for r in _load(p):
            if r["job_id"] in seen:
                raise ManifestValidationError(f"duplicate job_id {r['job_id']!r} across shards")
            seen[r["job_id"]] = r
    dest = out_dir / "gen_manifest.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        w.writeheader(); w.writerows(seen.values())
    print(f"merged {len(paths)} shards -> {dest}: {len(seen)} jobs, {sum(r['status']=='ok' for r in seen.values())} ok")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard")
    ap.add_argument("--refs-dir", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="cosyvoice2")
    ap.add_argument("--merge-shards", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if a.merge_shards:
            merge_shards(a.out_dir)
        else:
            if not (a.jobs and a.shard and a.refs_dir):
                raise ManifestValidationError("--jobs, --shard and --refs-dir are required unless --merge-shards")
            run_shard(a.jobs, a.out_dir, a.shard, a.refs_dir, a.device, a.backend)
    except ManifestValidationError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
