#!/usr/bin/env python
"""Ghost-clip probe for the dense-scoring pilot (fresh data; experiment-only).

141 ACOUSTIC_GHOST clips (podcast audio that sounds command-like but is NOT a
command; `.scratch/ghosts-filtered/`) were never in the production
checkpoint's training manifest, and never used to calibrate any threshold
here, so they are independent of val/test. Ground truth: every ghost must be
REJECTED, so any accept is a false accept (FA).

Configs (thresholds fit on FULL clean val exactly as in the main pilot):
  B    baseline -0.1 + margin 4.0 (production)
  Bn   baseline -0.1, margin off (reference only)
  C    d2-global, margin off      (iso-accept tau)
  H    d2 relax-only per-intent, margin off
Pre-registered rule (frozen before the run): candidate S in {C, H} PASSES iff
FA_S <= FA_B on the 141 clips. Reported, not gated: per-intent / per-podcast
FA, accepted intent == ghost's suspected intent, Wilson 95% upper bound.
Power note: if FA_B == 0 then S must also be 0; 0/141 bounds the true rate at
~2.6%, so this probe can detect regressions, not prove near-zero FAR.
CPU only. Run: .venv/bin/python scripts/dense_pilot_ghosts.py --dump-dir <dense_pilot dir>
"""

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/checkpoints/checkpoint.pt"
MANIFEST = ROOT / "out/conversions/v2/optionb-v3-vcmx/manifest-ghosts-only.csv"


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def _ghost_worker(args):
    """Same waveform loading as `VCMDataset._load_waveform` (the main tree's
    `vcm/text.py` predates the `acoustic_ghost` source, so VCMDataset can't
    open this manifest here)."""
    import torch
    import torchaudio

    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform

    checkpoint, manifest, idx_rows = args
    torch.set_num_threads(1)
    model, _ = load_checkpoint(checkpoint, device="cpu")
    extractor = LogMelFeatureExtractor()
    out = []
    for i, row in idx_rows:
        wav, sr = torchaudio.load(str(Path(manifest).parent / row["path"]))
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        if sr != int(row["sample_rate"]):
            wav = torchaudio.functional.resample(wav, sr, int(row["sample_rate"]))
        out.append((i, logp_for_waveform(model, extractor, wav, device="cpu")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    from me2_voicegen.vcm import dense_pilot as dp

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(str(MANIFEST) + ".ghost_intents.csv", newline="", encoding="utf-8") as f:
        suspected = {r["filename"]: r["intent"] for r in csv.DictReader(f)}
    n = len(rows)
    indexed = list(enumerate(rows))
    jobs = [(str(CHECKPOINT), str(MANIFEST), indexed[k::args.workers]) for k in range(args.workers)]
    jobs = [j for j in jobs if j[2]]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        parts = list(pool.map(_ghost_worker, jobs))
    by_index = {i: lp for part in parts for i, lp in part}
    logps = [by_index[i] for i in range(n)]
    recs = dp.score_rows(
        logps,
        [{"index": i, "bucket": "acoustic_ghost", "label": suspected[r["filename"]]} for i, r in enumerate(rows)],
        workers=args.workers,
    )

    val_clean = dp._load_records(args.dump_dir / "records")[("val", "clean")]
    base = {"default": dp.BASELINE_THRESHOLD}
    configs = {
        "B": ("baseline", base, dp.PRODUCTION_MARGIN),
        "Bn": ("baseline", base, None),
        "C": ("d2", dp.global_tau(val_clean), None),
        "H": ("d2", dp.relax_only_tau(val_clean), None),
    }
    out = {"n": n, "configs": {}}
    for name, (scorer, tau, margin) in configs.items():
        acc = [(i, r) for i, r in enumerate(recs) if dp.accepts(r, scorer, tau, margin)]
        out["configs"][name] = {
            "fa": len(acc),
            "wilson95_upper": wilson_upper(len(acc), n),
            "accepted_intent": dict(Counter(r["intent"] for _, r in acc)),
            "by_podcast": dict(Counter(rows[i]["group_id"] for i, _ in acc)),
            "intent_matches_suspected": sum(r["intent"] == r["label"] for _, r in acc),
        }
    fa_b = out["configs"]["B"]["fa"]
    out["rule"] = {s: out["configs"][s]["fa"] <= fa_b for s in ("C", "H")}
    out["verdict"] = {s: ("PASS" if ok else "FAIL") for s, ok in out["rule"].items()}
    (args.dump_dir / "metadata").mkdir(exist_ok=True)
    (args.dump_dir / "metadata" / "dense_ghost_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
