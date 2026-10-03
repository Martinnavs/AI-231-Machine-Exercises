"""End-to-end hybrid evaluation from audio: loads the CTC checkpoint (and, optionally, a different checkpoint for the heads),
runs `vcm.hybrid.hybrid_decode_all` on every clip (clean, or the fixed-seed perturbation of `vcm.noisy_eval`), and scores the
decision against the manifest label/slot truth with its own scoring code (not the cached-row helpers in `semantic_eval`).
Writes one JSONL per shard; `--summarize` merges them.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.hybrid import hybrid_decode_all, load_hybrid_part
from me2_voicegen.vcm.noisy_eval import apply_perturbation, build_perturbations, build_rir_pool_for_seed
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.semantic_eval import _slot_truth


def score(a) -> None:
    torch.set_num_threads(1)
    ctc_model = load_hybrid_part(a.ctc_checkpoint, "ctc")
    cls_model = ctc_model if a.cls_checkpoint is None else load_hybrid_part(a.cls_checkpoint, "cls")
    ds = VCMDataset(a.manifest, split=a.split, augmenter=None)
    ex = LogMelFeatureExtractor()
    pert = None
    if a.noisy_seed is not None:
        rir = build_rir_pool_for_seed(a.noisy_seed)
        pert, noise = build_perturbations(ds, rir, a.noisy_seed), ds._noise_pool()
    cond = "clean" if a.noisy_seed is None else "noisy"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.split}_{cond}.shard{a.shard}of{a.n_shards}.jsonl"
    with path.open("w") as f:
        for i in range(len(ds)):
            if i % a.n_shards != a.shard or (a.limit and i >= a.limit):
                continue
            row, wav = ds.rows[i], ds[i].waveform
            if pert is not None:
                r, n, snr = pert[i]
                wav = apply_perturbation(wav, rir[r], noise[n], snr)
            slot_value, _ = _slot_truth(row)
            t0 = time.perf_counter()
            dec = hybrid_decode_all(ctc_model, cls_model, ex, wav, OPTIONB_GRAMMAR, cls_threshold=a.cls_threshold)
            ms = 1000 * (time.perf_counter() - t0)
            rec = {"i": i, "bucket": row["bucket"], "label": row["label"], "source_dataset": row["source_dataset"],
                   "slot_value": slot_value, "ms": ms}
            for p, d in dec.items():
                rec[p] = {"intent": d.intent, "slots": d.slots, "source": d.source}
            f.write(json.dumps(rec) + "\n")
    print("wrote", path)


def correct(r: dict, p: str) -> bool:
    d = r[p]
    if d["intent"] != r["label"]:
        return False
    return not r["slot_value"] or r["slot_value"].lower() in [str(v).lower() for v in d["slots"].values()]


def summarize(a) -> None:
    rows = [json.loads(l) for fp in sorted(Path(a.dir).glob(f"{a.split}_{a.cond}.shard*.jsonl")) for l in fp.open()]
    T = [r for r in rows if r["bucket"] == "target_commands"]
    N = [r for r in rows if r["bucket"] in ("babble", "silence")]
    print(f"{len(rows)} rows, {len(T)} targets, {len(N)} babble/silence, mean decode {sum(r['ms'] for r in rows)/len(rows):.0f} ms")
    for p in ("fallback", "agree"):
        ok = sum(correct(r, p) for r in T)
        fa = sum(r[p]["intent"] is not None for r in N)
        src = sum(r[p]["source"] == "cls" for r in T)
        print(f"{p}: {100*ok/len(T):.2f}% ({ok}/{len(T)}), FA {fa}/{len(N)}, classifier answered {src} targets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--manifest", type=Path, required=True); s.add_argument("--ctc-checkpoint", type=Path, required=True)
    s.add_argument("--cls-checkpoint", type=Path, default=None); s.add_argument("--cls-threshold", type=float, required=True)
    s.add_argument("--split", required=True); s.add_argument("--noisy-seed", type=int, default=None)
    s.add_argument("--shard", type=int, default=0); s.add_argument("--n-shards", type=int, default=1)
    s.add_argument("--limit", type=int, default=None); s.add_argument("--out-dir", type=Path, required=True)
    m = sub.add_parser("summarize"); m.add_argument("--dir", type=Path, required=True); m.add_argument("--split", required=True)
    m.add_argument("--cond", default="clean")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else summarize(a)
