#!/usr/bin/env python
"""Compare the user-voice retrain against the production checkpoint (feature user-voice-weak-phrases).

Same val/test rows for both (asserted byte-identical across the two manifests), clean and the fixed-seed
noisy gate (noisy_eval seed 0), each model at ITS OWN clean-val-chosen threshold (from its eval_report.json,
the same convention as the noisy gate), margin off. Reports overall accuracy/FARs, the per-intent exact
accuracy for every intent (weak intents called out), and a probe on the user's 20 raw recordings.
Not a pre-registered gate: differences under ~0.3pp are within what a single seed can produce.
"""
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
WEAK = ("LIGHT_OFF", "COLOR", "TIME", "CALL", "MESSAGE", "LIST_REMINDERS")
CONDS = ("clean", "noisy_s0")


def rows_hash(manifest: Path, splits=("val", "test")) -> str:
    h = hashlib.sha256()
    for r in csv.DictReader(manifest.open(newline="", encoding="utf-8")):
        if r["split"] in splits:
            h.update("|".join(r[k] for k in sorted(r) if k != "path").encode() + b"\n")
    return h.hexdigest()


def choose_thr(recs_val_clean: list[dict]) -> float:
    """The exact rule `vcm.evaluate` uses (Youden-J over the val sweep, ties -> least permissive), applied to
    the model's own clean-val records so the two models are treated identically."""
    from me2_voicegen.vcm.evaluate import RowResult, choose_operating_threshold, sweep_thresholds
    rows = [RowResult(index=r["index"], bucket=r["bucket"], label=r["label"], text="", intent=r["intent"],
                      confidence=r["confidence"]) for r in recs_val_clean]
    return float(choose_operating_threshold(sweep_thresholds(rows))["threshold"])


def dump_and_score(run: Path, manifest: Path, tag: str, workers: int) -> dict:
    from me2_voicegen.vcm import dense_pilot as dp
    spec = importlib.util.spec_from_file_location("dpd", ROOT / "scripts/dense_pilot_dump.py")
    dpd = importlib.util.module_from_spec(spec); sys.modules["dpd"] = dpd; spec.loader.exec_module(dpd)
    out = run / f"dense_eval_{tag}"
    recs = {}
    for split in ("val", "test"):
        for cond in CONDS:
            if not (out / "logits" / f"{split}_{cond}.npz").exists():
                dpd.dump_split(run / "checkpoints/checkpoint.pt", manifest, split, cond, out, workers)
            logps, rows = dpd.load_dump(out, split, cond)
            recs[(split, cond)] = dp.score_rows(logps, rows, workers)
    return recs


def raw_probe(run: Path, thr: float, clips: list[dict]) -> list[dict]:
    import torch, torchaudio
    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.decoder import decode_utterance
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
    from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform
    model, _ = load_checkpoint(run / "checkpoints/checkpoint.pt", device="cpu")
    fe = LogMelFeatureExtractor()
    out = []
    for c in clips:
        wav, sr = torchaudio.load(c["path"])
        wav = wav.mean(0)
        lp = logp_for_waveform(model, fe, wav, device="cpu")
        d = decode_utterance(lp, OPTIONB_GRAMMAR, thr, beam_width=50)
        out.append({"clip_id": c["clip_id"], "label": c["label"], "intent": d.intent, "conf": d.confidence,
                    "text": d.text, "correct": d.intent == c["label"] and not d.no_match})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-run", type=Path, required=True); ap.add_argument("--old-run", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="eval manifest used for BOTH models")
    ap.add_argument("--tag", required=True, help="name for this eval manifest (dump dirs)")
    ap.add_argument("--old-records", type=Path, default=None, help="reuse existing score records of the OLD run on this same manifest")
    ap.add_argument("--clips", type=Path, required=True); ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    from me2_voicegen.vcm import dense_pilot as dp

    new = dump_and_score(a.new_run, a.manifest, a.tag, a.workers)
    old = dp._load_records(a.old_records) if a.old_records else dump_and_score(a.old_run, a.manifest, a.tag, a.workers)
    thr_new, thr_old = choose_thr(new[("val", "clean")]), choose_thr(old[("val", "clean")])
    res = {"eval_manifest": str(a.manifest), "val_test_rows_hash": rows_hash(a.manifest), "thr_new": thr_new, "thr_old": thr_old,
           "overall": {}, "per_intent": {}}
    for key in new:
        mo = dp.metrics(old[key], "baseline", {"default": thr_old}, None)
        mn = dp.metrics(new[key], "baseline", {"default": thr_new}, None)
        k = "/".join(key)
        res["overall"][k] = {"old": {f: mo[f] for f in ("n_targets", "accepted", "exact", "babble_fa", "silence_fa", "n_babble", "n_silence")},
                             "new": {f: mn[f] for f in ("n_targets", "accepted", "exact", "babble_fa", "silence_fa", "n_babble", "n_silence")}}
        res["per_intent"][k] = {lab: {"n": mo["per_intent_n"][lab], "old": mo["per_intent_exact"].get(lab, 0), "new": mn["per_intent_exact"].get(lab, 0)}
                                for lab in mo["per_intent_n"]}
    clips = list(csv.DictReader(a.clips.open(newline="", encoding="utf-8")))
    res["raw_probe"] = {"old": raw_probe(a.old_run, thr_old, clips), "new": raw_probe(a.new_run, thr_new, clips)}
    (a.new_run / "metadata").mkdir(exist_ok=True)
    (a.new_run / f"metadata/user_voice_eval_{a.tag}.json").write_text(json.dumps(res, indent=1))
    # markdown
    L = ["# user-voice retrain vs production (same val/test, each at its own chosen threshold, margin off)", "",
         f"eval manifest: `{a.manifest}`; clean-val-chosen thresholds: old {thr_old}, new {thr_new}", "",
         "| set | old exact | new exact | old babble/silence FA | new babble/silence FA |", "|---|---|---|---|---|"]
    for k, v in res["overall"].items():
        o, n = v["old"], v["new"]
        L.append(f"| {k} | {o['exact']}/{o['n_targets']} ({o['exact']/o['n_targets']:.4f}) | {n['exact']}/{n['n_targets']} ({n['exact']/n['n_targets']:.4f}) | "
                 f"{o['babble_fa']}/{o['n_babble']} , {o['silence_fa']}/{o['n_silence']} | {n['babble_fa']}/{n['n_babble']} , {n['silence_fa']}/{n['n_silence']} |")
    L += ["", "## Per-intent exact-correct, val+test x clean+noisy pooled (weak intents first)", "", "| intent | n | old | new | delta |", "|---|---|---|---|---|"]
    pooled = {}
    for k, d in res["per_intent"].items():
        for lab, v in d.items():
            p = pooled.setdefault(lab, [0, 0, 0]); p[0] += v["n"]; p[1] += v["old"]; p[2] += v["new"]
    for lab in list(WEAK) + sorted(set(pooled) - set(WEAK)):
        n, o, w = pooled[lab]
        L.append(f"| {'**'+lab+'**' if lab in WEAK else lab} | {n} | {o} ({100*o/n:.2f}%) | {w} ({100*w/n:.2f}%) | {w-o:+d} ({100*(w-o)/n:+.2f}pp) |")
    L += ["", "## Your 20 raw recordings (prosody was used in training via conversion; timbre unseen)", "",
          "| clip | label | old -> (conf) | new -> (conf) |", "|---|---|---|---|"]
    for o, n in zip(res["raw_probe"]["old"], res["raw_probe"]["new"]):
        L.append(f"| {o['clip_id']} | {o['label']} | {o['intent']} ({o['conf']:.3f}) {'OK' if o['correct'] else 'X'} | {n['intent']} ({n['conf']:.3f}) {'OK' if n['correct'] else 'X'} |")
    L.append(f"\nraw clips correct: old {sum(r['correct'] for r in res['raw_probe']['old'])}/20, new {sum(r['correct'] for r in res['raw_probe']['new'])}/20")
    (a.new_run / f"metadata/user_voice_eval_{a.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
