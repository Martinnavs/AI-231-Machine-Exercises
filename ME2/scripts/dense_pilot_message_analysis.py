"""Why do MESSAGE rows score low under D2 per_char? (dense-d2-loose-impl follow-up analysis)

Read-only diagnostic over the pilot's score records/logits (run from ME2/, needs the local
`out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot/{records,logits}`). Not a gate; no threshold is
selected from it. Sections: phrase inventory, per-intent d2 distribution, MESSAGE rows by
split/condition with the rows B accepts but C2 (-1.204) rejects, greedy (unconstrained) text for
the worst rows, speaker groups, and a phrase-length bias table.
"""
import csv, json, collections, statistics as st
import numpy as np
from pathlib import Path
from me2_voicegen.vcm import dense_pilot as dp
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR as G
D = Path("out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot")
recs = dp._load_records(D / "records")
phr = collections.defaultdict(list)
for t, i, sl in G.all_phrases(): phr[i].append(t)
print("MESSAGE phrases:", len(phr["MESSAGE"]), phr["MESSAGE"][:6])
print("phrase len (chars) median by intent:")
for i, ts in sorted(phr.items(), key=lambda x: st.median(len(t) for t in x[1])):
    print(f"  {i:16s} n={len(ts):3d} median_len={st.median(len(t) for t in ts):5.1f} min={min(len(t) for t in ts)} max={max(len(t) for t in ts)}")

rows = {c: {int(r["index"]): r for r in csv.DictReader(open(D / f"logits/val_{c}.rows.csv", newline=""))} for c in dp.CONDITIONS}
base = {"default": -0.1}; c2 = {"default": -1.204}
def tgt(split, cond, label="MESSAGE"):
    return [r for r in recs[(split, cond)] if r["bucket"] == dp.TARGET and r["label"] == label]

print("\n== per-intent d2 distribution (val clean targets, correct winner only) ==")
allv = {}
for lab in sorted(phr):
    v = [r["d2"] for r in tgt("val", "clean", lab) if r["intent"] == lab and r["d2"] is not None]
    if v: allv[lab] = v
for lab, v in sorted(allv.items(), key=lambda x: np.median(x[1])):
    print(f"  {lab:16s} n={len(v):3d} p05={np.percentile(v,5):6.2f} p10={np.percentile(v,10):6.2f} med={np.median(v):6.2f}  frac<-1.204={np.mean(np.array(v)<-1.204):.3f}")

for split in ("val", "test"):
  for cond in dp.CONDITIONS:
    ms = tgt(split, cond)
    print(f"\n== MESSAGE {split}/{cond}: n={len(ms)} ==")
    pred = collections.Counter(r["intent"] for r in ms); print(" predicted:", dict(pred))
    ph = collections.Counter((r["grammar_text"] if r["intent"] == "MESSAGE" else "-"+str(r["intent"])) for r in ms); print(" winning phrase:", dict(ph))
    for phrase in sorted({r["grammar_text"] for r in ms if r["intent"] == "MESSAGE"}):
        v = [r["d2"] for r in ms if r["intent"] == "MESSAGE" and r["grammar_text"] == phrase]
        print(f"   {phrase!r:20s} n={len(v):3d} d2 med={np.median(v):6.2f} p10={np.percentile(v,10):6.2f} min={min(v):6.2f}  reject@-1.204={sum(x<-1.204 for x in v)}")
    lost = [r for r in ms if dp.accepts(r, "baseline", base, 4.0) and r["intent"] == "MESSAGE" and not dp.accepts(r, "d2", c2, None)]
    gain = [r for r in ms if (not dp.accepts(r, "baseline", base, 4.0)) and dp.accepts(r, "d2", c2, None) and r["intent"] == "MESSAGE"]
    print(f" B-accepted-correct but C2 rejects: {len(lost)}; C2 gains: {len(gain)}")
    for r in lost:
        rw = rows[cond].get(r["index"]) if split == "val" else None
        print("   LOST idx", r["index"], "phrase", repr(r["grammar_text"]), "d2=%.3f conf=%.3f raw/T ratio T=%d" % (r["d2"], r["confidence"], r["T"]), "gap", r["incomplete_gap"], (rw or {}).get("group_id"))

print("\n\n==== deep dive ====")
from me2_voicegen.vcm.decoder import _greedy_unconstrained
def lp_of(cond, idx):
    z = np.load(D / f"logits/val_{cond}.npz")
    keys = list(rows[cond].keys())
    k = keys.index(idx)
    return z["logp"][z["offsets"][k]:z["offsets"][k+1]]
print("greedy (unconstrained) text vs. grammar winner for the lost + a few good rows, val/clean:")
ms = tgt("val", "clean")
ms_sorted = sorted([r for r in ms if r["intent"]=="MESSAGE"], key=lambda r: r["d2"])
for r in ms_sorted[:8] + ms_sorted[len(ms_sorted)//2:len(ms_sorted)//2+2]:
    lp = lp_of("clean", r["index"])
    gt, gs = _greedy_unconstrained(lp)
    g = rows["clean"][r["index"]]["group_id"]
    print(f" idx {r['index']:5d} grp={g:14s} winner={r['grammar_text']!r:18s} d2={r['d2']:7.3f} raw={r['command_raw_score']:8.2f} T={r['T']} greedy={gt!r} greedy_frame_score={gs:.3f}")

print("\nspeaker groups among MESSAGE val/clean rows (d2 stats):")
by = collections.defaultdict(list)
for r in ms:
    if r["intent"]=="MESSAGE": by[rows["clean"][r["index"]]["group_id"]].append(r["d2"])
for g, v in sorted(by.items(), key=lambda x: np.min(x[1]))[:8]:
    print(f"  {g:14s} n={len(v):2d} min={min(v):7.3f} med={np.median(v):7.3f}")
print(" n groups:", len(by), " ref_/fil groups:", sum(g.startswith("ref_") for g in by))
print("\nbare 'message' rows: d2 by reference-voice vs other speakers (val+test clean, all splits pooled):")
for split in ("val","test"):
    rws = {int(r["index"]): r for r in csv.DictReader(open(D / f"logits/{split}_clean.rows.csv", newline=""))}
    bare = [(rws[r["index"]]["group_id"], r["d2"]) for r in recs[(split,"clean")] if r["bucket"]==dp.TARGET and r["label"]=="MESSAGE" and r["intent"]=="MESSAGE" and r["grammar_text"]=="message"]
    ref = [d for g,d in bare if g.startswith("ref_")]; oth = [d for g,d in bare if not g.startswith("ref_")]
    print(f"  {split}: bare 'message' n={len(bare)}; ref_ voices n={len(ref)} med={np.median(ref) if ref else float('nan'):.3f} min={min(ref) if ref else float('nan'):.2f}; other n={len(oth)} med={np.median(oth):.3f} min={min(oth):.2f}")

print("\n\n==== length bias ====")
def bucket(n): return "<=6" if n<=6 else "7-10" if n<=10 else "11-15" if n<=15 else "16+"
tot = collections.defaultdict(lambda: [0,0,0,0])
for cond in dp.CONDITIONS:
    for split in ("val","test"):
        for r in recs[(split,cond)]:
            if r["bucket"]==dp.TARGET and r["intent"]==r["label"] and r["d2"] is not None:
                b = bucket(len(r["grammar_text"])); t = tot[b]
                t[0]+=1; t[1]+= r["d2"] < -1.204; t[2]+= r["command_raw_score"] < -8.0; t[3]+= (r["confidence"] >= -0.1)  # B(no margin) accepts
print("correct-winner target rows, val+test, clean+noisy pooled:")
print(f"  {'len(phrase)':11s} {'n':>6s} {'rej@D2 -1.204':>14s} {'raw<-8 (abs deficit)':>21s}")
for b in ("<=6","7-10","11-15","16+"):
    n,a,c,_ = tot[b]; print(f"  {b:11s} {n:6d} {a/n:14.4f} {c/n:21.4f}")
print("\nrejected-by-D2 correct rows: absolute raw deficit vs phrase length")
xs=[(len(r["grammar_text"]), r["command_raw_score"]) for cond in dp.CONDITIONS for split in ("val","test") for r in recs[(split,cond)]
    if r["bucket"]==dp.TARGET and r["intent"]==r["label"] and r["d2"] is not None and r["d2"] < -1.204]
print(" n rejected:", len(xs), "median len", np.median([x[0] for x in xs]), "median raw", np.median([x[1] for x in xs]))
print("\nFA-winning phrase lengths (babble+silence rows accepted by B m=4 vs by nothing under D2 C2), val+test noisy+clean:")
fa_b=[len(r["grammar_text"]) for cond in dp.CONDITIONS for split in ("val","test") for r in recs[(split,cond)] if r["bucket"] in dp.REJECT and dp.accepts(r,"baseline",{"default":-0.1},4.0)]
fa_nomargin=[len(r["grammar_text"]) for cond in dp.CONDITIONS for split in ("val","test") for r in recs[(split,cond)] if r["bucket"] in dp.REJECT and dp.accepts(r,"baseline",{"default":-0.1},None)]
print(" B margin off: n=%d median len=%.1f  frac<=6: %.2f" % (len(fa_nomargin), np.median(fa_nomargin), np.mean(np.array(fa_nomargin)<=6)))
print(" B margin 4  : n=%d median len=%.1f  frac<=6: %.2f" % (len(fa_b), np.median(fa_b), np.mean(np.array(fa_b)<=6)))
# how much of D2's FA removal comes from short-phrase FAs
rem=[len(r["grammar_text"]) for cond in dp.CONDITIONS for split in ("val","test") for r in recs[(split,cond)] if r["bucket"] in dp.REJECT and dp.accepts(r,"baseline",{"default":-0.1},None) and not dp.accepts(r,"d2",{"default":-1.204},None)]
print(" FAs (B margin off) removed by C2: n=%d, len<=6: %.2f, median len %.1f" % (len(rem), np.mean(np.array(rem)<=6), np.median(rem)))
