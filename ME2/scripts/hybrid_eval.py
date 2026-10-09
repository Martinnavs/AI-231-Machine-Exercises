"""Offline hybrid decode: CTC answer from one model, classifier fallback from another (or the same), scored from the
per-clip JSONL rows that `vcm.semantic_eval score` already wrote. Nothing is re-run and nothing is tuned: the CTC
threshold is -0.1 (margin 4.0, already applied in the rows) and each classifier threshold is the val-fit one from the report.

Policy A ("fallback"): accept if CTC accepts or the classifier accepts; answer = CTC if it accepted, else the classifier.
Policy B ("agree"): classifier gates; answer = CTC when it accepted and names the same intent, else the classifier.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from me2_voicegen.vcm.semantic_eval import (cls_accepted, cls_correct, ctc_accepted, ctc_correct, is_babble, is_noise_silence,
                                            is_target, read_rows, top_intent)

O = Path("/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/vcm")
TW = -0.1
THR = {"H-wide": 0.9562, "H-xl": 0.8787, "H-int": 0.9866}
A, I, U = O / "ai231fil50-compare", O / "internal-heldout-compare", O / "user-voice-compare"
# name, dir(model) template, split, cond, row filter
SETS = [
    ("ai231 test, original rows, clean", A, "{m}/orig_clean", "test", "clean", lambda r: r["source_dataset"] == "optionb"),
    ("ai231 test, original rows, perturbed", A, "{m}/orig_noisy_ds", "test", "noisy", lambda r: r["source_dataset"] == "optionb"),
    ("ai231 persona rows (voice-disjoint), clean", A, "{m}/persona_clean", "test", "clean", lambda r: True),
    ("old internal test, non-persona, clean", A, "{m}/old_clean", "test", "clean", lambda r: r["source_dataset"] != "fil50_persona"),
    ("old internal test, 129 real recordings", A, "{m}/old_clean", "test", "clean", lambda r: r["source_dataset"] == "vcm_balanced" and is_target(r)),
    ("holdout (186 real + 16 OOS)", A, "{m}/holdout", "holdout", "clean", lambda r: True),
    ("leak-free internal held-out, clean", I, "{m}", "test", "clean", lambda r: True),
    ("leak-free internal held-out, perturbed", I, "{m}", "test", "noisy", lambda r: True),
    ("user voice raw (20), clean", U, "raw/{m}", "test", "clean", lambda r: True),
    ("user voice raw (20), perturbed", U, "raw/{m}", "test", "noisy", lambda r: True),
    ("user voice converted (648), clean", U, "converted/{m}", "test", "clean", lambda r: True),
]


def load(root: Path, tmpl: str, m: str, split: str, cond: str) -> dict | None:
    d = root / tmpl.format(m=m)
    try:
        rows = read_rows(d, split, cond)
    except Exception:
        return None
    return {r["i"]: r for r in rows} if rows else None


def cls_ans(r, thr): return cls_accepted(r, thr), cls_correct(r, thr, with_slot=True)
def ctc_ans(r): return ctc_accepted(r, TW), ctc_correct(r, TW, with_slot=True)


def hybrid(rc, rk, thr, policy):
    ca, cc = ctc_ans(rc)
    ka, kc = cls_ans(rk, thr)
    if policy == "A":
        return (ca or ka, cc if ca else kc)
    agree = ca and rc["ctc"]["intent"] == top_intent(rk)[0]
    return (ka, cc if agree else kc)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    cols = ["old-production CTC", "H-int CTC", "H-int cls", "H-wide CTC", "H-wide cls", "H-xl CTC", "H-xl cls",
            "wide+wide A", "xl+xl A", "wide CTC + xl cls A", "wide CTC + xl cls B"]
    acc_t, fa_t = [], []
    for name, root, tmpl, split, cond, flt in SETS:
        D = {m: load(root, tmpl, m, split, cond) for m in ("old-production", "H-int", "H-wide", "H-xl")}
        base = D["H-wide"]
        ids = [i for i in base if flt(base[i])]
        T = [i for i in ids if is_target(base[i])]
        N = [i for i in ids if is_babble(base[i]) or is_noise_silence(base[i])]

        def run(col):
            def f(i):
                try:
                    if col == "old-production CTC": return ctc_ans(D["old-production"][i])
                    if col == "H-int CTC": return ctc_ans(D["H-int"][i])
                    if col == "H-int cls": return cls_ans(D["H-int"][i], THR["H-int"])
                    if col == "H-wide CTC": return ctc_ans(D["H-wide"][i])
                    if col == "H-wide cls": return cls_ans(D["H-wide"][i], THR["H-wide"])
                    if col == "H-xl CTC": return ctc_ans(D["H-xl"][i])
                    if col == "H-xl cls": return cls_ans(D["H-xl"][i], THR["H-xl"])
                    if col == "wide+wide A": return hybrid(D["H-wide"][i], D["H-wide"][i], THR["H-wide"], "A")
                    if col == "xl+xl A": return hybrid(D["H-xl"][i], D["H-xl"][i], THR["H-xl"], "A")
                    if col == "wide CTC + xl cls A": return hybrid(D["H-wide"][i], D["H-xl"][i], THR["H-xl"], "A")
                    if col == "wide CTC + xl cls B": return hybrid(D["H-wide"][i], D["H-xl"][i], THR["H-xl"], "B")
                except (KeyError, TypeError):
                    return None
            return f
        arow, frow = [], []
        for c in cols:
            f = run(c)
            res = [f(i) for i in T]
            if not T or any(x is None for x in res):
                arow.append("n/a")
            else:
                ok = sum(x[1] for x in res); arow.append(f"{100*ok/len(T):.1f}")
            resn = [f(i) for i in N]
            frow.append("n/a" if (not N or any(x is None for x in resn)) else f"{sum(x[0] for x in resn)}/{len(N)}")
        acc_t.append(f"| {name} ({len(T)}) | " + " | ".join(arow) + " |")
        fa_t.append(f"| {name} | " + " | ".join(frow) + " |")
    hdr = "| set (targets) | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    hdr2 = "| set | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    txt = ("## Accuracy, intent+slot (%)\n\n" + hdr + "\n" + "\n".join(acc_t) +
           "\n\n## False accepts on babble + noise/silence rows\n\n" + hdr2 + "\n" + "\n".join(fa_t) + "\n")
    a.out.write_text(txt); print(txt)


if __name__ == "__main__":
    main()
