"""Beam 10 against beam 50 on the whole-clip test sets, and beam 50 against the cached original-search rows.

Usage: compare_beam_widths.py <dir with <set>/b50 and <set>/b10> <main checkout out/vcm>
See docs/BEAM-SEARCH.md section 6."""
import sys
from pathlib import Path
from me2_voicegen.vcm.semantic_eval import (ctc_accepted, ctc_correct, is_babble, is_noise_silence, is_target, read_rows)
S = Path(sys.argv[1])  # dir holding <set>/b50 and <set>/b10 from `semantic_eval score --beam 50|10`
O = Path(sys.argv[2])  # the main checkout's out/vcm (cached beam-50 rows)
A = O / "ai231fil50-compare/H-wide"
TW = -0.1
# name, cached dir, split, cond, row filter
SETS = [
 ("orig_clean", A/"orig_clean", "test", "clean", lambda r: r["source_dataset"] == "optionb"),
 ("orig_noisy", A/"orig_noisy_ds", "test", "noisy", lambda r: r["source_dataset"] == "optionb"),
 ("persona_clean", A/"persona_clean", "test", "clean", lambda r: True),
 ("old_clean", A/"old_clean", "test", "clean", lambda r: r["source_dataset"] != "fil50_persona"),
 ("holdout", A/"holdout", "holdout", "clean", lambda r: True),
 ("internal_clean", O/"internal-heldout-compare/H-wide", "test", "clean", lambda r: True),
 ("internal_noisy", O/"internal-heldout-compare/H-wide", "test", "noisy", lambda r: True),
 ("uv_raw_clean", O/"user-voice-compare/raw/H-wide", "test", "clean", lambda r: True),
 ("uv_raw_noisy", O/"user-voice-compare/raw/H-wide", "test", "noisy", lambda r: True),
 ("uv_conv_clean", O/"user-voice-compare/converted/H-wide", "test", "clean", lambda r: True),
]
def key(r):
    c = r["ctc"]; return (c["intent"], tuple(sorted(c["slots"].items())), c["confidence"], ctc_accepted(r, TW))
def stats(rows, f):
    t = [r for r in rows if is_target(r) and f(r)]
    fa = [r for r in rows if (is_babble(r) or is_noise_silence(r)) and f(r)]
    return len(t), sum(ctc_correct(r, TW, with_slot=True) for r in t), len(fa), sum(ctc_accepted(r, TW) for r in fa)
print(f"{'set':15} {'targets':>7} | {'acc50':>6} {'acc10':>6} {'cached50':>8} | {'FA50':>6} {'FA10':>6} | rows differing 50 vs 10 (decision / any) | new50 vs cached rows")
tot = [0]*4
for name, cdir, split, cond, f in SETS:
    r50 = read_rows(S/name/"b50", split, cond); r10 = read_rows(S/name/"b10", split, cond)
    assert [r["i"] for r in r50] == [r["i"] for r in r10]
    n, c50, nfa, fa50 = stats(r50, f); _, c10, _, fa10 = stats(r10, f)
    try:
        rc = read_rows(cdir, split, cond)
    except Exception:
        rc = []
    cm = ""
    cc = ""
    if rc:
        _, cc_, _, _ = stats(rc, f); cc = f"{100*cc_/n:.1f}%"
        cd = {r["i"]: r for r in rc}
        same = sum(key(r) == key(cd[r["i"]]) for r in r50 if r["i"] in cd)
        cm = f"{same}/{len(r50)} identical to cached"
    dec = sum((r["ctc"]["intent"], tuple(sorted(r["ctc"]["slots"].items())), ctc_accepted(r, TW)) != (q["ctc"]["intent"], tuple(sorted(q["ctc"]["slots"].items())), ctc_accepted(q, TW)) for r, q in zip(r50, r10))
    anyd = sum(key(r) != key(q) for r, q in zip(r50, r10))
    print(f"{name:15} {n:7d} | {100*c50/n:5.1f}% {100*c10/n:5.1f}% {cc:>8} | {fa50:2d}/{nfa:<3d} {fa10:2d}/{nfa:<3d} | {dec:4d} / {anyd:4d} of {len(r50)} | {cm}")
