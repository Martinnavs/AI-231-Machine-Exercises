# Summary: training the 1.01M-param MatchboxNet CTC model on Option B vs. Option B + VCM Dataset B

## Bottom line

**Adding VCM Dataset B to training clearly helps and doesn't hurt. Ship the
treatment checkpoint** (`out/vcm/option-d-dataset-v2`). It beats Option-B-only
on every metric that matters, on an identical held-out eval, and loses
nothing on Option B's own original domain. One cheap, unused lever remains
before deployment: a margin threshold (~4.5) that would clear 81% of the
model's one remaining weak spot (TIME/STOP false accepts) for free.

## Setup

- **Model:** MatchboxNetCTC, preset `optiond` (~1.01M params, 5 TCS blocks).
- **Two datasets, same everything else** (seed 0, 60-min budget, same
  refreshed grammar): **control** = Option B alone; **treatment** = Option B
  + VCM Dataset B (in-grammar commands, UNKNOWN, ESC-50 silence rebuilt by
  fold). Control and treatment share byte-identical val/test.

## Result

| | Control (Option B only) | Treatment (+ Dataset B) |
|---|---|---|
| Epochs used | 40, early-stopped at 27 min (overfit) | 90, used the full hour |
| Test exact accuracy | 0.953 | **0.979** |
| Babble false-accept rate | 0.086 | **0.012** (7x lower) |
| Filipino-accent accuracy gap | 0.162 | **0.050** (3x smaller) |

Control ran out of what its smaller dataset could teach it — it wasn't cut
off by time, it stopped improving on its own. A per-clip diff confirms this
isn't just a threshold shift: treatment correctly fixes 54 specific clips
control got wrong, against only 13 new mistakes.

**Did it hurt Option B's own domain?** No — re-tested leak-free (see
caveats), the treatment checkpoint scores 0.989 exact accuracy and low
false-accept rates on Option B alone, matching or beating its performance
before Dataset B was added.

## What's still wrong, and the free fix available

The model's remaining false accepts aren't random — they concentrate almost
entirely on two short, generic words: **TIME** and **STOP** (16 of 21
remaining false accepts). A scoped check found `required_command_margin ≈
4.5` (an existing, already-built gate that's currently off) would catch 13
of those 16 at **zero cost** to real commands. Not applied yet — this is a
finding, not a shipped change.

## Caveats, stated plainly

- One early eval attempt (comparing the trained checkpoint against a stale
  reference manifest) leaked training data into its "held-out" set for
  babble/silence; caught, and redone leak-free (numbers above are the
  corrected ones).
- Tech-lead review of the code that built this data (approved, no
  blockers) also caught and fixed a real pre-existing split leak unrelated
  to this specific comparison (old reject-probe rows whose speaker/video
  spanned splits).
- These are 1-hour budget runs for signal, not final numbers — treatment
  was still improving at epoch 90 when the clock ran out.
- The margin-gate number only checked two intents (TIME/STOP), not the
  whole grammar — a full calibration run should confirm it before it ships.

## Full detail

- `option-d-dataset-v2-COMPARISON.md` — control vs. treatment, in-domain re-eval
- `option-d-dataset-v2-FALSE-ACCEPT-ANALYSIS.md` — per-clip diff, margin-gate check
- `.scratch/optionb-v3-vcmx/tickets/00-RECAP.md` — implementation ticket + review
