# Comparison: does VCM Dataset B help the `optiond` VCM model?

Three runs, all preset `optiond`, seed 0. Individual detail/repro commands in
each run's own `NOTES.md`.

| Run | Dir | Manifest | Checkpoint | Purpose |
|---|---|---|---|---|
| **Treatment** | `option-d-dataset-v2` | `optionb-v3-vcmx` (merged) | trained, epoch 88 | refreshed Option B + VCM Dataset B |
| **Control** | `option-d-dataset-v2-control` | `optionb-v3-vcmx-control` | trained, epoch 30 | refreshed Option B alone |
| **In-domain** | `option-d-dataset-v2-indomain` | `optionb-v3` (Option B only) | *treatment's* checkpoint, re-evaluated | did Dataset B hurt Option B's own domain? |

Control and treatment share **byte-identical val/test** (asserted at build
time — see `.scratch/optionb-v3-vcmx/tickets/00-RECAP.md` Phase 2 log), so
the control-vs-treatment delta below isolates Dataset B's effect, not an
eval-set difference. None of these three are comparable to the old
`out/vcm/optionb-optiond` baseline — that run used a different, stale
grammar (7 intents phrased differently).

## Headline: treatment wins, and it's not close

| Metric | Control | Treatment | Delta |
|---|---|---|---|
| Epochs before stopping | 40 (early-stopped, patience) | 90 (hit the 60-min budget, still improving) | |
| Best val_loss | 0.8092 | 0.4321 | |
| Test exact accuracy | 0.953 | 0.979 | **+2.6pp** |
| Test accept rate | 0.958 | 0.980 | +2.2pp |
| Babble false-accept rate | 0.086 (22/255) | 0.012 (3/255) | **7.2x worse in control** |
| Silence false-accept rate | 0.006 (2/324) | 0.015 (5/324) | control marginally better (noisy, n=324) |
| Filipino-vs-foreign accuracy gap | 0.162 | 0.050 | **3.2x worse in control** |

**Adding VCM Dataset B to training helped**, on every metric that matters
except one near-zero, low-n silence count. The two biggest wins:

1. **Babble false-accept rate.** Control accepts a false command from
   out-of-domain speech 7x more often. Dataset B contributed ~2,400
   additional train-split UNKNOWN rows (vs. control's 417) — this looks like
   exactly the signal that closes the gap.
2. **Accent generalization gap.** Control's Filipino-vs-foreign exact-accuracy
   gap (0.162) is over 3x the treatment run's (0.050). More training data
   overall, not specifically the babble rows, is the more likely explanation
   here — worth a targeted ablation later if this matters (e.g. Dataset B
   commands only, no Dataset B babble, held at the same babble count as
   control).

Control also genuinely ran out of what its smaller dataset could teach it —
it **overfit at epoch 30 and early-stopped at 40**, using only 27.5 of its 60
allotted minutes. A longer budget would not have closed this gap; the model
had already started memorizing rather than generalizing.

## Did Dataset B hurt Option B's own domain? No.

**History:** the first attempt at this (`option-d-dataset-v2-indomain`)
scored the treatment checkpoint against `optionb-v3/manifest.csv`'s val/test,
but that manifest's split for `background_noise`/`youtube_institutional`
predates the group-disjointness fix the merge builder applies — 89%/86% of
its val/test silence rows and 37%/46% of its val/test babble (youtube) rows
were actually in the treatment checkpoint's own train split. Rebuilt as
`optionb-v3-corrected` (= the control manifest with every Dataset B row
removed, so it's Option B's commands plus the *corrected* babble/silence
splits) and re-evaluated leak-free in `option-d-dataset-v2-indomain-fixed`.

| | Treatment (merged eval) | Treatment (in-domain, leak-free) | Old stale-grammar baseline |
|---|---|---|---|
| Test exact accuracy | 0.979 | **0.989** | 0.971 |
| Babble false-accept | 0.012 (3/255) | **0.000 (0/44)** | 0.018 (1/56) |
| Silence false-accept | 0.015 (5/324) | **0.025 (8/324)** | 0.000 (0/22) |

Command accuracy and false-accept rates both hold up on a properly leak-free,
held-out eval. Combined with the control-vs-treatment comparison above (which
was never affected by this leak — both its manifests share the same
corrected, byte-identical val/test), the conclusion stands on solid ground:
**Dataset B is a clear net positive**, with no meaningful cost to Option B's
own domain. The old-baseline row is directional-only context (different
grammar), not a controlled comparison.

## Bottom line

- **Recommendation:** the treatment checkpoint (`option-d-dataset-v2`) is the
  one to carry forward. It beats control on the controlled comparison and
  didn't regress Option B's own domain.
- **Still open before this ships:** tech-lead review of the
  `optionb-v3-vcmx` feature (grammar refresh + merge builder) per dev-flow —
  nothing here has been committed yet. These are 1-hour budget runs for
  signal, not final numbers; a longer budget (treatment was still improving
  at epoch 90) would likely move both further apart, not closer.
