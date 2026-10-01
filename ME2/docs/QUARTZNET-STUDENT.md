# QuartzNet-5x3-tiny CTC student vs. the production MatchboxNet (`optiond`)

Status: **experiment complete; promotion pending** (see [What is not done](#what-is-not-done)).
Dates: 2026-09-22 (research) to 2026-10-01 (results). Branch `optionb-grammar-v2`, commit `e513eef`.
Design record: `feature-engineering/quartznet-ctc/SPEC.md`. Research basis: `docs/research/ctc-edge-model-alternatives.md`.

## Result in one paragraph

A 911k-parameter QuartzNet-style character-CTC model with 2x temporal subsampling beat the production
`optiond` (1.01M parameters) on every gate we measured, with the same data, recipe and scoring, in both seeds
we trained. On the fixed-seed noisy/reverb gate it scores 97.8% exact-intent against 94.8% (two-seed means),
cuts noisy babble false accepts from 10 and 3 to 1 and 1, exports a smaller INT8 model, and runs end to end
about 1.8x faster because the grammar beam search sees half as many frames. Two ablations suggest the gain
comes mostly from the 20 ms output frame rate, not from a wider receptive field or from the layer layout
alone. The model is not yet in production: its operating threshold, incomplete-prefix margin and real-audio
soak behaviour have not been re-validated.

## What was compared

| | `optiond` (production) | QuartzNet stride 2 |
|---|---|---|
| Layout | prologue conv, 5 single TCS blocks (kernels 11-19), dense k=29 epilogue | prologue (stride 2), 5 blocks x 3 separable sub-modules (kernels 7-15), separable k=15 epilogue |
| Output frames for 2.5 s | 251 (one per 10 ms feature frame) | 126 (20 ms) |
| Parameters | 1,009,725 | 911,189 (width 192) |
| Frontend, alphabet, decoder | 40-mel, 29 symbols, grammar-constrained CTC prefix beam search (unchanged) | same |

Everything else is identical: training manifest `out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv`,
all `vcm.train` defaults, `--p-rir 0.7`, a 135-minute wall-clock budget, AMP, and the same evaluation: the
`fil50` manifest, `optionb` grammar, beam 50, clean-val-chosen threshold, noisy gate seed 0.
The baseline was not retrained: the recipe code is unchanged since the production commit `7c94abb`.

Most of `optiond`'s parameters are not where its receptive field is. Its dense epilogue convolution
(128 to 224 channels, kernel 29) is 831k parameters, 82% of the model, while the five depthwise kernels that
set the receptive field total 8.9k.

## Results

All numbers are on the test split (3,494 command clips, 255 babble, 324 silence) at each model's own
clean-val-chosen threshold. "Noisy" is one fixed RIR plus additive-noise perturbation per clip, identical
across models.

### Main comparison, two seeds

| Run | Epochs | Threshold | Clean exact | Noisy exact | Noisy babble FA | Noisy silence FA | Clean babble / silence FA |
|---|---|---|---|---|---|---|---|
| `optiond` seed 0 (production) | 67 (deadline) | -0.1 | 3448 (98.7%) | 3345 (95.7%) | 10 | 6 | 3 / 1 |
| `optiond` seed 1 | 58 (deadline) | -0.075 | 3430 (98.2%) | 3278 (93.8%) | 3 | 8 | 0 / 2 |
| QuartzNet seed 0 | 61 (early stop) | -0.1 | 3468 (99.3%) | 3416 (97.8%) | 1 | 2 | 0 / 0 |
| QuartzNet seed 1 | 58 (deadline) | -0.1 | 3457 (98.9%) | 3419 (97.9%) | 1 | 6 | 0 / 5 |

Two-seed means: clean exact 99.1% (QuartzNet) vs 98.4% (`optiond`); noisy exact 97.8% vs 94.8%.

- QuartzNet's worst run beats `optiond`'s best run on both clean (3457 vs 3448) and noisy (3416 vs 3345) exact.
- QuartzNet's two noisy scores differ by 3 clips. `optiond`'s differ by 67, so part of the seed-0 gap was
  `optiond` drawing a good seed. Even so, `optiond`'s better seed is about 70 clips behind QuartzNet.
- The clean-exact advantage (+23 clips on average, 0.7 pp) is small compared with seed-to-seed noise. The noisy
  and babble advantages are not.
- Silence false accepts are **not** an advantage. QuartzNet seed 1 has 5 clean silence false accepts out of 324,
  against 1 and 2 for `optiond`. The counts are small and the noisy counts are mixed (2 and 6 vs 6 and 8), so
  read this as no clear difference, not as a regression, but do not claim an improvement.
- Seed 1 of `optiond` selected a different clean-val threshold (-0.075), so its numbers are not perfectly like
  for like with the others.

### Size, parameters, speed

Seed 0 for both. Model-only numbers come from `vcm.benchmark` (1.5 s window, 1 thread, AMD EPYC, not a Pi).

| | `optiond` | QuartzNet stride 2 |
|---|---|---|
| Parameters | 1,009,725 | 911,189 |
| INT8 ONNX | 1,042,226 B | 994,898 B (limit 1,048,576 B) |
| Forward pass INT8 p50 / p95 | 3.70 / 4.14 ms | 2.30 / 2.34 ms |

End-to-end latency (`vcm.e2e_benchmark`): features, ONNX forward, log-softmax and grammar decode at threshold
-inf, on 200 seeded test clips cropped or padded to 2.5 s, beam 50, INT8, one ORT thread. Median of three
alternating A/B runs.

| | `optiond` | QuartzNet stride 2 |
|---|---|---|
| Output frames | 251 | 126 |
| Total p50 / p95 | 245.7 / 279.8 ms | 134.7 / 171.9 ms |
| Of which features + ONNX forward (p50) | 73.7 ms | 50.1 ms |
| Of which grammar decode (p50) | 172.8 ms | 84.4 ms |
| INT8 vs FP32 winning-intent agreement | 1.000 | 0.995 |

The decoder dominates (70% of the time for `optiond`) and its cost tracks frame count, so halving the frames
roughly halves it. That is the mechanism behind the 0.55x total-latency ratio. The "model" row includes
feature extraction, so it is larger than `benchmark.py`'s model-only figure above.

### Ablations (seed 0, same recipe)

Two further runs to separate possible causes of the gain.

| Run | Epochs | Threshold | Clean exact | Noisy exact | Noisy babble FA | Noisy silence FA |
|---|---|---|---|---|---|---|
| `optiond` | 67 | -0.1 | 3448 | 3345 | 10 | 6 |
| `optiond-wide` (block kernels 61, receptive field ~3 s, otherwise unchanged; 1.036M params) | 56 | -0.075 | 3438 | 3348 | 6 | 6 |
| QuartzNet stride 1 (same layout, no subsampling) | 56 | -0.05 | 3446 | 3358 | 0 | 3 |
| QuartzNet stride 2 | 61 | -0.1 | 3468 | 3416 | 1 | 2 |

- Widening `optiond`'s receptive field from about 1.1 s to about 3 s changed accuracy by -10 clean and +3 noisy,
  which is noise. Receptive field alone is not what is helping.
- Removing the stride from QuartzNet gives results close to `optiond` (-2 clean, +13 noisy). Its babble false
  accepts are low (0), but a count that small is weak evidence.
- Adding stride 2 to the same QuartzNet layout adds 22 clean and 58 noisy clips. That difference is well
  outside the roughly +/-9 clips of binomial noise per run on 3,494 clips.

Reading: the benefit is tied to the 20 ms output frame rate. We did not test why. Plausible but unverified
explanations: fewer blank-dominated frames make CTC training and the per-frame confidence score cleaner;
the layout with 15 separable sub-modules uses the same budget for more nonlinearity. Treat these as
hypotheses.

Limits of the ablations: one seed each; the stride-1 and wide runs ended at 56 epochs (deadline) against
61 and 67 for their comparison runs; and their operating thresholds moved (-0.05, -0.075), so numbers are
approximately but not exactly comparable.

## Acceptance gates

Set before training in `.scratch/quartznet-ctc/tickets/00-RECAP.md`, against seed 0 unless noted.

| Gate | Criterion | Result |
|---|---|---|
| G1 | clean exact >= 0.9818 | 0.9926 pass |
| G2 | noisy exact >= 0.9524 | 0.9777 pass (0.9785 seed 1) |
| G3 | noisy babble FA <= 10/255 | 1/255 pass |
| G4 | INT8 ONNX <= 1,048,576 B | 994,898 B pass |
| G5 | parameters <= 1,009,725 | 911,189 pass |
| G6 | e2e p95 no slower than baseline; p50 ratio <= 0.65 | p95 171.9 vs 279.8 ms; p50 ratio 0.55 pass |
| G7 | INT8-vs-FP32 agreement >= 0.98 and within 1 pp of baseline | 0.995 pass |
| G8 | threshold not at a grid edge | -0.1 pass |
| G9 | no CTC-infeasible loss-bearing rows at stride 2 in train | 0 pass |

## Decisions and deviations that matter

- **Width 192, not 200.** Width 200 fits the parameter budget (978,053) but exports an INT8 file of
  1,062,125 B, over the size gate. Width 192 exports 974,006 B untrained. The INT8 size, not the parameter cap,
  set the width. The 950k parameter floor in the original plan was dropped.
- **Stride 2 is a trade, not free.** Subsampling is only safe if every training transcript fits in the output
  frames. A real-data check found 0 infeasible rows in train and val at stride 2, and exactly 1 in the test
  split (`cv-valid-train__sample-121890_c04.wav`, a 0.59 s Common Voice clip whose transcript needs 55 frames).
  It is scored but never trained on, and was not filtered. At stride 4 there are 15 in train, which is why
  stride 4 was not trained.
- **The code is stride-aware.** CTC loss uses `model.output_lengths` at both call sites; passing feature
  lengths to a strided model would silently mis-train (`zero_infinity=True` hides it). The count of infeasible
  items is logged per epoch and saved in `loss_history.json`.
- **Checkpoints are self-describing.** They carry a `model_type` key; an absent key means MatchboxNet, so every
  existing checkpoint loads unchanged. The class is chosen only from a closed table, never from names in the
  checkpoint.
- **The noisy gate was ported.** The recorded production noisy numbers had been produced from a different
  worktree (`me2-iteration3`) that can only load MatchboxNet. The gate is now in this branch
  (`vcm/noisy_eval.py`), with the acoustic-ghost parts left out. On the production checkpoint it reproduces the
  recorded numbers exactly: threshold -0.1, clean exact 3448, noisy exact 3345, noisy babble FA 10, noisy
  silence FA 6.
- **Seed-1 trainings were restarted once** after the first launch shared a busy node with other CPU-heavy
  jobs and ran at about half the epoch rate. The reported runs ran at the same pace as each other.

## What is not done

- **Production still serves `optiond`.** Nothing in `make app-pipeline`, the streaming `MODEL_REGISTRY` or the
  serving defaults has changed.
- **The operating point is not re-tuned.** The accept threshold (-0.1 here), the incomplete-prefix margin (4.0)
  and the opt-in per-character trial threshold (-1.204) were all tuned on `optiond`'s output and depend on the
  frame count or raw beam mass. They are not portable to a model with half the frames.
- **Two tools assume 10 ms per posterior frame** (`vcm/incomplete_calibration.py`,
  `vcm/optionb/incomplete_probes.py`) and give wrong time values on a strided model. Neither is on the
  evaluation or streaming gate path.
- **No real-audio soak.** The 8-hour cascade soak validated `optiond` only
  (`docs/CASCADE-SOAK-TEST.md`). QuartzNet has synthetic noisy-gate evidence only.
- **No Pi measurement.** All latency figures are from a shared EPYC server. Raspberry Pi 4/5 end-to-end p95,
  fresh-process RSS and dropped streaming windows are untested.
- **Only two seeds** for the main comparison and one for the ablations; INT8 size and latency were measured on
  seed 0 only.
- The silence false-accept count in seed 1 (above) is unexplained.

These are written up as three tickets for a separate worktree in `.scratch/quartznet-promotion/tickets/`:
stride-aware tools plus serving wiring, threshold and margin recalibration, and a QuartzNet cascade soak
test with promotion criteria fixed in advance.

## Reproducing

```bash
# train (GPU: any free one; never GPU 6). QuartzNet; use --preset optiond for the baseline
CUDA_VISIBLE_DEVICES=7 uv run python -m me2_voicegen.vcm.train \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv \
  --out-dir out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m \
  --preset quartznet5x3 --device cuda:0 --max-minutes 135 --seed 0 --p-rir 0.7

# gate eval (clean + fixed-seed noisy), ~45-70 min
CUDA_VISIBLE_DEVICES=7 uv run python -m me2_voicegen.vcm.evaluate \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
  --checkpoint out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/checkpoints/checkpoint.pt \
  --out-dir out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/noisy_eval \
  --device cuda:0 --beam-width 50 --grammar optionb --noisy-eval-seed 0

# export + model benchmark, then end-to-end latency (both run dirs, alternate A/B)
uv run python -m me2_voicegen.vcm.benchmark --checkpoint <run>/checkpoints/checkpoint.pt \
  --out-dir <run> --manifest out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv \
  --calibration-samples 32 --n-iters 100
uv run python -m me2_voicegen.vcm.e2e_benchmark --run-dir <run> \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv --split test \
  --n-clips 200 --window-s 2.5 --beam-width 50 --grammar optionb --seed 0 --out <out.json>
```

Presets: `quartznet5x3` (stride 2), `quartznet5x3-s1` and `optiond-wide` (ablations only, not shipping presets).
Run directories: `out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m`, `...-seed1-...`, `quartznet5x3-s1-...`,
`optiond-wide-...`, `optiond-seed1-...`; comparison artifacts in `out/vcm/quartznet-ctc-compare/`.

## License note

The checkpoints are trained on data that includes ESC-50 background noise (CC-BY-NC-SA-4.0) and inherit that
license: non-commercial use, share-alike on redistribution (`docs/VCM-CONTRACT.md` section 9).
