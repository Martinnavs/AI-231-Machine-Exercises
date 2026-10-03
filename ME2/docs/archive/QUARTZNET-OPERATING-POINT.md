# QuartzNet-5x3-tiny operating point (quartznet-promotion ticket 02)

Date: 2026-10-01 · Worktree branch `quartznet-promotion` · Checkpoint
`out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m` (seed 0, width 192, stride 2;
`docs/archive/QUARTZNET-STUDENT.md` for the model comparison). The three accept/reject
knobs were tuned on the production MatchboxCTC model (`optiond`,
`out/vcm/option-d-fil50-ambient-rir-135m`) and depend on posterior frame count
or raw beam mass, so each was re-measured on the seed-0 QuartzNet checkpoint
with the same data, beam width and selection rules.

## Operating point summary

| knob | optiond (production) | QuartzNet (chosen) | evidence |
|---|---|---|---|
| `mean_frame` threshold | -0.075 served (`APP_PIPELINE_THRESHOLD`); -0.1 clean-val-chosen | **-0.1** (section 1) | `out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/metadata/eval_report.json` (full val grid -> -0.1, clean + noisy test) and `.../noisy_eval/metadata/eval_report.json` (same point, gate eval); `out/vcm/quartznet-ctc-compare/eval-th-0p075/eval_report.json` (the -0.075 test); optiond reference `out/vcm/quartznet-ctc-compare/optiond-parity/metadata/eval_report.json` |
| incomplete-prefix margin | 4.0 | **20.0** (section 2) | `out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/metadata/incomplete_calibration_realtarget/{val_selection.json,test_report.json}`; optiond reference `out/vcm/option-d-dataset-v2/metadata/incomplete_calibration_beam50_realtarget/` |
| `per_char` opt-in trial | -1.204 (band -1.1 .. -1.3) | not adopted: val-Youden lands at -1.3, the band edge (section 3, informational) | `out/vcm/quartznet-ctc-compare/eval-perchar/metadata/eval_report.json` (grid) and `.../eval-perchar-1p204/metadata/eval_report.json` (fixed -1.204) |

The Makefile values `APP_QUARTZNET_THRESHOLD` / `APP_QUARTZNET_MARGIN`
(`make app-pipeline-quartznet`) carry the chosen values.

## 1. `mean_frame` threshold: -0.075 vs -0.1

Both points are interior to `DEFAULT_THRESHOLD_GRID` (not grid edges). Clean-val
Youden sweep on the fil50 manifest (3,330 val targets, 738 reject probes):
-0.075 -> accept 0.9826, reject FA 6/738, J 0.97445; -0.1 -> accept 0.98649,
reject FA 8/738, J 0.97565 (the sweep's choice, matching the seed-0 gate eval).

Test split (3,494 targets, 255 babble, 324 silence), same fixed-seed noisy
perturbation as `docs/archive/QUARTZNET-STUDENT.md`:

| point | clean exact | clean babble FA | clean silence FA | noisy exact | noisy babble FA | noisy silence FA |
|---|---|---|---|---|---|---|
| QuartzNet -0.075 | 3463 (0.9911) | 0/255 | 0/324 | 3390 (0.9702) | 0/255 | 1/324 |
| **QuartzNet -0.1 (chosen)** | 3468 (0.9926) | 0/255 | 0/324 | 3416 (0.9777) | 1/255 | 2/324 |
| optiond -0.1 (parity rerun) | 3448 (0.9868) | 3/255 | 1/324 | 3345 (0.9574) | 10/255 | 6/324 |

Pre-registered floors (G1/G2): noisy exact >= 0.9524, noisy babble FA <= 10/255.
Both QuartzNet points clear both floors by wide margins.

**Chosen: -0.1.** It is the point the QuartzNet clean-val Youden sweep selects
(and the threshold at which every gate in `docs/archive/QUARTZNET-STUDENT.md` was
measured), it is not at a grid edge, and it dominates -0.075 on recall at both
conditions (clean +5 exact, noisy +26 exact) for a negligible false-accept
increase (noisy babble 0 -> 1, still 1/10 of optiond's 10/255; noisy silence
1 -> 2). The stricter -0.075 (what `optiond` serves) buys only 1 noisy babble
FA and 1 noisy silence FA back against 26 noisy exact clips -- not the trade
this data supports.

Commands:

```bash
# test at -0.075 (writes the run dir's metadata/eval_report.json, the layout
# the streaming CLI's resolve_threshold() reads -- docs/STREAMING-CONTRACT.md)
make quartznet-eval-report VCM_DEVICE=cuda:7  # full default val grid -> -0.1
CUDA_VISIBLE_DEVICES=7 uv run python -m me2_voicegen.vcm.evaluate \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
  --checkpoint out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/checkpoints/checkpoint.pt \
  --out-dir out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m \
  --device cuda:0 --beam-width 50 --grammar optionb --noisy-eval-seed 0 \
  --threshold-grid -0.075
# -0.1 point: out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/noisy_eval (existing gate eval)
```

## 2. Incomplete-prefix margin

Reference (optiond): `out/vcm/option-d-dataset-v2/metadata/incomplete_calibration_beam50_realtarget/`
-- beam 50, real-target manifest `out/conversions/v2/optionb-v3-vcmx/manifest.realtarget.csv`
(195 real `vcm_balanced` val targets / 129 test targets + babble/silence rows),
probes regenerated per checkpoint, margin grid -10..50 (incl. 4, 4.5),
regression budget 1.0 pp, threshold -0.075 (its own val sweep). Result: margin 4.0
(val target exact 0.9795, far_excl_digital_zero 0.0403).

QuartzNet rerun: same manifest, same grid, same rule, probes regenerated with the
QuartzNet checkpoint (stride-aware `vcm/optionb/incomplete_probes.py`,
10,672 val / 10,688 test probes from the same 2,670/2,672 source rows), beam 50.
Its threshold is the one its own val sweep chooses on the real-target subset.

| | optiond (reference) | QuartzNet |
|---|---|---|
| threshold used (own val sweep) | -0.075 | -0.01 |
| **chosen margin** | 4.0 | **20.0** |
| val target exact (baseline -> gated) | 0.9846 -> 0.9795 | 0.9897 -> 0.9897 (zero cost at every grid margin <= 20) |
| val far_excl_digital_zero (selection criterion) | 0.0776 -> 0.0403 | 0.0232 -> 0.0139 |
| val FAR at margin 4.0 | 0.0403 (chosen) | 0.0212 |
| test (holdout) target exact (baseline -> gated) | 114/129 (0.8837) -> 112/129 (0.8682) at 4.0 | 116/129 (0.8992) -> 115/129 (0.8915) at 20.0 |
| test (holdout) far_excl_digital_zero (baseline -> gated) | 568/7298 (0.0778) -> 333/7298 (0.0456) | 35/2648 (0.0132) -> 15/2648 (0.0057) |
| test (holdout) gate-caused FRR | 2/114 | 1/116 |

Holdout is consistent with the val selection: the gate halves the
non-digital-zero probe FAR (35/2648 -> 15/2648; all-probe FAR 325/10,688 ->
159/10,688) and costs one target clip (a `wake me up at six` row re-scored as
`ALARM`, 0.78 pp, inside the 1.0 pp regression budget). QuartzNet's holdout
target exact is +2 clips above the optiond reference at baseline (116 vs 114).

**Is 4.0 still adequate? No -- it is no longer the calibrated value.** Two
structural findings, both consistent with `docs/archive/QUARTZNET-STUDENT.md`'s reading
that the 20 ms frame rate cleans up the per-frame mass structure:

1. QuartzNet rejects incomplete prefixes intrinsically. Gate-off val FAR is
   0.0232 (291/10,672) at its own sweep threshold, against 0.0776 for the
   optiond reference at its own sweep threshold -- the model alone removes most
   of what the margin gate was built for.
2. The margin gate still buys FA reduction at zero recall cost across the whole
   grid up to 20.0 (val target exact is 0.9897 at every margin 0..20; the
   selection rule's 1.0 pp budget is first broken at 30.0, 0.9692). The rule
   therefore picks the grid maximum that stays in budget: **20.0**
   (far_excl_digital_zero 0.0139). Margin 4.0 would still be "safe" (zero
   recall cost, FAR 0.0212) but leaves ~50% more residual probe FAR than the
   calibrated choice; 20.0 costs no measured recall on this set.

The threshold difference (-0.01 vs -0.075) is a property of the two models'
own val sweeps on the 195 real-target subset (the real recordings accept
readily at -0.01); the reference protocol calibrates margin at the threshold
the sweep chooses, which is kept here for like-for-like comparison.

Commands:

```bash
PYTHONPATH=<worktree>/ME2/src uv run python -m me2_voicegen.vcm.optionb.incomplete_probes \
  --manifest out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv \
  --checkpoint out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/checkpoints/checkpoint.pt \
  --out-dir out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/incomplete_prefix/probes/<split> \
  --split <val|test> --device cpu
PYTHONPATH=<worktree>/ME2/src uv run python -m me2_voicegen.vcm.incomplete_calibration select \
  --manifest out/conversions/v2/optionb-v3-vcmx/manifest.realtarget.csv \
  --probe-manifest out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/incomplete_prefix/probes/val/manifest.csv \
  --checkpoint out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/checkpoints/checkpoint.pt \
  --out-dir out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m/metadata/incomplete_calibration_realtarget \
  --device cuda:0 --beam-width 50 --margins=-10,-5,-3,-2,-1,-0.5,0,0.5,1,2,3,4,4.5,5,6,7,10,15,20,30,50
# holdout: same, subcommand `holdout --split test`, probe-manifest .../probes/test/manifest.csv
```

## 3. `per_char` opt-in trial (informational)

`evaluate --score-mode per_char` sweeps `PER_CHAR_THRESHOLD_GRID` on val and
reports the test split at its Youden choice (optiond reference: -1.1 chosen,
-1.204 the shipped trial point, band -1.1 .. -1.3 per
`docs/archive/DENSE-SCORING-DECISION.md`). Optiond baselines: C2 (-1.204) removed all
test clean/noisy FAs, 17 VCM-only soak triggers vs 303 at B, 737/821 first-trigger
recall vs 720.

QuartzNet grid run (val sweep in
`out/vcm/quartznet-ctc-compare/eval-perchar/metadata/eval_report.md`): the
val curve is flat across the whole plausible band -- accept 0.992 -> 0.995,
reject FA 0.008 at every band point, J 0.984 -> 0.986 -- so the D2 val
FA-budget rule is satisfied at -1.204, and the Youden maximum lands at
**-1.3, the band's edge** (not a grid edge; the grid runs to -5.0). Test at
-1.3: 3479/3494 exact (0.996), babble 1/255, silence 0/324.

Test at the optiond-shipped point -1.204 (same seed, fixed grid,
`out/vcm/quartznet-ctc-compare/eval-perchar-1p204/metadata/eval_report.md`):

| point | clean exact | clean babble FA | clean silence FA | noisy exact | noisy babble FA | noisy silence FA |
|---|---|---|---|---|---|---|
| QuartzNet -1.3 (val-Youden) | 3479 (0.996) | 1/255 | 0/324 | n/a (grid run clean-only) | - | - |
| QuartzNet -1.204 (optiond's point) | 3477 (0.995) | 1/255 | 0/324 | 3452 (0.988) | 2/255 | 1/324 |
| optiond C2 -1.204 (reference) | - | 0/255 | 0/324 | - | 0/255 | 0/324 |

(optiond C2 row: `docs/archive/DENSE-SCORING-DECISION.md` -- "removed all test
clean/noisy FAs, 17 VCM-only soak triggers vs 303 at B, 737/821 first-trigger
recall vs 720".) So -1.204 sits inside the plausible band and satisfies the
D2 val FA-budget rule on QuartzNet, but unlike optiond it leaves a small
residual FA (1 clean / 2 noisy babble); nothing in this run recommends
adopting per_char, and the ticket's non-goal stands (mean_frame stays the
served mode).

## Served configuration

`make -n app-pipeline-quartznet` (from the `quartznet-promotion` worktree):

```
make app-pipeline APP_PIPELINE_MODEL=out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m APP_PIPELINE_THRESHOLD=-0.1 APP_PIPELINE_MARGIN=20.0
.venv/bin/python -m me2_voicegen.vcm.streaming --model out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m --backend onnx --onnx-variant int8 --grammar optionb --threshold -0.1 --required-command-margin 20.0 --score-mode mean_frame --beam-width 50 --gate wakeword --policy single_period --gate-period 3 --wakeword-model out/wakeword-sesame --wakeword-backend onnx --log-periods --source "mic" | .venv/bin/python -m app.forward
```

Switching the served model from `optiond` to this checkpoint is this one Make
target plus these two values; the streaming CLI, INT8 export, and
`resolve_threshold()` contract are unchanged (ticket 01).
