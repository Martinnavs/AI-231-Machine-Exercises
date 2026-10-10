# Baselines: every model tried, side by side

This page puts every model of the ai231 work in one table, so a reader can see what the hybrid is compared against.
Nothing here was re-run: each number is copied from the file named next to it. **n/m = not measured** (no source has the number; it is not guessed).
All numbers are intent + slot accuracy on the CTC path (threshold -0.1, margin 4.0, beam 50) unless a row says otherwise. One training run (one seed) per model.
Read the caveats at the end first if you plan to quote a number.

Source keys: **[F]** = `docs/AI231-FIL50.md` (section named in the cell notes below); **[B]** = `docs/BENCHMARKS.md`; **[M]** = [`out/vcm/v2s1-heads-A/MODEL-CARD.md`](https://github.com/Martinnavs/AI-231-Machine-Exercises/blob/4812047/ME2/out/vcm/v2s1-heads-A/MODEL-CARD.md) (removed from `master`, kept at commit `4812047`);
**[H]** = `out/vcm/hybrid-ctcwide-clsxl/eval/headline-metrics.md`; **[X]** = `out/vcm/hybrid-ctcwide-clsxl/eval/report-xl-full.md` and `report-xl-nonoverlap.md`;
**[C]** = `docs/CURRENT-MODEL.md`; **[O]** = `out/vcm/hybrid-ctcwide-clsxl/eval/hybrid-onnx-vs-pytorch.md` (quoted in [F] "Hybrid in ONNX").
`out/vcm/ai231fil50-compare/*.md` is not in this branch; its tables are quoted in [F] and [X].

## Main table

| Model | Params | INT8 MB | Training data | ai231 test clean (3,823) | Perturbed | Holdout (186) | Leak-free subset (393) | Babble FA (of 76) / silence-noise FA (of 100) | Pi 4 latency |
|---|---|---|---|---|---|---|---|---|---|
| **1 M QuartzNet, ai231 only** (`v2s1-heads-A`, seed 1) | 954,237 [M] | 1.0 (CTC only) [M] | public (ai231) | 87.5% [M] | 76.1% [F r1 table] | 52.2% [F r1 "Holdout"; [M] says 202 clips] | n/m | 0 / 0 [F r1 table] | n/m |
| **P** plain CTC, ai231 + persona (round 1) | n/m | n/m | public (ai231 + persona clips) | 90.2% [F r1 table] | 76.8% | 60.2% [F r1 "Holdout"] | 79.4% [F fair-comparison table] | 0 / 0 | n/m |
| **H** CTC + heads (round 1) | 954,237 (same preset as `v2s1-heads-A`) | n/m | public | 91.7% [F r1 table] | 79.8% | 65.6% [F r1 "Holdout"] | n/m | 1 / 0 | n/m |
| **H-sched** (H, schedule fixed) | same preset as H | n/m | public | 88.0% [F r2/3 verdicts] | 71.9% | 58.1% [F "Holdout, all 186"] | n/m | 1 / 0 | n/m |
| **H-all** (+ `supplemental_synth`, no stacked noise, random noise offset) | same preset as H | n/m | public | 92.4% [F r2/3 verdicts] | 77.5% | 60.8% [X] | 83.0% [F fair-comparison table] | 0 / 0 | n/m |
| **H-wide** = wide CTC alone | 4,181,309 [C] | 4.16 [C] | public | 95.0% [H] | 87.8% [F] | 66.7% [H] | 90.1% [F fair-comparison table] | 0 / 0 [X] | n/m (server: see speed section) |
| **H-xl**, CTC path | 9,989,677 [C] | 9.86 (CTC export) [MODEL-CARD] | public | 94.0% [X] | 83.3% [X] | 71.5% [X] | n/m (1,261-row subset: 91.8% [X nonoverlap]) | 1 / 0 [X] | n/m |
| **H-xl**, classifier path alone (threshold 0.8787) | 9,989,677 [C] | 10.10 (heads export) [C] | public | 97.7% [X] | 92.9% [X] | 76.9% [X] | n/m (1,261-row subset: 95.6% [X nonoverlap]) | 4 / 0 [X] | n/m |
| **Hybrid** = wide CTC + XL classifier fallback, policy A, INT8 ONNX | 14.17 M (4,181,309 + 9,989,677) [B] | 14.3 (4.16 + 10.10) [B] | public | 98.8% (3,778/3,823) [H] | 94.8% [O] | 76.9% (143/186) [H] | n/m | 3 / n/m [H]; (combined 326 non-commands: 3.4% clean, 6.1% perturbed [C]) | est. live latency 0.48 / 0.92 s median / p95; compute 100 / 239 ms mean / p95 per window; RTF p95 0.96 (fast beam, beam 50, 1 thread, fan) [B] |
| **Old production, QuartzNet s2** (`quartznet5x3-s2-fil50-ambient-rir-135m`, the "old production" of [F]) | n/m | n/m | internal (overlaps ai231 test) | 96.9% [F] (inflated, see caveats) | 93.5% [F] | 83.9% [F r1 "Holdout"] | 89.6% [F fair-comparison table] | 0 / 2 [F] | n/m |
| **Old production, MatchboxNet** (`option-d-fil50-ambient-rir-135m`, ~1 M) | ~1 M [docs/ARCHIVED-CHECKPOINTS.md]; 1.01 M for the sibling `optiond` preset [docs/MLOPS-PROJECTS.md] | n/m | internal | n/m | n/m | n/m | n/m | n/m | n/m |
| **H-int** (H architecture, internal data minus clips identical to ai231 test/holdout) | 954,237 (same preset as H) | n/m | internal | 96.7% [F r2/3 verdicts] (inflated, see caveats) | 91.8% | 84.9% [X] | 90.8% [F fair-comparison table] | 0 / 2 [F] | n/m |

Notes on cells (section of [F] in brackets):

- Perturbed = fixed-seed reverb + the split's own `noise_only` clips, no babble ([F] "Pre-registered success rules", B).
- Holdout = 186 commands: 84 clips from one real Filipino speaker, 2 Fluent Speech Commands clips, and 100 clips from 2 synthetic voices; scored once ([F] round 1 "Holdout").
- Leak-free subset (393 rows) = ai231 test originals that are neither byte-identical to the internal data nor from a speaker in the internal train split ([F] "A leak in the evaluation").
- Hybrid holdout 76.9% is the INT8 figure; fp32 PyTorch is 77.4% ([O]). Soak and replay results for the hybrid are in [`SOAK-TEST.md`](SOAK-TEST.md) and [B], not repeated here.
- The `v2s1-heads-A` perturbed figure (76.1%) is its seed-1 value in the round-1 reference table; its classifier path is 95.4% clean / 88.9% perturbed ([M]).
- "Old production" has two meanings in the repo docs. The numbers in [F] are for the **QuartzNet stride-2** model `quartznet5x3-s2-fil50-ambient-rir-135m`
  (`docs/CTC-ATTENTION.md` line 80 calls it "production QuartzNet stride 2"). The **MatchboxNet** `option-d-fil50-ambient-rir-135m` is the older "production" in
  `docs/ARCHIVED-CHECKPOINTS.md`; no ai231 score for it was found in the files read, so its row is n/m.

## Size-matched comparisons

Models of similar size, same ai231 test clean (3,823) and perturbed gate. Compare within a block, not across blocks.

| Size class | Model | Training data | Clean | Perturbed | Holdout | Leak-free (393) |
|---|---|---|---|---|---|---|
| ~1 M (0.95 M) | `v2s1-heads-A` | public, ai231 only | 87.5% | 76.1% | 52.2% | n/m |
| ~1 M | P (plain) | public + persona | 90.2% | 76.8% | 60.2% | 79.4% |
| ~1 M | H | public + persona | 91.7% | 79.8% | 65.6% | n/m |
| ~1 M | H-all | public + persona + supplemental | 92.4% | 77.5% | 60.8% | 83.0% |
| ~1 M | H-int | internal | 96.7% (inflated) | 91.8% (inflated) | 84.9% | 90.8% |
| ~1 M | old production (QuartzNet s2) | internal | 96.9% (inflated) | 93.5% (inflated) | 83.9% | 89.6% |
| ~4 M | H-wide | public + persona + supplemental | 95.0% | 87.8% | 66.7% | 90.1% |
| ~10 M | H-xl, CTC | same as H-wide | 94.0% | 83.3% | 71.5% | n/m |
| ~10 M | H-xl, classifier | same as H-wide | 97.7% | 92.9% | 76.9% | n/m |
| ~14 M | hybrid (wide CTC + xl classifier) | public | 98.8% | 94.8% | 76.9% | n/m |

What the blocks say (all from the rows above, one seed):

- **At ~1 M, the training data matters more than anything tried on the public side.** Public-data models go 87.5% to 92.4% clean; the internal-data H-int reaches 96.7% (inflated by overlap), and 84.9% on the holdout against 52.2-65.6% ([F] "What this says").
- **At ~4 M, public data alone is enough on rows the internal data could not have seen.** H-wide scores 90.1% on the 393 leak-free rows against 89.6% (old production) and 90.8% (H-int) ([F] fair-comparison table); on the 1,261 not-byte-identical rows it scores 94.3% against 92.0% and 93.0%. The small subsets are about plus or minus 3 to 7 points.
- **Going to ~10 M did not help the CTC path** (94.0% against 95.0% clean, 83.3% against 87.8% perturbed) but did help the classifier path and the holdout (classifier 97.7% clean; holdout 71.5% CTC / 76.9% classifier against 66.7% / 69.4%) ([F] round 4, [X]).
- **The ~14 M hybrid is the best public-data model on clean and perturbed audio** (98.8% / 94.8%) and about 5 points better than wide CTC alone on the holdout (76.9% against 66.7%). It is not size-matched to anything else here ([B]).
- Not on this branch: a single 14.2 M "xxl" model (98.6% against the hybrid's 98.8%) was mentioned in the brief, but the number was not found in `../me2-wide-private-heads/ME2/docs/AI231-FIL50.md` ("Round 5") or `out/vcm/wide-private-heads/eval/report.md` (the only "xxl" hit in that worktree is a note in `.scratch/wide-private-heads/PLAN.md` that says not to touch the uncommitted mid/xxl presets), so it is omitted. The 4.7 MB single model of Round 5 (frozen wide encoder plus a private classifier branch) scores test clean 98.6% and holdout 136/186 (fp32, policy A); that worktree's verdict is "stop head work" and it is not on this branch either.

## Speed and size

Server numbers only; INT8 ONNX Runtime, one 2.5 s window (126 output frames), medians of 60 runs, measured 2026-10-03 on the training server ([F] "Inference cost: tiny (1 MB) vs wide (4 MB)").
The Raspberry Pi was not used for this table.

| | tiny (`quartznet5x3-heads`, 0.95 M) | wide (`quartznet5x3-wide-heads`, 4.18 M) |
|---|---|---|
| Feature extraction + network forward, 1 thread | 4.3 ms | 14.6 ms |
| Same, 4 threads | 3.3 ms | 7.4 ms |
| Grammar beam search (beam 50), server, original search | ~90 ms | ~90 ms (does not depend on the network) |

A later measurement under load puts the network forward at wide 13.6 ms / 7.5 ms and xl 30.2 ms / 13.1 ms (1 / 4 threads) ([F] "Round 4"). The xl network was not timed in the table above.
The beam search dominates a window for either model; the wide network adds about 10 ms ([F] "Inference cost").

Raspberry Pi 4 Model B, INT8 ONNX, hybrid only, 186-command holdout soak ([B] "Latency"):

| Run | Estimated live latency median / p95 | Compute per window mean / p95 / max | Real-time factor p95 |
|---|---|---|---|
| Fast (numba) beam search, beam 50, 1 thread, fan on | 0.48 / 0.92 s | 100 / 239 / 305 ms | 0.96 |
| Original beam search, beam 50, 1 thread (throttled, under-volted) | 0.99 / 1.49 s | 481 / 802 / 1,654 ms | 3.21 |

The Pi latency of the tiny model, the wide CTC alone and the other baselines in the main table was not measured (n/m). Accuracy is the same in every Pi run ([B]).
The server CPU with the original search needs 76 ms per window on average (p95 132 ms) ([B]).

Sizes: wide CTC INT8 4.16 MB (4,156,784 B), xl heads INT8 10.10 MB, so the hybrid is 14.3 MB INT8 ([B], [F]). The tiny model's INT8 export is 1.0 MB, CTC only ([M]).

## Caveats

- **One seed per model.** Differences under about 3 points on the full test set are not reliable; on the 393-row and 1,261-row subsets and on the 186-clip holdout, 3 to 7 points is noise ([F] "What each change did", "Fair comparison").
- **The leak.** 2,632 of the 4,895 ai231 test and holdout rows are byte-identical to clips in the internal training data, and 75% of the test originals (2,885 of 3,823) come from speakers that also appear in the internal train split.
  Old production and the `fil50-*` models therefore score inflated on the ai231 test: old production drops from 96.9% to 92.0% clean (1,261 rows) and 89.6% (393 rows) ([F] "A leak in the evaluation").
  H-int had byte-identical clips removed but near-duplicates and shared speakers remain, so its full-set numbers are inflated too. **Compare internal-data models with public-data models on the leak-free columns only**, not on the "ai231 test clean" and "Perturbed" columns.
  The public-data models (everything marked public) are not affected.
- **The holdout's real speech is mostly one speaker.** It is 186 commands: 84 clips from one real Filipino speaker, 2 Fluent Speech Commands clips, and 100 clips from 2 synthetic voices (100% for most models on the synthetic part). The human-voice part is those 86 clips ([H]). It measures accent coverage for that one person, not accent coverage in general.
- **The test split was scored many times** during the project and val chose every setting ([B] "Results by split"), so the test numbers are not a single-shot estimate.
- **The pre-registered pass bars (96.5% clean, 90% perturbed) were set from old production's inflated numbers**; old production itself fails them on the non-overlapping rows ([F] "A leak in the evaluation"). Verdicts in [F] are kept as registered.
- **Pi latency exists only for the hybrid.** Other rows: n/m.
- Where a model has no source file in this branch for a cell (the plain P and H sizes, INT8 sizes of the 1 M family, the MatchboxNet scores), the cell is n/m.
