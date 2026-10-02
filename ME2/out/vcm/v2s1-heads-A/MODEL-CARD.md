# v2s1-heads-A: QuartzNet-5x3-tiny CTC + attention heads (seed 1)

- **Architecture:** `quartznet5x3-heads`, stride 2, 954,237 parameters. CTC head plus an attention-pooled 21-way intent head
  and six slot heads. Weights: ctc 1.0 / intent 0.3 / slot 0.1.
- **Data:** public `airimonda/ai231-me2-voice-commands` (commit 6947f130; train split incl. val speakers and
  `synthetic_negatives`), imported with `vcm.optionb.import_ai231`. No ESC-50 or other external noise. Trained under the
  dataset's own source terms (several are non-commercial); see the dataset card.
- **Training:** `vcm.train --preset quartznet5x3-heads --seed 1 --patience 20 --max-minutes 135 --p-rir 0.7
  --p-timestretch 0.25 --noise-source dataset --p-babble 0.15 --perturbation-plan table --dump-plan`; early-stopped at epoch 79; the checked-in
  checkpoint is the best epoch, 59 (val loss 0.6266). Learning rate had not annealed (OneCycle sized to 150 epochs).
- **Evaluation (ai231 test, exact rows):** CTC 87.5% intent+slot clean / 76.1% perturbed; classifier (T=0.9) 95.4% / 88.9%;
  holdout (202 clips) CTC 52.2%, classifier 58.1%; false accepts on 226 non-commands 0.4% (CTC), 4.4% (classifier). Real
  Filipino speaker in the holdout: 6.0% (CTC), 8.3% (classifier). Details: `eval/` and `docs/CTC-ATTENTION.md`.
- **ONNX (`export/`):** CTC path only (`features` (B, 40, T) -> `logits` (B, T_out, 29)); the heads are not exported.
  fp32 3.6 MB, static INT8 1.0 MB. The classifier needs the PyTorch checkpoint (`forward_heads`).
- **Status:** best of the two v2 heads-A seeds for streaming tests; chosen on one seed's test split. Streaming behaviour
  not measured. Not promoted.
- **`eval/` notes:** `final-table-v2.*` is the two-seed comparison. `report-all.*` is the scorer's report for the seed-0 pair
  plus the leaky v1 baseline and the fil50 models (its Gate A line is seed 0, which fails; seed 1 passes); its "ESC-50"
  labels are stale and refer to the dataset's own `noise_only` clips.
- **Metadata:** `checkpoints/checkpoint.pt` and `metadata/eval_report.json` carry a corrected licence note (the training code
  writes a hard-coded ESC-50 note into every checkpoint; this model used no ESC-50). Streaming usage: `docs/CTC-ATTENTION.md`.
