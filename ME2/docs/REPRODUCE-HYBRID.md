# Reproducing the hybrid: what works today and what does not

The goal for this model is reproducibility from public data. This page separates three things: what you can re-run today and get the
documented numbers, what you can retrain, and the part of the training recipe that **cannot yet be rebuilt from the published
datasets**. Nothing below was run end to end on a clean machine; each step says what was actually checked.

## 1. Check the checked-in model (verified)

Needs only the public ai231 dataset and the checked-in INT8 ONNX files (no GPU, no training). About 5 minutes with 8 shards.

```bash
module load uv && make sync                                   # see the README, "Prerequisites"
R=raw_datasets/ai231-me2-voice-commands-v2                    # the downloaded dataset: data/, synthetic_negatives/, supplemental_synth/, variations.csv
uv run python -m me2_voicegen.vcm.optionb.import_ai231 --src $R --out out/conversions/v2/ai231-v2
uv run python -m me2_voicegen.vcm.semantic_eval build-manifest --ai231 out/conversions/v2/ai231-v2/manifest.csv \
  --out out/conversions/v2/ai231-v2/manifest.eval-exact.csv                           # no --noise: dataset-only noise
H=out/vcm/hybrid-ctcwide-clsxl
for k in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=2 uv run python scripts/hybrid_score.py score \
  --manifest out/conversions/v2/ai231-v2/manifest.eval-exact.csv --split test --shard $k --n-shards 8 \
  --ctc-checkpoint $H/ctc-wide/export/vcm_model.int8.onnx --cls-checkpoint $H/cls-xl/export/vcm_heads.int8.onnx \
  --cls-threshold 0.8787 --out-dir out/vcm/hybrid-eval/repro & done; wait
uv run python scripts/hybrid_score.py summarize --dir out/vcm/hybrid-eval/repro --split test --cond clean
```

Expected (checked 2026-10-03 on the INT8 ONNX files): `4149 rows, 3823 targets, 326 babble/silence`, **`fallback: 98.82% (3778/3823)`**,
`FA 10/326` (the PyTorch checkpoints give 11/326), and `agree: 97.78%`. Add `--noisy-seed 0` (and `--cond noisy`) for the perturbed gate
(about 94.8% INT8); use `--split holdout` on the full `manifest.csv` for the 186-command holdout (about 77%). The persona-only and
old-internal-test rows need the persona manifest and the internal data, which this path does not cover.

## 2. Retrain from the manifests (works if you have them)

Given `out/conversions/v2/ai231-fil50-supp/manifest.csv` (see section 3 for how it was built), the training and export commands in
[`CURRENT-MODEL.md`](CURRENT-MODEL.md) reproduce the models up to seed noise: about 94 min (wide) and 43 min (XL) on one A100.
Expected: best epoch about 38 / 23, validation loss about 0.39 / 0.34. One seed each, so a rerun will differ; the size of that
difference has not been measured. Retraining, export and scoring were all run for these two models; the *rebuild of the manifest* was not.

## 3. Rebuilding the training manifest (NOT reproducible from public data yet)

`ai231-fil50-supp` = ai231 v2 (public) + persona clips + the ai231 `supplemental_synth` train-voice clips (public).

| Step | Code | Inputs | Public? |
|---|---|---|---|
| ai231 v2 -> `ai231-v2/manifest.csv` | `vcm.optionb.import_ai231` | the ai231 dataset | yes |
| persona-padded base `ai231-fil50/manifest.csv` | `accent_balance/build_ai231_fil50.py` (cap 50/12/35 per wording, 10/2/5 voices, seed 0) | `optionb-v3-vcmx-fil50/manifest.csv` persona clips, **plus 1,849 freshly generated gap-fill clips** (`gen_manifest.csv`, `qa_pass.csv`) | **no** |
| + `supplemental_synth` -> `ai231-fil50-supp/manifest.csv` | `scripts/build_ai231_fil50_supp.py` | the base manifest and `supplemental_synth/` | the second input yes, the base no |

Why it fails: the published persona dataset (`martinnavarez/ai231-fil-supplemental-data`, 14,120 clips) was exported from the internal
persona manifest alone. It does **not** contain the 1,849 gap-fill clips that cover the thin wordings, and its files were renamed (`fil50_NNNNN_<voice>_<COMMAND>.wav`), so the builder's selection (which sorts by
file name) would pick different clips even from the same pool. A rebuild from the public data would therefore be a similar recipe, not
the same training set, and its numbers would not be comparable to the ones in this repo.

What would close it (not done; publishing needs the owner's go-ahead):
1. Publish the 1,849 gap-fill clips (or regenerate and publish the full pool) and the exact list of persona clips used per split.
2. Add a builder that reads only the two public datasets and reproduces `ai231-fil50-supp/manifest.csv`, with a test that its row
   counts match (persona train 4,535 / val 924 / test 2,946; train 18,300, val 2,717, test 7,639, holdout 202).
3. Retrain once from that manifest and compare against the checked-in models.

Until then, treat the training recipe as documented but only the evaluation of the checked-in model (section 1) as reproducible.

## 4. Other items a reviewer will ask about

| Item | State |
|---|---|
| Single reproduction command | no; sections 1-2 are commands per step; `make hybrid-decode`, `hybrid-stream`, `hybrid-test` cover the checked-in model |
| Dataset DOI and licence | the two Hugging Face datasets are public; no DOI; several source corpora are non-commercial; the persona reference-voice provenance is an open item the uploader accepted (see the dataset card) |
| Test set with unseen speakers | ai231 test is speaker-disjoint from train; the persona test uses 5 held-out voices; the real-speaker holdout is one person (186 clips) |
| Baselines | same architecture at 1, 4 and 10 MB, the internal-data model (not reproducible) and old production (inflated on ai231 test by training overlap) |
| Pi 4 | not measured; no Pi hardware on the node |
| Seeds | one per model |
