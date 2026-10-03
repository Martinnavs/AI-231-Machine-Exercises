# Reproducing the hybrid: what works today and what does not

The goal for this model is reproducibility from public data. This page separates three things: what you can re-run today and get the
documented numbers, what you can retrain, and the part of the training recipe that is
rebuildable from the published datasets (section 3). Nothing below was run end to end on a clean machine; each step says what was actually checked.

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

## 3. Rebuilding the training manifest (reproducible from public data)

`ai231-fil50-supp` = ai231 v2 (public) + persona clips + the ai231 `supplemental_synth` train-voice clips (public).

```bash
PYTHONPATH=src uv run python scripts/rebuild_training_manifest.py --ai231 raw_datasets/ai231-me2-voice-commands-v2 --out out/conversions/v2/rebuilt
```

It downloads the persona dataset (default split and the `gap_fill` and `numeral_wordings` configs of `martinnavs/ai231-fil-supplemental-data`), runs the two builders below,
extracts the audio the manifest uses, and asserts the result equals the committed `recipes/ai231-fil50-supp/manifest.ai231-fil50-supp.csv`. **Verified
2026-10-03** (about 36 s with local copies of the data): **all 28,858 rows equal the committed manifest**, and the 8,405 persona and gap-fill clips it uses were
extracted from the public shards (the persona ones byte-identical to the originals, by SHA-256). Four of those clips are spelled-out-number wordings
("wake me up at six am", "set an alarm for eight am", two "...twenty two degrees" commands) that the persona dataset's default split leaves out; they are
published as the small `numeral_wordings` config.

| Step | Code | Inputs | Public? |
|---|---|---|---|
| ai231 v2 -> `ai231-v2/manifest.csv` | `vcm.optionb.import_ai231` | the ai231 dataset | yes |
| persona-padded base `ai231-fil50/manifest.csv` | `accent_balance/build_ai231_fil50.py` (cap 50/12/35 per wording, 10/2/5 voices, seed 0) | the persona pool and the gap-fill jobs, as committed CSVs in `recipes/ai231-fil50-supp/`, plus their audio | yes, except 4 clips |
| + `supplemental_synth` -> `ai231-fil50-supp/manifest.csv` | `scripts/build_ai231_fil50_supp.py` | the base manifest and `supplemental_synth/` | yes |

How it closes the earlier gap: the published persona dataset (14,120 clips) renamed its files (`fil50_NNNNN_<voice>_<COMMAND>.wav`) and the builder selects by the
original name, so `recipes/ai231-fil50-supp/persona-pool/published_file_map.csv` maps the two by audio SHA-256 (9,774 of the 9,958 pool clips are published, counting the four in `numeral_wordings`; the rest
are old wordings and other spelled-out numbers that no manifest row uses). The 1,849 gap-fill clips are the `gap_fill` config, with `qa_passed`, `qa_rescued` and `manifest_split` columns.
The selection is a pure function of the committed CSVs: run from them it reproduced all 24,875 base rows and all 28,858 final rows exactly, with nothing dropped.

What is still not reproducible:
1. Regenerating the audio: generation is not seeded, so the same job gives the same wording, voice and label but different audio. The published clips
   are the audio to use. The reference-voice prompt recordings behind `persona-pool/voices.csv` are not published.
2. Retraining on the rebuilt manifest was not re-run, and the seed-to-seed spread of the models is unmeasured, so a retrain will land near the checked-in
   models, not on them.

## 4. Other items a reviewer will ask about

| Item | State |
|---|---|
| Single reproduction command | no; sections 1-2 are commands per step; `make hybrid-decode`, `hybrid-stream`, `hybrid-test` cover the checked-in model |
| Dataset DOI and licence | the two Hugging Face datasets are public; no DOI; several source corpora are non-commercial; the persona reference-voice provenance is an open item the uploader accepted (see the dataset card) |
| Test set with unseen speakers | ai231 test is speaker-disjoint from train; the persona test uses 5 held-out voices; the real-speaker holdout is one person (186 clips) |
| Baselines | same architecture at 1, 4 and 10 MB, the internal-data model (not reproducible) and old production (inflated on ai231 test by training overlap) |
| Pi 4 | measured 2026-10-03: same answers as the server, but decode 443 / 764 ms mean / p95 per window, RTF 3.21 at the 0.25 s stride (does not keep up live); see `BENCHMARKS.md` |
| Seeds | one per model |
