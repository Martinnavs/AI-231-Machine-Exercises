# Reproducing the hybrid: what works today and what does not

The goal for this model is reproducibility from public data. This page separates three things: what you can re-run today and get the
documented numbers, what you can retrain, and the part of the training recipe that is
rebuildable from the published datasets (section 3). **Shortcut:** `make reproduce` runs sections 1 and 3 and the soak in one go (see section 4); the steps below are what it runs.

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

Run alone like this, it downloads the persona dataset (default split and the `gap_fill` and `numeral_wordings` configs of `martinnavs/ai231-fil-supplemental-data`). `make reproduce` instead
passes the persona clips from the DOI dataset's `supplemental_fil/` (`--persona-dir`) and pins `martinnavs` at revision 230c8b8 for the gap-fill and numeral clips; both give the same 28,858 rows (checked 2026-10-10). It runs the two builders below,
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
| Single reproduction command | `make reproduce` (below) |
| Dataset DOI and licence | `airimonda/ai231-me2-voice-commands` revision `e8283202634f23257ad944ee104b9de7a1223bb5`, DOI 10.57967/hf/10723 (Ailene Nunez 2026, "ai231-me2-voice-commands (Revision e828320)", Hugging Face). The gap-fill (1,849) and numeral-wording (4) clips are not in it: they come from `martinnavs/ai231-fil-supplemental-data` @ `230c8b8`; persona clips come from the DOI dataset's `supplemental_fil`. Several source corpora are non-commercial; the persona reference-voice provenance is an open item the uploader accepted (see the dataset card) |
| Test set with unseen speakers | ai231 test is speaker-disjoint from train; the persona test uses 5 held-out voices; the real-speaker holdout is one person (186 clips) |
| Baselines | every model tried, with sources and caveats: [`BASELINES.md`](BASELINES.md) |
| Pi 4 | same answers as the server; with the fast beam search (beam 50, 1 thread, fan on) estimated live latency is 0.48 / 0.92 s median / p95 and the p95 real-time factor is 0.96 at the 0.25 s stride, so it keeps up live (just). The original beam search (p95 RTF 3.21) did not; see `BENCHMARKS.md` |
| Seeds | one per model |

### The single command

```bash
make reproduce                                          # data, manifest, verify, soak: about 9 min on a server CPU, ~4.8 GB download
make reproduce REPRO_TRAIN=1 REPRO_GPU=N                # also retrain both models (about 2.5 h on one A100; never GPU 6)
make reproduce REPRO_STAGES=verify,soak                 # a subset of stages
```

Measured on a server CPU (all 12 checks PASS): data 0:29, manifest 6:19, verify 1:06, soak 1:15. Each stage skips if its output exists. The report is `out/reproduce/REPORT.md`: one row per check (PASS/FAIL, expected, got), stage wall-clock, and the data source and revision used.
Expected values are in `recipes/reproduce/expected.json`. Retrained-model checks report the delta and a plus or minus 2-point band (within/outside), not PASS/FAIL.
Data source: the ai231 shards and persona clips (`supplemental_fil`) from the DOI revision above; the gap-fill and numeral-wording clips from `martinnavs/ai231-fil-supplemental-data` @ `230c8b8` (the driver looks in the DOI dataset first and falls back).
Holdout: 143/186 (76.9%) is on the full imported `manifest.csv` (`--split holdout`), as in section 1.
