# Recipe for the training manifest `ai231-fil50-supp`

Metadata only (CSV, no audio). Together with the public audio it says which clip went into which split and why. Context and what
is still missing: [`docs/REPRODUCE-HYBRID.md`](../../docs/REPRODUCE-HYBRID.md) section 3.

| File | What it is |
|---|---|
| `manifest.ai231-fil50-supp.csv` | The final training manifest (train 18,300, val 2,717, test 7,639, holdout 202) |
| `manifest.ai231-fil50.csv` | The persona-padded base it extends (`build_ai231_fil50.py` output) |
| `manifest.optionb-v3-vcmx-fil50.csv` | The persona pool the base builder reads (`fil50_persona` rows, original file names) |
| `persona-pool/gen_manifest.csv`, `qa_pass.csv`, `qa_summary.md` | The original persona generation (19,281 jobs) and which passed the Whisper check |
| `persona-pool/voices.csv` | The 140 reference voices (the 17 persona voices are among them) and their prompt text (the prompt audio is not published) |
| `gap-fill/jobs.csv` | The 1,849 gap-fill jobs from `accent_balance/plan_gap_jobs.py` (seeded): voice, exact text, label |
| `gap-fill/gen_manifest.csv` | What `accent_balance/generate.py` produced for them (the two GPU shards, concatenated) |
| `gap-fill/qa_pass.csv`, `qa_summary.md` | 1,042 of the 1,849 passed the check |
| `gap-fill/qa-reports/`, `qa-shim/` | The inputs of `build_ai231_fil50.rescued_job_ids` (278 more clips that failed only on "100%" vs "100 percent") |

Audio: the 1,849 gap-fill clips, all QA outcomes with `qa_passed`, `qa_rescued` and `manifest_split` columns, are the `gap_fill` config of
[`martinnavs/ai231-fil-supplemental-data`](https://huggingface.co/datasets/martinnavs/ai231-fil-supplemental-data) (written by
`scripts/export_gap_fill_hf.py`; the default `train` split is unchanged). 1,157 of them are in the final manifest.

Not recorded here: generation is not seeded, so re-running a job gives the same wording, voice and label with different audio, and the
renamed files of the published persona pool (`fil50_NNNNN_<voice>_<COMMAND>.wav`) are not mapped back to `job_id`.
`tests/test_recipe_ai231_fil50_supp.py` checks the counts above.
