# Documentation index

Start with [`CURRENT-MODEL.md`](CURRENT-MODEL.md): the current model, how to run it, what it scores and how far to trust that.
Then [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) for what can and cannot be reproduced from public data today.

## Current model and experiments (2026-09/10, branch `optionb-ctc-attention`)

| Doc | What it holds |
|---|---|
| [`CURRENT-MODEL.md`](CURRENT-MODEL.md) | The hybrid (wide CTC + XL heads): decision rule, quick start, numbers, data, training, evaluation, code map, open items |
| [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) | Verified check of the checked-in model; retraining; the part of the data recipe that is not public |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Every benchmark table: headline metrics, results by split, seen vs unseen data, the unseen-noise replay, the soak on a Pi and a server |
| [`SOAK-TEST.md`](SOAK-TEST.md) | The deployment-shaped soak: wake word + gap + command with reverb and noise, how to run it (also on a Raspberry Pi), results, tuning, failure analysis |
| [`RASPBERRY-PI.md`](RASPBERRY-PI.md) | Running the hybrid on a Pi: the torch +cu121 fix, the lean install, the microphone command, the soak run |
| [`ARCHIVED-CHECKPOINTS.md`](ARCHIVED-CHECKPOINTS.md) | Which model binaries were untracked from git, where the archive is, what needs them |
| [`AI231-FIL50.md`](AI231-FIL50.md) | The experiment log: persona padding, rules A-D, wider models, internal data, the leak finding, the hybrid, streaming, wake-word polling, ticket 04 re-evaluation |
| [`CTC-ATTENTION.md`](CTC-ATTENTION.md) | The heads architecture, joint loss, the reproducible perturbation table, v2 results on the ai231 data, streaming commands |

## Model, grammar and decoding

| Doc | What it holds |
|---|---|
| [`VCM-CONTRACT.md`](VCM-CONTRACT.md) | The VCM interface contract (features, decoder I/O, licence provenance) |
| [`OPTIONB-GRAMMAR-CONTRACT.md`](OPTIONB-GRAMMAR-CONTRACT.md) | The 19-intent, 93-wording grammar |
| [`INCOMPLETE-GRAMMAR-REJECTION.md`](INCOMPLETE-GRAMMAR-REJECTION.md) | The incomplete-prefix margin gate and its calibration |
| [`VCM-DATASET-COMPATIBILITY.md`](VCM-DATASET-COMPATIBILITY.md) | Dataset schema compatibility |

## Streaming and wake word

| Doc | What it holds |
|---|---|
| [`STREAMING-CONTRACT.md`](STREAMING-CONTRACT.md) | Pipeline order, policies, gates, JSONL schema, config precedence |
| [`WAKEWORD-SLIDING.md`](WAKEWORD-SLIDING.md) | Wake word + sliding window, answer when the command ends |
| [`WAKEWORD-DATASET-CONTRACT.md`](WAKEWORD-DATASET-CONTRACT.md) | The wake-word dataset |

## Data generation and project history

[`adding-a-tts-backend.md`](adding-a-tts-backend.md),
[`MLOPS-PROJECTS.md`](MLOPS-PROJECTS.md) (dataset iterations and decisions), [`KAGGLE.md`](KAGGLE.md), [`BACKLOG.md`](BACKLOG.md),
`raw_requirements/` and `research/` (original planning notes, read-only).

## Archive (the earlier ~1M-parameter model and its design notes)

Everything in [`archive/`](archive/) describes the earlier model (`option-d-fil50-ambient-rir-135m`, a 1 MB QuartzNet student) or the process that produced it. It is kept for reference and is not maintained.

| Doc | What it holds |
|---|---|
| [`archive/README-EARLIER-MODEL.md`](archive/README-EARLIER-MODEL.md) | The main README's earlier-model part: project background, 2026-09-26 status table, TTS data generation, the toy VCM, Option B grammar, first wake word, VCMX serving, streaming inference, UI site |
| [`archive/PROCESS-OVERVIEW.md`](archive/PROCESS-OVERVIEW.md) (hub), [`PROCESS-DATA-GENERATION.md`](archive/PROCESS-DATA-GENERATION.md), [`PROCESS-VCM-MODEL.md`](archive/PROCESS-VCM-MODEL.md), [`PROCESS-WAKEWORD.md`](archive/PROCESS-WAKEWORD.md), [`PROCESS-STREAMING-SERVING.md`](archive/PROCESS-STREAMING-SERVING.md) | The process narrative, as of late September |
| [`archive/QUARTZNET-STUDENT.md`](archive/QUARTZNET-STUDENT.md), [`QUARTZNET-OPERATING-POINT.md`](archive/QUARTZNET-OPERATING-POINT.md) | The 1 MB QuartzNet model, its threshold and margin |
| [`archive/CASCADE-SOAK-TEST.md`](archive/CASCADE-SOAK-TEST.md) | The earlier wake-word -> VCM cascade soak test (the current soak is [`SOAK-TEST.md`](SOAK-TEST.md)) |
| [`archive/DENSE-SCORING-DECISION.md`](archive/DENSE-SCORING-DECISION.md) | Per-character scoring trial (not adopted) |
| [`archive/ACCENT-BALANCE-FIL50-COMMANDS.md`](archive/ACCENT-BALANCE-FIL50-COMMANDS.md), [`20260925_suggestions.md`](archive/20260925_suggestions.md) | The earlier fil50 rebalance and the 2026-09-25 re-architecture notes |

Dates matter in the remaining docs too: the model-history docs (`MLOPS-PROJECTS.md`, `KAGGLE.md`, `BACKLOG.md`) are as of late September; the current model is the one in `CURRENT-MODEL.md`.
