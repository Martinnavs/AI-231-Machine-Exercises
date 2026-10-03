# Documentation index

Start with [`CURRENT-MODEL.md`](CURRENT-MODEL.md): the current model, how to run it, what it scores and how far to trust that.
Then [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) for what can and cannot be reproduced from public data today.

## Current model and experiments (2026-09/10, branch `optionb-ctc-attention`)

| Doc | What it holds |
|---|---|
| [`CURRENT-MODEL.md`](CURRENT-MODEL.md) | The hybrid (wide CTC + XL heads): decision rule, quick start, numbers, data, training, evaluation, code map, open items |
| [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) | Verified check of the checked-in model; retraining; the part of the data recipe that is not public |
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
| [`DENSE-SCORING-DECISION.md`](DENSE-SCORING-DECISION.md) | Per-character scoring trial (not adopted) |
| [`QUARTZNET-STUDENT.md`](QUARTZNET-STUDENT.md), [`QUARTZNET-OPERATING-POINT.md`](QUARTZNET-OPERATING-POINT.md), [`CASCADE-SOAK-TEST.md`](CASCADE-SOAK-TEST.md) | The 1 MB QuartzNet model, its threshold and margin, and the wake-word -> VCM soak test |
| [`VCM-DATASET-COMPATIBILITY.md`](VCM-DATASET-COMPATIBILITY.md) | Dataset schema compatibility |

## Streaming and wake word

| Doc | What it holds |
|---|---|
| [`STREAMING-CONTRACT.md`](STREAMING-CONTRACT.md) | Pipeline order, policies, gates, JSONL schema, config precedence |
| [`WAKEWORD-SLIDING.md`](WAKEWORD-SLIDING.md) | Wake word + sliding window, answer when the command ends |
| [`WAKEWORD-DATASET-CONTRACT.md`](WAKEWORD-DATASET-CONTRACT.md) | The wake-word dataset |

## Process narrative (earlier ~1M-parameter model)

[`PROCESS-OVERVIEW.md`](PROCESS-OVERVIEW.md) (hub), [`PROCESS-DATA-GENERATION.md`](PROCESS-DATA-GENERATION.md),
[`PROCESS-VCM-MODEL.md`](PROCESS-VCM-MODEL.md), [`PROCESS-WAKEWORD.md`](PROCESS-WAKEWORD.md),
[`PROCESS-STREAMING-SERVING.md`](PROCESS-STREAMING-SERVING.md).

## Data generation and project history

[`ACCENT-BALANCE-FIL50-COMMANDS.md`](ACCENT-BALANCE-FIL50-COMMANDS.md) (the earlier fil50 rebalance), [`adding-a-tts-backend.md`](adding-a-tts-backend.md),
[`MLOPS-PROJECTS.md`](MLOPS-PROJECTS.md) (dataset iterations and decisions), [`KAGGLE.md`](KAGGLE.md), [`BACKLOG.md`](BACKLOG.md),
[`20260925_suggestions.md`](20260925_suggestions.md), `raw_requirements/` and `research/` (original planning notes, read-only).

Dates matter here: the PROCESS docs and the README status table describe the model as of late September; the current model is the one in `CURRENT-MODEL.md`.
