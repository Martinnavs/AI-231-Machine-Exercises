# Wakeword DS-CNN eval report

Checkpoint: `out/wakeword-sesame-ambient-rir-45m/checkpoints/checkpoint.pt` (preset=default, epoch=16).

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.998 | 0.993 | 0.995 | 408 |
| `_unknown_` | 0.994 | 0.994 | 0.994 | 497 |
| `_silence_` | 0.966 | 1.000 | 0.982 | 56 |

## `_wakeword_` recall by voice accent

| group | recall | correct | total |
|---|---|---|---|
| filipino | 0.990 | 200 | 202 |
| non_filipino | 0.995 | 205 | 206 |

Checkpoint trained on out/conversions/v2/wakeword-sesame/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) additively mixed into adversaries_noisy/positives_converted_noisy. Per docs/WAKEWORD-DATASET-CONTRACT.md section 7, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.
