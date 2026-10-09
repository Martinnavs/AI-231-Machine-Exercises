# Wakeword DS-CNN eval report

Checkpoint: `out/wakeword-fil50/checkpoints/checkpoint.pt` (preset=default, epoch=29).

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.999 | 0.995 | 0.997 | 1424 |
| `_unknown_` | 0.994 | 0.998 | 0.996 | 1099 |
| `_silence_` | 1.000 | 1.000 | 1.000 | 270 |

## `_wakeword_` recall by voice accent

| group | recall | correct | total |
|---|---|---|---|
| filipino | 0.993 | 707 | 712 |
| non_filipino | 0.997 | 710 | 712 |

Checkpoint trained on out/conversions/v2/wakeword/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) additively mixed into adversaries_noisy/positives_converted_noisy. Per docs/WAKEWORD-DATASET-CONTRACT.md section 7, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.
