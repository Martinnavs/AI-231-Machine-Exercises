# Wakeword DS-CNN eval report

Checkpoint: `out/wakeword/checkpoints/checkpoint.pt` (preset=default, epoch=20).

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.999 | 0.929 | 0.963 | 1424 |
| `_unknown_` | 0.916 | 0.998 | 0.955 | 1099 |
| `_silence_` | 0.996 | 1.000 | 0.998 | 270 |

## `_wakeword_` recall by voice accent

| group | recall | correct | total |
|---|---|---|---|
| filipino | 0.864 | 615 | 712 |
| non_filipino | 0.994 | 708 | 712 |

Checkpoint trained on out/conversions/v2/wakeword/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) additively mixed into adversaries_noisy/positives_converted_noisy. Per docs/WAKEWORD-DATASET-CONTRACT.md section 7, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.
