# Wakeword DS-CNN eval report

Checkpoint: `/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/wakeword-sesame/checkpoints/checkpoint.pt` (preset=default, epoch=16).

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.984 | 0.994 | 0.989 | 311 |
| `_unknown_` | 0.992 | 0.980 | 0.986 | 249 |
| `_silence_` | 1.000 | 1.000 | 1.000 | 56 |

## `_wakeword_` recall by voice accent

| group | recall | correct | total |
|---|---|---|---|
| filipino | 0.993 | 150 | 151 |
| non_filipino | 0.994 | 159 | 160 |

Checkpoint trained on out/conversions/v2/wakeword/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) additively mixed into adversaries_noisy/positives_converted_noisy. Per docs/WAKEWORD-DATASET-CONTRACT.md section 7, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.
