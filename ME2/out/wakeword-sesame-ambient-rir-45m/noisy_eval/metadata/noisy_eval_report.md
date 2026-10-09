# Wakeword DS-CNN noisy/reverb eval report

Checkpoint: `out/wakeword-sesame-ambient-rir-45m/checkpoints/checkpoint.pt` (preset=default, epoch=16).

## Noisy/reverb eval gate (fixed seed 0)

Perturbation: rir+additive_noise: every row gets one RIR (RT60 U[0.1,0.5]s randomized room from a fixed 200-entry pool) then one noise clip from the run's --noise-root at SNR U[5,25] dB, both always applied, in that order. The perturbation is fixed by (`manifest, seed 0`): identical inputs give a bit-identical perturbed signal on every run, against any checkpoint.

**Threshold rule:** scored through the same argmax per-class path as the clean eval (wakeword.train.per_class_metrics / wakeword_accent_recall); the DS-CNN has no operating threshold to sweep, so the vcm.noisy_eval rule (clean-val chosen threshold, never re-swept on noisy val) holds trivially — nothing is re-fit on the noisy split.

### val_split (961 rows)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.988 | 0.975 | 0.982 | 408 |
| `_unknown_` | 0.976 | 0.972 | 0.974 | 497 |
| `_silence_` | 0.857 | 0.964 | 0.908 | 56 |

`_wakeword_` recall by voice accent:

| group | recall | correct | total |
|---|---|---|---|
| filipino | 0.980 | 198 | 202 |
| non_filipino | 0.971 | 200 | 206 |

### test_split (487 rows)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.975 | 0.952 | 0.964 | 208 |
| `_unknown_` | 0.960 | 0.960 | 0.960 | 251 |
| `_silence_` | 0.848 | 1.000 | 0.918 | 28 |

`_wakeword_` recall by voice accent:

| group | recall | correct | total |
|---|---|---|---|
| filipino | 0.942 | 98 | 104 |
| non_filipino | 0.962 | 100 | 104 |

Checkpoint trained on out/conversions/v2/wakeword-sesame/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) additively mixed into adversaries_noisy/positives_converted_noisy. Per docs/WAKEWORD-DATASET-CONTRACT.md section 7, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.
