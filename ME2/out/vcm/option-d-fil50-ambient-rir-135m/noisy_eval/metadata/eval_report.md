# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-fil50-ambient-rir-135m/checkpoints/checkpoint.pt` (preset=optiond, epoch=66, val_loss=0.2520240666445887)
Manifest: `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv`
Device: `cuda:0`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.1** (val target_accept_rate=0.981, val false_accept_rate=0.018)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.867 | 0.001 | 0.866 |
| -0.02 | 0.911 | 0.001 | 0.910 |
| -0.03 | 0.933 | 0.001 | 0.932 |
| -0.05 | 0.958 | 0.004 | 0.954 |
| -0.075 | 0.971 | 0.009 | 0.961 |
| -0.1 | 0.981 | 0.018 | 0.963 |
| -0.15 | 0.988 | 0.106 | 0.882 |
| -0.2 | 0.990 | 0.241 | 0.749 |
| -0.25 | 0.992 | 0.341 | 0.651 |
| -0.3 | 0.993 | 0.400 | 0.593 |
| -0.35 | 0.994 | 0.423 | 0.571 |
| -0.4 | 0.994 | 0.424 | 0.570 |
| -0.5 | 0.995 | 0.427 | 0.568 |
| -0.75 | 0.997 | 0.427 | 0.570 |
| -1.0 | 0.997 | 0.427 | 0.570 |
| -1.5 | 0.998 | 0.427 | 0.571 |
| -2.0 | 0.998 | 0.427 | 0.571 |
| -5.0 | 0.998 | 0.427 | 0.571 |

### Test-split results at the chosen threshold

- `target_commands`: 3494 clips, 3450 accepted (0.987 accept rate), 3448 exact-intent-correct (0.987 exact accuracy)
- `babble` false-accept rate: 3/255 (0.012) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 1/324 (0.003)
- `acoustic_ghost` false-accept rate: 0/0 (n/a) -- Iteration 3 open-vocabulary hard negatives: real podcast audio that phonetically collides with a short command, trained with an empty transcript so the model must reject them
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat:** the test-split Filipino group is a single held-out speaker (`s100`, 180 clips), so a gap here confounds accent with speaker identity. 13 of the 16 Filipino speakers (2,146 clips) are in the train split, so this checkpoint is not accent-naive.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 1747 | 0.986 | 0.985 | [0.978, 0.989] |
| foreign_reference | 1618 | 0.992 | 0.992 | [0.986, 0.995] |

- exact_accuracy_gap_foreign_minus_filipino: 0.007; n_unclassified: 129

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 1057/1057 intent-correct clips also had every slot value correct (1.000), out of 1068 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 172 | 172 | 1.000 |
| COLOR | 180 | 180 | 1.000 |
| DEGREES | 178 | 178 | 1.000 |
| DURATION | 172 | 172 | 1.000 |
| PERCENT | 176 | 176 | 1.000 |
| TASK | 179 | 179 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=327, REJECTED=3 |
| BRIGHTNESS | BRIGHTNESS=267, REJECTED=4 |
| CALL | CALL=92, REJECTED=2 |
| COLOR | COLOR=341 |
| CREATE_REMINDER | CREATE_REMINDER=312, REJECTED=5 |
| LIGHT_OFF | LIGHT_OFF=119, LIGHT_ON=2, REJECTED=2 |
| LIGHT_ON | LIGHT_ON=116 |
| LIST_REMINDERS | LIST_REMINDERS=105, REJECTED=1 |
| MESSAGE | MESSAGE=112, REJECTED=1 |
| NEXT | NEXT=218, REJECTED=2 |
| PAUSE | PAUSE=100 |
| PLAY_MUSIC | PLAY_MUSIC=115, REJECTED=2 |
| STOP | REJECTED=2, STOP=132 |
| TEMPERATURE | REJECTED=2, TEMPERATURE=336 |
| TIME | REJECTED=4, TIME=103 |
| TIMER | REJECTED=11, TIMER=310 |
| VOLUME_DOWN | REJECTED=2, VOLUME_DOWN=105 |
| VOLUME_UP | VOLUME_UP=132 |
| WEATHER | REJECTED=1, WEATHER=106 |

## Noisy/reverb eval gate (fixed seed 0)

> **What this is:** a separate eval pass over a DETERMINISTIC perturbed copy of the val and test splits -- rir+additive_noise: every row gets one RIR (RT60 U[0.1,0.5]s randomized room from a fixed 200-entry pool) then one ESC-50 noise clip at SNR U[5,25] dB, both always applied, in that order. The perturbation is fixed by (`manifest, seed 0`): identical inputs give a bit-identical perturbed signal on every run, so these numbers are comparable checkpoint-to-checkpoint. SpecAugment and time-stretch are deliberately NOT part of this pass (training regularizers, not real acoustic conditions). **Threshold rule:** scored at the clean-val chosen operating threshold per grammar (choose_operating_threshold on the CLEAN val split); never re-swept on noisy val.

### OPTIONB_GRAMMAR (clean-val chosen threshold -0.1)

### Clean vs. noisy (test split, same threshold)

| metric | clean | noisy | delta (noisy - clean) |
|---|---|---|---|
| target accept_rate | 0.987 | 0.961 | -0.026 |
| target exact_accuracy | 0.987 | 0.957 | -0.029 |
| babble false-accept rate | 0.012 | 0.039 | +0.027 |
| silence false-accept rate | 0.003 | 0.019 | +0.015 |
| acoustic_ghost false-accept rate | n/a | n/a | n/a |

- Noisy test split: 3359/3494 targets accepted (0.961 accept rate), 3345 exact-intent-correct (0.957 exact accuracy)
- Noisy false-accept rates: babble 10/255 (0.039), silence 6/324 (0.019)
- acoustic_ghost false-accept rate: 0/0 (n/a)
- Noisy val split at the same threshold (diagnostic -- does the clean-chosen operating point still behave on noisy val?): accept_rate=0.926, babble_FAR=0.073, silence_FAR=0.012

## Slot-eval-set (Task 07) results

Skipped: /mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-iteration3/ME2/out/vcm/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

