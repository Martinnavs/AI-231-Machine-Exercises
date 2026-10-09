# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt` (preset=optiond, epoch=88, val_loss=0.43209828355519403)
Manifest: `out/conversions/v2/optionb-v3-vcmx/manifest.csv`
Device: `cuda:7`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.1** (val target_accept_rate=0.964, val false_accept_rate=0.018)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.757 | 0.001 | 0.756 |
| -0.02 | 0.830 | 0.001 | 0.828 |
| -0.03 | 0.866 | 0.001 | 0.865 |
| -0.05 | 0.913 | 0.001 | 0.912 |
| -0.075 | 0.948 | 0.005 | 0.942 |
| -0.1 | 0.964 | 0.018 | 0.947 |
| -0.15 | 0.980 | 0.108 | 0.872 |
| -0.2 | 0.987 | 0.248 | 0.739 |
| -0.25 | 0.991 | 0.341 | 0.649 |
| -0.3 | 0.991 | 0.386 | 0.605 |
| -0.35 | 0.993 | 0.401 | 0.591 |
| -0.4 | 0.995 | 0.407 | 0.588 |
| -0.5 | 0.996 | 0.409 | 0.587 |
| -0.75 | 0.997 | 0.409 | 0.588 |
| -1.0 | 0.998 | 0.409 | 0.588 |
| -1.5 | 0.998 | 0.409 | 0.588 |
| -2.0 | 0.998 | 0.409 | 0.588 |
| -5.0 | 0.998 | 0.409 | 0.588 |

### Test-split results at the chosen threshold

- `target_commands`: 1927 clips, 1889 accepted (0.980 accept rate), 1887 exact-intent-correct (0.979 exact accuracy)
- `babble` false-accept rate: 3/255 (0.012) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 5/324 (0.015)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat:** the test-split Filipino group is a single held-out speaker (`s100`, 180 clips), so a gap here confounds accent with speaker identity. 13 of the 16 Filipino speakers (2,146 clips) are in the train split, so this checkpoint is not accent-naive.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 180 | 0.944 | 0.939 | [0.894, 0.966] |
| foreign_reference | 1618 | 0.989 | 0.989 | [0.982, 0.993] |

- exact_accuracy_gap_foreign_minus_filipino: 0.050; n_unclassified: 129

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 1057/1057 intent-correct clips also had every slot value correct (1.000), out of 1068 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 172 | 172 | 1.000 |
| COLOR | 180 | 180 | 1.000 |
| DEGREES | 180 | 180 | 1.000 |
| DURATION | 172 | 172 | 1.000 |
| PERCENT | 176 | 176 | 1.000 |
| TASK | 177 | 177 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=176, REJECTED=6 |
| BRIGHTNESS | BRIGHTNESS=176, REJECTED=4 |
| CALL | CALL=53, STOP=1 |
| COLOR | COLOR=180 |
| CREATE_REMINDER | CREATE_REMINDER=177, REJECTED=3 |
| LIGHT_OFF | LIGHT_OFF=57, LIGHT_ON=1, REJECTED=2 |
| LIGHT_ON | LIGHT_ON=57 |
| LIST_REMINDERS | LIST_REMINDERS=55, REJECTED=3 |
| MESSAGE | MESSAGE=58 |
| NEXT | NEXT=119, REJECTED=1 |
| PAUSE | PAUSE=60, REJECTED=1 |
| PLAY_MUSIC | PLAY_MUSIC=65, REJECTED=2 |
| STOP | STOP=65 |
| TEMPERATURE | TEMPERATURE=180 |
| TIME | REJECTED=7, TIME=53 |
| TIMER | REJECTED=4, TIMER=172 |
| VOLUME_DOWN | REJECTED=3, VOLUME_DOWN=61 |
| VOLUME_UP | VOLUME_UP=71 |
| WEATHER | REJECTED=2, WEATHER=52 |

## Slot-eval-set (Task 07) results

Skipped: out/vcm/option-d-dataset-v2/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

