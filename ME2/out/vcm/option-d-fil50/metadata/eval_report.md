# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-fil50/checkpoints/checkpoint.pt` (preset=optiond, epoch=47, val_loss=0.30504226004960305)
Manifest: `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv`
Device: `cuda:4`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.075** (val target_accept_rate=0.965, val false_accept_rate=0.009)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.813 | 0.000 | 0.813 |
| -0.02 | 0.874 | 0.001 | 0.872 |
| -0.03 | 0.908 | 0.001 | 0.907 |
| -0.05 | 0.946 | 0.007 | 0.939 |
| -0.075 | 0.965 | 0.009 | 0.956 |
| -0.1 | 0.976 | 0.023 | 0.953 |
| -0.15 | 0.986 | 0.150 | 0.835 |
| -0.2 | 0.991 | 0.283 | 0.707 |
| -0.25 | 0.993 | 0.379 | 0.614 |
| -0.3 | 0.994 | 0.402 | 0.592 |
| -0.35 | 0.994 | 0.413 | 0.581 |
| -0.4 | 0.995 | 0.415 | 0.581 |
| -0.5 | 0.996 | 0.416 | 0.580 |
| -0.75 | 0.996 | 0.416 | 0.580 |
| -1.0 | 0.997 | 0.416 | 0.581 |
| -1.5 | 0.998 | 0.416 | 0.582 |
| -2.0 | 0.998 | 0.416 | 0.582 |
| -5.0 | 0.998 | 0.416 | 0.582 |

### Test-split results at the chosen threshold

- `target_commands`: 3494 clips, 3420 accepted (0.979 accept rate), 3420 exact-intent-correct (0.979 exact accuracy)
- `babble` false-accept rate: 2/255 (0.008) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 0/324 (0.000)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat:** the test-split Filipino group is a single held-out speaker (`s100`, 180 clips), so a gap here confounds accent with speaker identity. 13 of the 16 Filipino speakers (2,146 clips) are in the train split, so this checkpoint is not accent-naive.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 1747 | 0.978 | 0.978 | [0.970, 0.984] |
| foreign_reference | 1618 | 0.986 | 0.986 | [0.979, 0.991] |

- exact_accuracy_gap_foreign_minus_filipino: 0.008; n_unclassified: 129

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 1053/1053 intent-correct clips also had every slot value correct (1.000), out of 1068 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 172 | 172 | 1.000 |
| COLOR | 180 | 180 | 1.000 |
| DEGREES | 174 | 174 | 1.000 |
| DURATION | 172 | 172 | 1.000 |
| PERCENT | 176 | 176 | 1.000 |
| TASK | 179 | 179 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=323, REJECTED=7 |
| BRIGHTNESS | BRIGHTNESS=266, REJECTED=5 |
| CALL | CALL=92, REJECTED=2 |
| COLOR | COLOR=339, REJECTED=2 |
| CREATE_REMINDER | CREATE_REMINDER=311, REJECTED=6 |
| LIGHT_OFF | LIGHT_OFF=119, REJECTED=4 |
| LIGHT_ON | LIGHT_ON=115, REJECTED=1 |
| LIST_REMINDERS | LIST_REMINDERS=103, REJECTED=3 |
| MESSAGE | MESSAGE=111, REJECTED=2 |
| NEXT | NEXT=218, REJECTED=2 |
| PAUSE | PAUSE=99, REJECTED=1 |
| PLAY_MUSIC | PLAY_MUSIC=114, REJECTED=3 |
| STOP | REJECTED=2, STOP=132 |
| TEMPERATURE | REJECTED=7, TEMPERATURE=331 |
| TIME | REJECTED=9, TIME=98 |
| TIMER | REJECTED=11, TIMER=310 |
| VOLUME_DOWN | REJECTED=3, VOLUME_DOWN=104 |
| VOLUME_UP | REJECTED=1, VOLUME_UP=131 |
| WEATHER | REJECTED=3, WEATHER=104 |

## Slot-eval-set (Task 07) results

Skipped: out/vcm/option-d-fil50/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

