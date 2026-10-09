# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt` (preset=optiond, epoch=88, val_loss=0.43209828355519403)
Manifest: `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv`
Device: `cuda:4`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.1** (val target_accept_rate=0.947, val false_accept_rate=0.018)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.639 | 0.001 | 0.638 |
| -0.02 | 0.744 | 0.001 | 0.743 |
| -0.03 | 0.798 | 0.001 | 0.796 |
| -0.05 | 0.865 | 0.001 | 0.864 |
| -0.075 | 0.918 | 0.005 | 0.913 |
| -0.1 | 0.947 | 0.018 | 0.930 |
| -0.15 | 0.975 | 0.108 | 0.867 |
| -0.2 | 0.986 | 0.248 | 0.738 |
| -0.25 | 0.989 | 0.341 | 0.648 |
| -0.3 | 0.990 | 0.386 | 0.604 |
| -0.35 | 0.992 | 0.401 | 0.591 |
| -0.4 | 0.993 | 0.407 | 0.587 |
| -0.5 | 0.994 | 0.409 | 0.585 |
| -0.75 | 0.995 | 0.409 | 0.586 |
| -1.0 | 0.996 | 0.409 | 0.587 |
| -1.5 | 0.996 | 0.409 | 0.587 |
| -2.0 | 0.996 | 0.409 | 0.587 |
| -5.0 | 0.996 | 0.409 | 0.587 |

### Test-split results at the chosen threshold

- `target_commands`: 3494 clips, 3362 accepted (0.962 accept rate), 3341 exact-intent-correct (0.956 exact accuracy)
- `babble` false-accept rate: 3/255 (0.012) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 5/324 (0.015)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat:** the test-split Filipino group is a single held-out speaker (`s100`, 180 clips), so a gap here confounds accent with speaker identity. 13 of the 16 Filipino speakers (2,146 clips) are in the train split, so this checkpoint is not accent-naive.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 1747 | 0.940 | 0.929 | [0.916, 0.940] |
| foreign_reference | 1618 | 0.989 | 0.989 | [0.982, 0.993] |

- exact_accuracy_gap_foreign_minus_filipino: 0.060; n_unclassified: 129

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
| ALARM | ALARM=319, REJECTED=11 |
| BRIGHTNESS | BRIGHTNESS=254, REJECTED=17 |
| CALL | CALL=91, PAUSE=2, STOP=1 |
| COLOR | COLOR=332, REJECTED=9 |
| CREATE_REMINDER | CREATE_REMINDER=302, REJECTED=15 |
| LIGHT_OFF | LIGHT_OFF=103, LIGHT_ON=13, REJECTED=7 |
| LIGHT_ON | LIGHT_ON=115, REJECTED=1 |
| LIST_REMINDERS | LIST_REMINDERS=103, REJECTED=3 |
| MESSAGE | MESSAGE=110, REJECTED=3 |
| NEXT | LIGHT_ON=1, NEXT=210, REJECTED=6, STOP=3 |
| PAUSE | PAUSE=95, REJECTED=5 |
| PLAY_MUSIC | PLAY_MUSIC=111, REJECTED=6 |
| STOP | REJECTED=1, STOP=133 |
| TEMPERATURE | REJECTED=13, TEMPERATURE=325 |
| TIME | REJECTED=9, TIME=98 |
| TIMER | REJECTED=17, TIMER=304 |
| VOLUME_DOWN | REJECTED=5, VOLUME_DOWN=102 |
| VOLUME_UP | REJECTED=2, VOLUME_DOWN=1, VOLUME_UP=129 |
| WEATHER | REJECTED=2, WEATHER=105 |

## Slot-eval-set (Task 07) results

Skipped: /mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/vcm/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

