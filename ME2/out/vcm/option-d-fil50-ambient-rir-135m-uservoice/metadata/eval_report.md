# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-fil50-ambient-rir-135m-uservoice/checkpoints/checkpoint.pt` (preset=optiond, epoch=57, val_loss=0.2522448843550209)
Manifest: `out/conversions/v2/optionb-v3-vcmx-fil50-ambient-uservoice/manifest.csv`
Device: `cuda:7`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.075** (val target_accept_rate=0.960, val false_accept_rate=0.011)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.833 | 0.001 | 0.831 |
| -0.02 | 0.881 | 0.001 | 0.880 |
| -0.03 | 0.910 | 0.001 | 0.909 |
| -0.05 | 0.943 | 0.005 | 0.938 |
| -0.075 | 0.960 | 0.011 | 0.949 |
| -0.1 | 0.972 | 0.030 | 0.942 |
| -0.15 | 0.984 | 0.129 | 0.856 |
| -0.2 | 0.990 | 0.302 | 0.688 |
| -0.25 | 0.992 | 0.375 | 0.617 |
| -0.3 | 0.994 | 0.402 | 0.592 |
| -0.35 | 0.995 | 0.411 | 0.584 |
| -0.4 | 0.995 | 0.413 | 0.582 |
| -0.5 | 0.995 | 0.413 | 0.582 |
| -0.75 | 0.996 | 0.413 | 0.583 |
| -1.0 | 0.997 | 0.413 | 0.584 |
| -1.5 | 0.997 | 0.413 | 0.584 |
| -2.0 | 0.997 | 0.413 | 0.584 |
| -5.0 | 0.997 | 0.413 | 0.584 |

### Test-split results at the chosen threshold

- `target_commands`: 5527 clips, 5352 accepted (0.968 accept rate), 5348 exact-intent-correct (0.968 exact accuracy)
- `babble` false-accept rate: 1/255 (0.004) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 1/324 (0.003)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat (filipino_reference):** the test-split `filipino_reference` group spans 18 distinct speakers/voices (1747 clips). 17/18 of these also appear in the train split, so this checkpoint is not accent-naive for this group.
> **Caveat (foreign_reference):** the test-split `foreign_reference` group spans 9 distinct speakers/voices (1618 clips). None of these appear in the train split -- a genuinely held-out comparison for this group.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 1747 | 0.979 | 0.978 | [0.970, 0.984] |
| foreign_reference | 1618 | 0.988 | 0.988 | [0.982, 0.992] |

- exact_accuracy_gap_foreign_minus_filipino: 0.011; n_unclassified: 2162

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 1054/1054 intent-correct clips also had every slot value correct (1.000), out of 1068 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 172 | 172 | 1.000 |
| COLOR | 180 | 180 | 1.000 |
| DEGREES | 175 | 175 | 1.000 |
| DURATION | 170 | 170 | 1.000 |
| PERCENT | 177 | 177 | 1.000 |
| TASK | 180 | 180 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=504, REJECTED=21 |
| BRIGHTNESS | BRIGHTNESS=380, REJECTED=7 |
| CALL | CALL=143, REJECTED=10, TIME=1 |
| COLOR | COLOR=552, REJECTED=1 |
| CREATE_REMINDER | CREATE_REMINDER=491, LIST_REMINDERS=1, REJECTED=14 |
| LIGHT_OFF | LIGHT_OFF=185, REJECTED=5 |
| LIGHT_ON | LIGHT_ON=183, REJECTED=5 |
| LIST_REMINDERS | LIST_REMINDERS=170, REJECTED=8 |
| MESSAGE | MESSAGE=177, REJECTED=6 |
| NEXT | NEXT=325, REJECTED=8 |
| PAUSE | PAUSE=150, REJECTED=7 |
| PLAY_MUSIC | PLAY_MUSIC=185, REJECTED=4 |
| STOP | REJECTED=2, STOP=224 |
| TEMPERATURE | REJECTED=19, TEMPERATURE=529 |
| TIME | REJECTED=8, TIME=149 |
| TIMER | REJECTED=35, TIMER=466 |
| VOLUME_DOWN | REJECTED=6, VOLUME_DOWN=160 |
| VOLUME_UP | REJECTED=5, VOLUME_DOWN=2, VOLUME_UP=204 |
| WEATHER | REJECTED=4, WEATHER=171 |

## Slot-eval-set (Task 07) results

Skipped: /mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/vcm/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

