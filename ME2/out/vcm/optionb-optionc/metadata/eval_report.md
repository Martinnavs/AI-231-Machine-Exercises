# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/optionb-optionc/checkpoints/checkpoint.pt` (preset=optionc, epoch=39, val_loss=0.4336458110023202)
Manifest: `out/conversions/v2/optionb/manifest.csv`
Device: `cuda`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.1** (val target_accept_rate=0.938, val false_accept_rate=0.064)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.631 | 0.000 | 0.631 |
| -0.02 | 0.737 | 0.000 | 0.737 |
| -0.03 | 0.805 | 0.006 | 0.799 |
| -0.05 | 0.863 | 0.025 | 0.837 |
| -0.075 | 0.914 | 0.045 | 0.869 |
| -0.1 | 0.938 | 0.064 | 0.875 |
| -0.15 | 0.958 | 0.102 | 0.856 |
| -0.2 | 0.975 | 0.185 | 0.791 |
| -0.25 | 0.984 | 0.242 | 0.742 |
| -0.3 | 0.986 | 0.280 | 0.706 |
| -0.35 | 0.989 | 0.318 | 0.670 |
| -0.4 | 0.989 | 0.344 | 0.645 |
| -0.5 | 0.990 | 0.344 | 0.647 |
| -0.75 | 0.993 | 0.344 | 0.649 |
| -1.0 | 0.994 | 0.344 | 0.650 |
| -1.5 | 0.994 | 0.344 | 0.650 |
| -2.0 | 0.995 | 0.344 | 0.651 |
| -5.0 | 0.995 | 0.344 | 0.651 |

### Test-split results at the chosen threshold

- `target_commands`: 1766 clips, 1701 accepted (0.963 accept rate), 1696 exact-intent-correct (0.960 exact accuracy)
- `babble` false-accept rate: 0/56 (0.000) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 0/22 (0.000)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat:** the test-split Filipino group is a single held-out speaker (`s100`, 180 clips), so a gap here confounds accent with speaker identity. 13 of the 16 Filipino speakers (2,146 clips) are in the train split, so this checkpoint is not accent-naive.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 180 | 0.989 | 0.989 | [0.960, 0.997] |
| foreign_reference | 1586 | 0.960 | 0.957 | [0.946, 0.966] |

- exact_accuracy_gap_foreign_minus_filipino: -0.032; n_unclassified: 0

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 1021/1021 intent-correct clips also had every slot value correct (1.000), out of 1056 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 168 | 168 | 1.000 |
| COLOR | 162 | 162 | 1.000 |
| DEGREES | 172 | 172 | 1.000 |
| DURATION | 173 | 173 | 1.000 |
| PERCENT | 169 | 169 | 1.000 |
| TASK | 177 | 177 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=168, REJECTED=2 |
| BRIGHTNESS | BRIGHTNESS=169, REJECTED=11 |
| CALL | CALL=44, REJECTED=4 |
| COLOR | COLOR=162, REJECTED=8 |
| CREATE_REMINDER | CREATE_REMINDER=177, REJECTED=3 |
| LIGHT_OFF | LIGHT_OFF=50, LIGHT_ON=5, REJECTED=3 |
| LIGHT_ON | LIGHT_ON=54, REJECTED=2 |
| LIST_REMINDERS | LIST_REMINDERS=54, REJECTED=4 |
| MESSAGE | MESSAGE=58 |
| NEXT | NEXT=52 |
| PAUSE | PAUSE=50 |
| PLAY_MUSIC | PLAY_MUSIC=57, REJECTED=1 |
| STOP | REJECTED=2, STOP=56 |
| TEMPERATURE | REJECTED=8, TEMPERATURE=172 |
| TIME | REJECTED=8, TIME=46 |
| TIMER | REJECTED=3, TIMER=173 |
| VOLUME_DOWN | REJECTED=2, VOLUME_DOWN=50 |
| VOLUME_UP | REJECTED=2, VOLUME_UP=56 |
| WEATHER | REJECTED=2, WEATHER=48 |

## Slot-eval-set (Task 07) results

Skipped: out/vcm/optionb-optionc/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

