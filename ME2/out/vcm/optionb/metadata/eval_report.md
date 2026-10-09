# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/optionb/checkpoints/checkpoint.pt` (preset=default, epoch=68, val_loss=0.5248446956863554)
Manifest: `out/conversions/v2/optionb/manifest.csv`
Device: `cuda`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.1** (val target_accept_rate=0.904, val false_accept_rate=0.089)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.517 | 0.000 | 0.517 |
| -0.02 | 0.639 | 0.006 | 0.633 |
| -0.03 | 0.724 | 0.025 | 0.699 |
| -0.05 | 0.816 | 0.057 | 0.759 |
| -0.075 | 0.872 | 0.076 | 0.796 |
| -0.1 | 0.904 | 0.089 | 0.815 |
| -0.15 | 0.947 | 0.140 | 0.807 |
| -0.2 | 0.967 | 0.185 | 0.783 |
| -0.25 | 0.979 | 0.287 | 0.692 |
| -0.3 | 0.983 | 0.325 | 0.658 |
| -0.35 | 0.984 | 0.338 | 0.646 |
| -0.4 | 0.985 | 0.350 | 0.635 |
| -0.5 | 0.986 | 0.357 | 0.629 |
| -0.75 | 0.989 | 0.363 | 0.626 |
| -1.0 | 0.990 | 0.363 | 0.627 |
| -1.5 | 0.992 | 0.363 | 0.629 |
| -2.0 | 0.992 | 0.363 | 0.629 |
| -5.0 | 0.992 | 0.363 | 0.629 |

### Test-split results at the chosen threshold

- `target_commands`: 1766 clips, 1680 accepted (0.951 accept rate), 1675 exact-intent-correct (0.948 exact accuracy)
- `babble` false-accept rate: 4/56 (0.071) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 0/22 (0.000)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat:** the test-split Filipino group is a single held-out speaker (`s100`, 180 clips), so a gap here confounds accent with speaker identity. 13 of the 16 Filipino speakers (2,146 clips) are in the train split, so this checkpoint is not accent-naive.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 180 | 0.967 | 0.967 | [0.929, 0.985] |
| foreign_reference | 1586 | 0.950 | 0.946 | [0.934, 0.956] |

- exact_accuracy_gap_foreign_minus_filipino: -0.020; n_unclassified: 0

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 999/999 intent-correct clips also had every slot value correct (1.000), out of 1056 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 168 | 168 | 1.000 |
| COLOR | 163 | 163 | 1.000 |
| DEGREES | 156 | 156 | 1.000 |
| DURATION | 172 | 172 | 1.000 |
| PERCENT | 165 | 165 | 1.000 |
| TASK | 175 | 175 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=168, REJECTED=2 |
| BRIGHTNESS | BRIGHTNESS=165, REJECTED=15 |
| CALL | CALL=43, REJECTED=5 |
| COLOR | COLOR=163, REJECTED=7 |
| CREATE_REMINDER | CREATE_REMINDER=175, REJECTED=5 |
| LIGHT_OFF | LIGHT_OFF=51, LIGHT_ON=4, REJECTED=3 |
| LIGHT_ON | LIGHT_OFF=1, LIGHT_ON=52, REJECTED=3 |
| LIST_REMINDERS | LIST_REMINDERS=54, REJECTED=4 |
| MESSAGE | MESSAGE=58 |
| NEXT | NEXT=51, REJECTED=1 |
| PAUSE | PAUSE=49, REJECTED=1 |
| PLAY_MUSIC | PLAY_MUSIC=57, REJECTED=1 |
| STOP | STOP=58 |
| TEMPERATURE | REJECTED=24, TEMPERATURE=156 |
| TIME | REJECTED=7, TIME=47 |
| TIMER | REJECTED=4, TIMER=172 |
| VOLUME_DOWN | VOLUME_DOWN=52 |
| VOLUME_UP | REJECTED=2, VOLUME_UP=56 |
| WEATHER | REJECTED=2, WEATHER=48 |

## Slot-eval-set (Task 07) results

Skipped: out/vcm/optionb/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

