# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-fil50-ambient-rir-135m/checkpoints/checkpoint.pt` (preset=optiond, epoch=66, val_loss=0.2520240666445887)
Manifest: `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv`
Device: `cpu`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-1.1** (val target_accept_rate=0.984, val false_accept_rate=0.001)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.05 | 0.813 | 0.000 | 0.813 |
| -0.1 | 0.863 | 0.000 | 0.863 |
| -0.25 | 0.924 | 0.001 | 0.922 |
| -0.5 | 0.958 | 0.001 | 0.956 |
| -0.75 | 0.976 | 0.001 | 0.975 |
| -0.8 | 0.978 | 0.001 | 0.976 |
| -0.9 | 0.980 | 0.001 | 0.979 |
| -1.0 | 0.983 | 0.001 | 0.981 |
| -1.1 | 0.984 | 0.001 | 0.983 |
| -1.2 | 0.985 | 0.003 | 0.982 |
| -1.3 | 0.986 | 0.005 | 0.980 |
| -1.4 | 0.986 | 0.007 | 0.979 |
| -1.5 | 0.986 | 0.007 | 0.979 |
| -1.6 | 0.987 | 0.007 | 0.980 |
| -1.8 | 0.988 | 0.007 | 0.982 |
| -2.0 | 0.989 | 0.014 | 0.976 |
| -2.5 | 0.991 | 0.037 | 0.954 |
| -3.0 | 0.991 | 0.079 | 0.913 |
| -5.0 | 0.994 | 0.211 | 0.783 |

### Test-split results at the chosen threshold

- `target_commands`: 3494 clips, 3445 accepted (0.986 accept rate), 3443 exact-intent-correct (0.985 exact accuracy)
- `babble` false-accept rate: 0/255 (0.000) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 0/324 (0.000)
- score mode: `per_char` (confidences and thresholds are on the per-character scale, not comparable to per-frame reports)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat (filipino_reference):** the test-split `filipino_reference` group spans 18 distinct speakers/voices (1747 clips). 17/18 of these also appear in the train split, so this checkpoint is not accent-naive for this group.
> **Caveat (foreign_reference):** the test-split `foreign_reference` group spans 9 distinct speakers/voices (1618 clips). None of these appear in the train split -- a genuinely held-out comparison for this group.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 1747 | 0.983 | 0.982 | [0.974, 0.987] |
| foreign_reference | 1618 | 0.993 | 0.993 | [0.988, 0.996] |

- exact_accuracy_gap_foreign_minus_filipino: 0.012; n_unclassified: 129

### Slot accuracy

> Intent-level accuracy above only checks `intent == label` -- it does not check whether a slot value (e.g. which hour an ALARM clip named) was extracted correctly. This section does: among slot-bearing Option B target clips whose predicted intent was already correct, what fraction also got every slot value right.

- 1062/1062 intent-correct clips also had every slot value correct (1.000), out of 1068 slot-bearing target clips (0 with unparseable ground truth)

| slot | n (intent-correct clips) | n correct | accuracy |
|---|---|---|---|
| ALARM_TIME | 172 | 172 | 1.000 |
| COLOR | 180 | 180 | 1.000 |
| DEGREES | 180 | 180 | 1.000 |
| DURATION | 172 | 172 | 1.000 |
| PERCENT | 179 | 179 | 1.000 |
| TASK | 179 | 179 | 1.000 |

### Per-intent accept/confusion counts (test split)

| true intent | predicted distribution |
|---|---|
| ALARM | ALARM=326, REJECTED=4 |
| BRIGHTNESS | BRIGHTNESS=270, REJECTED=1 |
| CALL | CALL=92, REJECTED=2 |
| COLOR | COLOR=340, REJECTED=1 |
| CREATE_REMINDER | CREATE_REMINDER=312, REJECTED=5 |
| LIGHT_OFF | LIGHT_OFF=119, LIGHT_ON=2, REJECTED=2 |
| LIGHT_ON | LIGHT_ON=115, REJECTED=1 |
| LIST_REMINDERS | LIST_REMINDERS=105, REJECTED=1 |
| MESSAGE | MESSAGE=111, REJECTED=2 |
| NEXT | NEXT=216, REJECTED=4 |
| PAUSE | PAUSE=96, REJECTED=4 |
| PLAY_MUSIC | PLAY_MUSIC=115, REJECTED=2 |
| STOP | REJECTED=2, STOP=132 |
| TEMPERATURE | TEMPERATURE=338 |
| TIME | REJECTED=4, TIME=103 |
| TIMER | REJECTED=10, TIMER=311 |
| VOLUME_DOWN | REJECTED=3, VOLUME_DOWN=104 |
| VOLUME_UP | VOLUME_UP=132 |
| WEATHER | REJECTED=1, WEATHER=106 |

## Slot-eval-set (Task 07) results

Skipped: /mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/vcm/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

