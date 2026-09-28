# VCM toy CTC model -- evaluation report

License: Checkpoint trained on out/conversions/v2/test_set/, which includes background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on redistribution.

Checkpoint: `out/vcm/option-d-fil50-ambient-rir-135m/checkpoints/checkpoint.pt` (preset=optiond, epoch=66, val_loss=0.2520240666445887)
Manifest: `out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv`
Device: `cuda:3`  Beam width: 50

Threshold operating points below are chosen SEPARATELY per grammar by sweeping the candidate grid on the manifest's `val` split (never `test`) and maximizing Youden's J (target_accept_rate - false_accept_rate on babble+silence); the full sweep table is included so the choice is auditable.

## OPTIONB_GRAMMAR results

Chosen operating threshold (val sweep): **-0.1** (val target_accept_rate=0.974, val false_accept_rate=0.018)

### Val-split threshold sweep

| threshold | val target_accept_rate | val false_accept_rate | Youden J |
|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 |
| -0.01 | 0.831 | 0.001 | 0.829 |
| -0.02 | 0.883 | 0.001 | 0.881 |
| -0.03 | 0.913 | 0.001 | 0.911 |
| -0.05 | 0.943 | 0.004 | 0.939 |
| -0.075 | 0.961 | 0.009 | 0.952 |
| -0.1 | 0.974 | 0.018 | 0.957 |
| -0.15 | 0.985 | 0.106 | 0.880 |
| -0.2 | 0.990 | 0.241 | 0.749 |
| -0.25 | 0.992 | 0.341 | 0.651 |
| -0.3 | 0.993 | 0.400 | 0.594 |
| -0.35 | 0.994 | 0.423 | 0.572 |
| -0.4 | 0.995 | 0.424 | 0.571 |
| -0.5 | 0.995 | 0.427 | 0.568 |
| -0.75 | 0.997 | 0.427 | 0.570 |
| -1.0 | 0.997 | 0.427 | 0.570 |
| -1.5 | 0.998 | 0.427 | 0.571 |
| -2.0 | 0.998 | 0.427 | 0.571 |
| -5.0 | 0.998 | 0.427 | 0.571 |

### Test-split results at the chosen threshold

- `target_commands`: 5527 clips, 5411 accepted (0.979 accept rate), 5407 exact-intent-correct (0.978 exact accuracy)
- `babble` false-accept rate: 3/255 (0.012) -- includes filipino_speech_corpus rows per decision (B)
- `silence` false-accept rate: 1/324 (0.003)
- These false-accept rates hold **at the chosen operating threshold only** -- see the val-split sweep table above for how false-accept rate degrades at looser (more permissive) thresholds; it is not an unconditional property of the model.

### Speaker-group breakdown

> **Caveat (filipino_reference):** the test-split `filipino_reference` group spans 18 distinct speakers/voices (1747 clips). 17/18 of these also appear in the train split, so this checkpoint is not accent-naive for this group.
> **Caveat (foreign_reference):** the test-split `foreign_reference` group spans 9 distinct speakers/voices (1618 clips). None of these appear in the train split -- a genuinely held-out comparison for this group.

| group | n | accept_rate | exact_accuracy | 95% CI |
|---|---|---|---|---|
| filipino_reference | 1747 | 0.986 | 0.985 | [0.978, 0.989] |
| foreign_reference | 1618 | 0.992 | 0.992 | [0.986, 0.995] |

- exact_accuracy_gap_foreign_minus_filipino: 0.007; n_unclassified: 2162

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
| ALARM | ALARM=515, REJECTED=10 |
| BRIGHTNESS | BRIGHTNESS=381, REJECTED=6 |
| CALL | CALL=145, REJECTED=9 |
| COLOR | COLOR=551, REJECTED=2 |
| CREATE_REMINDER | CREATE_REMINDER=495, REJECTED=11 |
| LIGHT_OFF | LIGHT_OFF=182, LIGHT_ON=3, REJECTED=5 |
| LIGHT_ON | LIGHT_ON=188 |
| LIST_REMINDERS | LIST_REMINDERS=171, REJECTED=7 |
| MESSAGE | MESSAGE=181, REJECTED=2 |
| NEXT | NEXT=331, REJECTED=2 |
| PAUSE | PAUSE=153, REJECTED=4 |
| PLAY_MUSIC | PLAY_MUSIC=187, REJECTED=2 |
| STOP | REJECTED=4, STOP=222 |
| TEMPERATURE | REJECTED=10, TEMPERATURE=538 |
| TIME | REJECTED=5, TIME=152 |
| TIMER | REJECTED=28, TIMER=473 |
| VOLUME_DOWN | REJECTED=4, VOLUME_DOWN=161, VOLUME_UP=1 |
| VOLUME_UP | REJECTED=3, VOLUME_UP=208 |
| WEATHER | REJECTED=2, WEATHER=173 |

## Slot-eval-set (Task 07) results

Skipped: out/vcm/option-d-fil50-ambient-rir-135m/metadata/slot_eval/manifest.csv not found (Task 07 clips may not exist yet)

