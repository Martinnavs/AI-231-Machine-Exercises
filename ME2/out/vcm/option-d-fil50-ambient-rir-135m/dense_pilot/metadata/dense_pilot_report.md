# Dense phonetic scoring pilot

**Verdict: NO-GO** (criteria_failed:R3,R5,R6)

Selected config (val): `d2-global`; parity: PASS

## Criteria (test, margin 4.0, selected vs baseline)

| criterion | pass | value |
|---|---|---|
| R1 | True | `[0, 4]` |
| R2 | True | `[0, 2]` |
| R3 | False | `0.0625` |
| R4 | True | `[0, 2]` |
| R5 | False | `{"clean": [3425, 3434, 3494], "noisy": [3251, 3282, 3494]}` |
| R6 | False | `{"ALARM": true, "BRIGHTNESS": true, "CALL": true, "COLOR": true, "CREATE_REMINDER": true, "LIGHT_OFF": true, "LIGHT_ON": true, "LIST_REMINDERS": true, "MESSAGE": true, "NEXT": true, "PAUSE": false, "PLAY_MUSIC": true, "STOP": false, "TEMPERATURE": true, "TIME": false, "TIMER": true, "VOLUME_DOWN": true, "VOLUME_UP": true, "WEATHER": true}` |
| R7 | True | `[0, 1]` |

## Headline (test)

| config | margin | cond | exact | babble FA | silence FA | TIME+STOP FA |
|---|---|---|---|---|---|---|
| baseline-global | None | clean | 3448/3494 (0.9868) | 3/255 | 1/324 | 3 |
| baseline-global | None | noisy_s0 | 3345/3494 (0.9574) | 10/255 | 6/324 | 13 |
| baseline-global | 4.0 | clean | 3434/3494 (0.9828) | 1/255 | 1/324 | 2 |
| baseline-global | 4.0 | noisy_s0 | 3282/3494 (0.9393) | 2/255 | 2/324 | 1 |
| d2-global | None | clean | 3437/3494 (0.9837) | 0/255 | 0/324 | 0 |
| d2-global | None | noisy_s0 | 3297/3494 (0.9436) | 0/255 | 0/324 | 0 |
| d2-global | 4.0 | clean | 3425/3494 (0.9803) | 0/255 | 0/324 | 0 |
| d2-global | 4.0 | noisy_s0 | 3251/3494 (0.9305) | 0/255 | 0/324 | 0 |

## Margin x dense overlap (test, baseline FAs at margin None)

```json
{
  "clean": {
    "baseline_fa_margin_none": 4,
    "removed_by_margin_only": 0,
    "removed_by_dense_only": 2,
    "removed_by_both": 2,
    "removed_by_neither": 0
  },
  "noisy_s0": {
    "baseline_fa_margin_none": 16,
    "removed_by_margin_only": 0,
    "removed_by_dense_only": 4,
    "removed_by_both": 12,
    "removed_by_neither": 0
  }
}
```

MV2 alignment failures: `{'val/clean': 0, 'val/noisy_s0': 0, 'test/clean': 0, 'test/noisy_s0': 0}`
MV4 mass-minus-viterbi (val targets): `{'n': 3323, 'median': 3.9426441822105294, 'p95': 9.644039717282824, 'max': 21.065512466496894}`
