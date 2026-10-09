# False-accept analysis: control vs. treatment, per-clip

Ad hoc analysis answering "did the bigger dataset actually fix specific
false accepts, or just move the aggregate rate?" Script:
`.scratch/optionb-v3-vcmx/false_accept_analysis.py`. Raw output:
`.scratch/optionb-v3-vcmx/false_accept_analysis.out`. Decodes every
babble/silence (reject-probe) row in val and test, for both the control
(`out/vcm/option-d-dataset-v2-control`) and treatment
(`out/vcm/option-d-dataset-v2`) checkpoints, at each run's own chosen
operating threshold (both **-0.1**), and diffs the two per-row.

## Headline: not just fewer false accepts — different, narrower ones

| | Val (738 reject rows) | Test (579 reject rows) | Combined |
|---|---|---|---|
| Control false accepts | 38 (5.1%) | 24 (4.1%) | 62 (4.7%) |
| Treatment false accepts | 13 (1.8%) | 8 (1.4%) | 21 (1.6%) |
| **Fixed by treatment** (control wrong, treatment right) | 33 | 21 | **54** |
| **Regressed by treatment** (treatment wrong, control was right) | 8 | 5 | **13** |
| Persistently wrong in both | 5 | 3 | 8 |

54 specific clips control false-accepted are now correctly rejected, against
13 new mistakes — a ~4:1 fix-to-regression ratio. This is evidence of real
generalization, not just a shifted operating point (both checkpoints were
evaluated at their own independently-chosen threshold, and both happen to be
-0.1).

## The confusion surface narrowed, not just shrank

Predicted intent of every false accept:

| | Control | Treatment |
|---|---|---|
| Val | PAUSE=7, TIME=11, LIGHT_ON=2, STOP=6, LIST_REMINDERS=2, COLOR=2, CALL=1, VOLUME_UP=2, WEATHER=2, ALARM=1, VOLUME_DOWN=1, NEXT=1 (12 intents) | STOP=4, TIME=5, MESSAGE=2, PAUSE=1, LIST_REMINDERS=1 (5 intents) |
| Test | TIME=13, STOP=3, WEATHER=3, LIST_REMINDERS=1, MESSAGE=1, ALARM=1, PAUSE=1, VOLUME_UP=1 (8 intents) | STOP=6, PAUSE=1, TIME=1 (3 intents) |

Control's false accepts spread across ~12 different intents. Treatment's
collapse almost entirely into **TIME** and **STOP** (16 of 21 combined false
accepts, 76%) — both short, phonetically generic words/phrases that are easy
for noisy or garbled speech to accidentally resemble. This is the "weak
spot" this analysis targets next.

## One specific, honest regression: near-silence decoded as STOP

4 of the 5 test-split regressions are **silence** clips the treatment model
newly false-accepts as `STOP`, several from an empty or near-empty decoded
string (`decoded_text=''`). Control correctly rejected these (`intent=None`).
This is the concrete mechanism behind the silence false-accept rate going
from 0.006 (control) to 0.015 (treatment) in the merged eval — it is not
random noise, it is a specific, small, learned tendency to read near-silence
as "stop."

## Mechanism: garbled beam-search decodes landing on a legal short phrase

Many false accepts on both checkpoints come from decoded text that is
visibly garbled (`'de emails'` -> TIME, `' o fr o '` -> STOP) but happens to
match a short, complete, grammar-legal phrase. This is exactly the failure
mode `docs/INCOMPLETE-GRAMMAR-REJECTION.md`'s margin gate
(`required_command_margin`) exists to catch — and it is disabled
(`None`) in every eval run in this feature so far. Both "time" and "stop"
are themselves strict prefixes of longer accepted phrases ("time" -> "timer
..."; "stop" -> "stop playing"), which is exactly the condition
`derive_incomplete_prefixes` flags them for, so the gate is structurally
positioned to help here. See the margin check below.

## Margin-gate check on the weak words (TIME, STOP)

Scoped check (not a full calibration): decoded only the 16 known TIME/STOP
false accepts plus the 253 legitimate TIME/STOP val+test commands (269
utterances total, one decode pass, treatment checkpoint only), reading
`incomplete_gap` off each decode -- this field is computed on every decode
regardless of whether the gate is enabled (`vcm/decoder.py`'s contract), so
one pass is enough to simulate any candidate margin after the fact with no
extra GPU time. Script: `.scratch/optionb-v3-vcmx/margin_check_weak_words.py`
(+ raw output/JSON in the same dir).

240/253 legitimate TIME/STOP commands are currently correctly accepted; their
`incomplete_gap` floor is **4.74** (min), p5 7.80, median 67.47. The 16 weak
false accepts range from -3.08 to 15.50. There's a clean gap between the two
distributions right around 4.7-4.8:

| margin | weak false-accepts caught (of 16) | legit TIME/STOP collaterally rejected (of 240) |
|---|---|---|
| 0.0 | 6 | 0 |
| 2.0 | 9 | 0 |
| 3.5 | 12 | 0 |
| **4.0-4.7** | **13** | **0** |
| 4.8 | 13 | 1 |
| 6.0 | 14 | 1 |
| 8.0 | 14 | 14 |
| 10.0 | 14 | 34 |
| 16.0 | 16 (all) | 36 (15%) |

**Recommendation: `required_command_margin` around 4.5.** Catches 13 of 16
(81%) of this checkpoint's TIME/STOP weak-spot false accepts with **zero**
cost to legitimate command acceptance. Pushing further to catch the last 3
costs real accuracy fast (14 becomes 34, then 15% of all real TIME/STOP
commands wrongly rejected by margin 16) -- not worth it for 2-3 more babble
rejections.

**Cross-check, not a validation:** the old baseline checkpoint's own full
calibration (`out/vcm/optionb-optiond/metadata/incomplete_calibration/
val_selection.json`) -- a different checkpoint, different grammar, beam
width 25 instead of 50, calibrated across every intent's prefixes rather
than just TIME/STOP -- independently landed on **margin=5.0**. That's not
proof this checkpoint's optimal margin is the same value (too many
differences to treat as one), but it's a reassuring plausibility signal that
4.5-5 is the right order of magnitude for this decoder/grammar family, not
an artifact of this analysis's small 16-example sample.

**Scope caveat:** this only checked TIME/STOP, the two intents this
checkpoint's false accepts happen to concentrate in. A real calibration run
(`vcm.incomplete_calibration`, as already exists for the old baseline) would
sweep the whole grammar's designated incomplete prefixes and could land on a
different value once every intent's collateral cost is considered -- this is
a fast, scoped signal for a decision, not a substitute for that full run if
this ships.
