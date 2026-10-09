<!--
Derived from the real upstream `manifest.csv` (17,986 data rows), source repo
`markandrian30/AI231`, path `MEX2/OptionB/manifest.csv`. Fetched/derived
2026-09-24, commit `b9d86ea9a9ee0c0ec9f5cb475d5ec33e1aabfca0`.

This is NOT a copy of the vendored README
(`docs/raw_requirements/optionb-dataset-readme.md`) -- it is independently
derived by grouping the manifest's own `transcript` column by
`(intent, phrase_id)` and, for slotted intents, substituting each row's own
`slot_value` back out of its `transcript` to recover the template. It exists
because, historically, the README and the real dataset have disagreed on
individual phrases (see "Divergence from the README" below); the drift guard
in `tests/test_optionb_grammar.py` treats this table as the primary
correctness anchor while keeping the README as the secondary anchor. Quoted
reference material -- do not edit by hand; regenerate from a fresh manifest
fetch if the upstream dataset changes.
-->

# Option B dataset: manifest-derived transcript summary

Derived directly from the real `manifest.csv`'s `transcript`, `intent`,
`phrase_id`, `slot`, and `slot_value` columns (17,986 data rows, 93 distinct
transcripts -- matches `OPTIONB_GRAMMAR`'s canonical-93 count exactly, and,
as of this refresh, the same 93 strings as the README; see divergence
below).

## Phrase variations

Same 5-column shape as the README's own "Phrase variations" table (`Group`
retained but not meaningful here -- kept only so both anchors parse with the
same table-extraction logic in `tests/test_optionb_grammar.py`).

| Group | Intent | v1 | v2 | v3 |
|---|---|---|---|---|
| - | `PLAY_MUSIC` | Play music | Start music | Play some music |
| - | `VOLUME_UP` | Volume up | Increase the volume | Turn the volume up |
| - | `VOLUME_DOWN` | Volume down | Lower the volume | Turn the volume down |
| - | `NEXT` | Next song | Skip song | Play next song |
| - | `PAUSE` | Pause | Pause audio | Pause song |
| - | `STOP` | Stop | Stop playing | End playback |
| - | `LIGHT_ON` | Lights on | Power on the lights | Turn on the lights |
| - | `LIGHT_OFF` | Lights out | Kill the lights | Shut off the lights |
| - | `BRIGHTNESS` | Brightness {percent} | Adjust brightness to {percent} | Brightness level {percent} |
| - | `COLOR` | Change color to {color} | Switch color to {color} | Set color to {color} |
| - | `TEMPERATURE` | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees} |
| - | `WEATHER` | Weather | What's the weather? | Tell me the weather |
| - | `TIME` | Time | What time is it? | Tell me the time |
| - | `TIMER` | Timer {duration} | Countdown for {duration} | Start a timer for {duration} |
| - | `ALARM` | Alarm {time} | Wake me up at {time} | Set an alarm for {time} |
| - | `CALL` | Call | Make a call | Make a phone call |
| - | `MESSAGE` | Message | Send a message | Send my message |
| - | `CREATE_REMINDER` | Reminder {task} | Remind me to {task} | Create a reminder to {task} |
| - | `LIST_REMINDERS` | Reminders | Show my reminders | List my reminders |

### Slot values

Same 3-column shape as the README's own "Slot values" table.

| Intent | Slot | Values |
|---|---|---|
| `TIMER` | `{duration}` | 10 seconds; 30 seconds; 1 minute |
| `ALARM` | `{time}` | 6 AM; 8 AM; 9 PM |
| `TEMPERATURE` | `{degrees}` | 18 degrees; 22 degrees; 26 degrees |
| `BRIGHTNESS` | `{percent}` | 20 percent; 60 percent; 100 percent |
| `COLOR` | `{color}` | red; blue; green |
| `CREATE_REMINDER` | `{task}` | drink water; study; exercise |

Every slot value above is identical to the vendored README's slot-value
table -- no slot-value changes are in scope for this refresh.

## Divergence from the README

**None.** As of this refresh (upstream commit `b9d86ea9a9ee0c0ec9f5cb475d5ec33e1aabfca0`),
every one of the 93 manifest-derived transcripts above matches the README's
"Phrase variations" table exactly:

```
in MANIFEST but not in README: {}
in README but not in MANIFEST: {}
```

The prior divergence (`VOLUME_DOWN` v2: manifest said "Lower the volume",
README said "Decrease the volume") was resolved upstream -- the live README
now reads "Lower the volume" -- so `KNOWN_README_DIVERGENCES` in
`optionb/grammar.py` is now `[]`, and both drift-guard tests in
`tests/test_optionb_grammar.py` assert the two anchors agree on every
phrase rather than reconciling a known exception.

If a future upstream fetch reintroduces a disagreement between this file and
the README, add the new divergence to `KNOWN_README_DIVERGENCES` and update
this section together -- compare the fetched commit SHA above against a
fresh `git ls-remote` of `markandrian30/AI231` to detect that.
