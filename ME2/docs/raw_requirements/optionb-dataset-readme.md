<!--
Vendored verbatim from source repo `markandrian30/AI231`, path
`MEX2/OptionB/README.md`. Fetched 2026-09-24, commit `b9d86ea9a9ee0c0ec9f5cb475d5ec33e1aabfca0`.

Quoted reference material -- do not edit; the grammar's drift guard
(`tests/test_optionb_grammar.py`) parses this file's "Phrase variations"
and "Slot values" tables directly.
-->

# Option B: Spoken Command Dataset

- **100 speakers:** 84 foreign and 16 Filipino-English.
- **13 fixed intents:** 3 phrases each.
- **6 variable intents:** 3 phrase templates x 3 slot values = 9 utterances each.
- **Two acoustic conditions:** clean and light background noise.
- **18,600 original files; 17,986 remaining after filtering.**
  - **Transcriber used:** [simple-audio-transcriber by Martinnavs](https://github.com/Martinnavs/simple-audio-transcriber)

## Dataset overview

| Item | Count |
|---|---:|
| Reference speakers | 100 |
| Foreign speakers ([LibriSpeech](https://www.openslr.org/12), [small subsets](https://www.openslr.org/31)) | 84 |
| Filipino-English speakers ([SilencioPH](https://huggingface.co/datasets/SilencioNetwork/tagalog-filipino-speech)) | 16 |
| Intents | 19 |
| Intents without slots | 13 |
| Intents with slots | 6 |
| Acoustic conditions per utterance | 2 |
| Active clean WAV files | 8,993 |
| Active noisy WAV files | 8,993 |
| Original WAV files | 18,600 |
| Excluded WAV paths | 614 |
| WAV files stored in FLAGGED | 614 |
| **Remaining active WAV files** | 17,986 |

## Speakers

- All 100 speaker IDs remain after filtering.
- The table shows original file counts.

| Group | Speaker IDs | Speakers | WAV files |
|---|---|---:|---:|
| Foreign (LibriSpeech) | `s1-s67`, `s81-s88`, `s91-s99` | 84 | 15,624 |
| Filipino-English (SilencioPH) | `s68-s80`, `s89-s90`, `s100` | 16 | 2,976 |
| **Total** | `s1-s100` | **100** | **18,600** |

| Metadata split | Foreign speaker IDs | Filipino speaker IDs | Total speakers |
|---|---|---|---:|
| Train | `s1-s67` | `s68-s80` | 80 |
| Validation | `s81-s88` | `s89-s90` | 10 |
| Test | `s91-s99` | `s100` | 10 |

## Acoustic conditions

| Filename suffix | Condition | Files |
|---|---|---:|
| `_clean.wav` | Clean speech | 8,993 |
| `_noisy.wav` | Speech with light background noise; target approximately 30 dB SNR | 8,993 |

## Phrase variations

| Group | Intent | v1 | v2 | v3 |
|---|---|---|---|---|
| Music control | `PLAY_MUSIC` | Play music | Start music | Play some music |
| Music control | `VOLUME_UP` | Volume up | Increase the volume | Turn the volume up |
| Music control | `VOLUME_DOWN` | Volume down | Lower the volume | Turn the volume down |
| Music control | `NEXT` | Next song | Skip song | Play next song |
| Music control | `PAUSE` | Pause | Pause audio | Pause song |
| Music control | `STOP` | Stop | Stop playing | End playback |
| Lighting | `LIGHT_ON` | Lights on | Power on the lights | Turn on the lights |
| Lighting | `LIGHT_OFF` | Lights out | Kill the lights | Shut off the lights |
| Lighting | `BRIGHTNESS` | Brightness {percent} | Adjust brightness to {percent} | Brightness level {percent} |
| Lighting | `COLOR` | Change color to {color} | Switch color to {color} | Set color to {color} |
| Temperature control | `TEMPERATURE` | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees} |
| Information | `WEATHER` | Weather | What's the weather? | Tell me the weather |
| Information | `TIME` | Time | What time is it? | Tell me the time |
| Timers and alarms | `TIMER` | Timer {duration} | Countdown for {duration} | Start a timer for {duration} |
| Timers and alarms | `ALARM` | Alarm {time} | Wake me up at {time} | Set an alarm for {time} |
| Communication | `CALL` | Call | Make a call | Make a phone call |
| Communication | `MESSAGE` | Message | Send a message | Send my message |
| Reminders | `CREATE_REMINDER` | Reminder {task} | Remind me to {task} | Create a reminder to {task} |
| Reminders | `LIST_REMINDERS` | Reminders | Show my reminders | List my reminders |

### Slot values

| Intent            | Slot         | Values                              |
| ----------------- | ------------ | ----------------------------------- |
| `TIMER`           | `{duration}` | 10 seconds; 30 seconds; 1 minute    |
| `ALARM`           | `{time}`     | 6 AM; 8 AM; 9 PM                    |
| `TEMPERATURE`     | `{degrees}`  | 18 degrees; 22 degrees; 26 degrees  |
| `BRIGHTNESS`      | `{percent}`  | 20 percent; 60 percent; 100 percent |
| `COLOR`           | `{color}`    | red; blue; green                    |
| `CREATE_REMINDER` | `{task}`     | drink water; study; exercise       |

## Transcription-Based Filtering Results

Files are flagged when transcription similarity is **below 0.80**.

| Intent / slot folder | Original WAV files | Excluded WAV paths | Remaining active WAV files |
|---|---:|---:|---:|
| `PLAY_MUSIC` | 600 | 4 | 596 |
| `WEATHER` | 600 | 56 | 544 |
| `TIME` | 600 | 52 | 548 |
| `LIGHT_ON` | 600 | 30 | 570 |
| `LIGHT_OFF` | 600 | 34 | 566 |
| `PAUSE` | 600 | 26 | 574 |
| `STOP` | 600 | 40 | 560 |
| `NEXT` | 600 | 2 | 598 |
| `VOLUME_UP` | 600 | 34 | 566 |
| `VOLUME_DOWN` | 600 | 22 | 578 |
| `CALL` | 600 | 48 | 552 |
| `MESSAGE` | 600 | 72 | 528 |
| `LIST_REMINDERS` | 600 | 42 | 558 |
| `TIMER_10s` | 600 | 12 | 588 |
| `TIMER_30s` | 600 | 6 | 594 |
| `TIMER_1m` | 600 | 32 | 568 |
| `ALARM_6_00AM` | 600 | 16 | 584 |
| `ALARM_8_00AM` | 600 | 14 | 586 |
| `ALARM_9_00PM` | 600 | 16 | 584 |
| `TEMPERATURE_18` | 600 | 2 | 598 |
| `TEMPERATURE_22` | 600 | 0 | 600 |
| `TEMPERATURE_26` | 600 | 2 | 598 |
| `BRIGHTNESS_20` | 600 | 4 | 596 |
| `BRIGHTNESS_60` | 600 | 4 | 596 |
| `BRIGHTNESS_100` | 600 | 0 | 600 |
| `COLOR_RED` | 600 | 4 | 596 |
| `COLOR_BLUE` | 600 | 2 | 598 |
| `COLOR_GREEN` | 600 | 4 | 596 |
| `CREATE_REMINDER_DRINK_WATER` | 600 | 4 | 596 |
| `CREATE_REMINDER_STUDY` | 600 | 22 | 578 |
| `CREATE_REMINDER_EXERCISE` | 600 | 8 | 592 |
| **Total** | **18,600** | **614** | **17,986** |

## Filename convention

`<FOLDER_NAME>_s<speaker>_v<phrase variation>_<condition>.wav`

Example: `BRIGHTNESS_100/BRIGHTNESS_100_s68_v3_noisy.wav`

| Component | Meaning |
|---|---|
| `BRIGHTNESS_100/` | Intent folder with the 100-percent slot value |
| `BRIGHTNESS_100` | Folder name: intent and slot value |
| `s68` | Reference speaker 68, Filipino-English |
| `v3` | Third phrase variation |
| `noisy` | Acoustic condition |

- Source filenames match their folders.
- Source manifest and report references use the updated names.

## Metadata

- `manifest.csv`: relative audio paths, intent, speaker, split, phrase variation, acoustic condition, transcript, slot value, and duration.
- `labels.json` and `slots.json`: intent and slot definitions.
- `OPTIONB_DATA_SUMMARY.md` and `dataset_design.txt`: supporting dataset documentation.

Resolve manifest paths relative to this directory. Metadata records the split assignments.

## Updates (2026-09-22)

- Updated PLAY_MUSIC, NEXT, PAUSE, and STOP to the current three phrase variations.
- Active files: **17,964**; currently excluded paths: **636**.

### Previous updates (2026-09-21)

| Item | Previous | Current | Reason |
|---|---|---|---|
| `CREATE_REMINDER` task slot | `call home` | `exercise` | Reduce confusion with the `CALL` intent. |
| `VOLUME_DOWN` variation 2 | Decrease the volume | Lower the volume | Reduce confusion between the similar-sounding words increase and decrease. |
| `NEXT` phrases | Skip song; Next song; Play next song | Next song; Skip song; Play next song | Use explicit song-navigation phrases. |
| `PAUSE` phrases | Pause; Pause the music; Pause this song | Pause; Pause audio; Pause song | Use shorter pause phrases and avoid music-specific wording. |
| `STOP` phrases | Stop song; Stop music; Stop playing music | Stop; Stop playing; End playback | Use direct playback-stop phrases and avoid song or music wording. |
