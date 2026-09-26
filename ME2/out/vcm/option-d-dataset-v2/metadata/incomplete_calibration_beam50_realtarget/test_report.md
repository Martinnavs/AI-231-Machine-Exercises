# Incomplete-prefix rejection gate -- held-out test report

Checkpoint: `out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt`  Beam width: 50
Margin (from val selection): 4.0  Threshold: -0.075
Val selection: `out/vcm/option-d-dataset-v2/metadata/incomplete_calibration_beam50_realtarget/val_selection.json`

## baseline (gate off)

Target exact accuracy: 114/129 (0.8837209302325582)

Slot exact match: n/a (no classifiable slot-bearing Option B target row)

Incomplete-prefix FAR: 1182/10688 (0.11059131736526946)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 2 | 0.025 | PAUSE=1, STOP=1 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 6 | 0.075 | CALL=2, PAUSE=4 |
| alarm eight |  | 64 | 8 | 0.125 | ALARM=8 |
| alarm nine |  | 80 | 0 | 0.0 | - |
| alarm six |  | 80 | 1 | 0.0125 | PAUSE=1 |
| brightness |  | 80 | 5 | 0.0625 | PAUSE=1, TIME=4 |
| brightness level |  | 80 | 1 | 0.0125 | PAUSE=1 |
| brightness level one |  | 80 | 0 | 0.0 | - |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 0 | 0.0 | - |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 3 | 0.0375 | LIGHT_ON=3 |
| brightness one hundred |  | 80 | 2 | 0.025 | LIGHT_ON=2 |
| brightness sixty |  | 80 | 0 | 0.0 | - |
| brightness twenty |  | 80 | 0 | 0.0 | - |
| change |  | 80 | 13 | 0.1625 | STOP=13 |
| change color |  | 80 | 0 | 0.0 | - |
| change color to |  | 80 | 0 | 0.0 | - |
| change the |  | 80 | 4 | 0.05 | PLAY_MUSIC=4 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 4 | 0.05 | STOP=4 |
| countdown for |  | 80 | 0 | 0.0 | - |
| countdown for one |  | 80 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 3 | 0.0375 | STOP=3 |
| create |  | 80 | 2 | 0.025 | CALL=2 |
| create a |  | 80 | 3 | 0.0375 | CALL=2, PAUSE=1 |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 0 | 0.0 | - |
| end |  | 72 | 34 | 0.4722222222222222 | STOP=34 |
| increase |  | 80 | 1 | 0.0125 | TIME=1 |
| increase the |  | 80 | 0 | 0.0 | - |
| kill |  | 80 | 34 | 0.425 | STOP=34 |
| kill the |  | 80 | 19 | 0.2375 | CALL=4, STOP=14, TIME=1 |
| lights |  | 80 | 16 | 0.2 | LIGHT_ON=16 |
| list |  | 80 | 6 | 0.075 | CALL=1, STOP=4, WEATHER=1 |
| list my |  | 80 | 5 | 0.0625 | LIGHT_OFF=4, LIGHT_ON=1 |
| lower |  | 72 | 19 | 0.2638888888888889 | PAUSE=3, STOP=16 |
| lower the |  | 72 | 0 | 0.0 | - |
| make |  | 80 | 4 | 0.05 | STOP=2, TIME=2 |
| make a |  | 80 | 2 | 0.025 | TIME=2 |
| make a phone |  | 80 | 0 | 0.0 | - |
| next |  | 80 | 3 | 0.0375 | STOP=3 |
| pause for |  | 72 | 47 | 0.6527777777777778 | PAUSE=47 |
| place |  | 80 | 4 | 0.05 | PAUSE=1, STOP=3 |
| place a |  | 80 | 2 | 0.025 | PAUSE=1, STOP=1 |
| play |  | 80 | 3 | 0.0375 | PAUSE=2, STOP=1 |
| play next |  | 80 | 0 | 0.0 | - |
| play some |  | 80 | 6 | 0.075 | PLAY_MUSIC=6 |
| power |  | 72 | 43 | 0.5972222222222222 | CALL=9, STOP=34 |
| power on |  | 72 | 13 | 0.18055555555555555 | PAUSE=7, STOP=4, TIME=1, VOLUME_UP=1 |
| power on the |  | 72 | 7 | 0.09722222222222222 | PAUSE=5, STOP=2 |
| remind | YES | 80 | 5 | 0.0625 | LIST_REMINDERS=5 |
| remind me |  | 80 | 23 | 0.2875 | LIST_REMINDERS=22, STOP=1 |
| remind me to |  | 80 | 17 | 0.2125 | LIST_REMINDERS=14, TIME=3 |
| remind me to drink |  | 80 | 0 | 0.0 | - |
| reminder | YES | 80 | 15 | 0.1875 | LIST_REMINDERS=15 |
| reminder drink |  | 80 | 12 | 0.15 | LIST_REMINDERS=12 |
| send |  | 80 | 37 | 0.4625 | STOP=37 |
| send a |  | 72 | 22 | 0.3055555555555556 | CALL=1, PAUSE=1, STOP=20 |
| send my |  | 80 | 10 | 0.125 | PAUSE=2, STOP=8 |
| set |  | 80 | 31 | 0.3875 | STOP=31 |
| set an |  | 80 | 31 | 0.3875 | STOP=31 |
| set an alarm |  | 80 | 0 | 0.0 | - |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 3 | 0.0375 | ALARM=3 |
| set an alarm for nine |  | 80 | 0 | 0.0 | - |
| set an alarm for six |  | 72 | 0 | 0.0 | - |
| set color |  | 80 | 0 | 0.0 | - |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 30 | 0.375 | STOP=30 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 45 | 0.5625 | STOP=45 |
| show my |  | 80 | 14 | 0.175 | PAUSE=1, STOP=13 |
| shut |  | 80 | 50 | 0.625 | STOP=50 |
| shut off |  | 80 | 0 | 0.0 | - |
| shut off the |  | 80 | 0 | 0.0 | - |
| skip |  | 80 | 35 | 0.4375 | STOP=35 |
| start |  | 80 | 57 | 0.7125 | STOP=57 |
| start a |  | 80 | 55 | 0.6875 | STOP=55 |
| start a timer |  | 80 | 1 | 0.0125 | PAUSE=1 |
| start a timer for |  | 80 | 0 | 0.0 | - |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 16 | 0.2 | STOP=16 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 48 | 0.6 | CALL=6, STOP=42 |
| tell me |  | 80 | 12 | 0.15 | STOP=1, TIME=11 |
| tell me the |  | 80 | 1 | 0.0125 | TIME=1 |
| temperature |  | 80 | 2 | 0.025 | PAUSE=1, TIME=1 |
| temperature eighteen |  | 80 | 0 | 0.0 | - |
| temperature twenty |  | 80 | 1 | 0.0125 | TIME=1 |
| temperature twenty six |  | 80 | 0 | 0.0 | - |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 51 | 0.6375 | CALL=6, STOP=10, TIME=33, VOLUME_UP=2 |
| timer one |  | 64 | 1 | 0.015625 | TIME=1 |
| timer ten |  | 80 | 14 | 0.175 | PLAY_MUSIC=1, STOP=2, TIME=9, VOLUME_UP=2 |
| timer thirty |  | 80 | 17 | 0.2125 | TIME=17 |
| turn |  | 80 | 38 | 0.475 | STOP=18, TIME=20 |
| turn on |  | 80 | 20 | 0.25 | STOP=20 |
| turn on the |  | 80 | 11 | 0.1375 | STOP=9, TIME=2 |
| turn the |  | 80 | 28 | 0.35 | STOP=3, TIME=25 |
| turn the volume |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| volume |  | 80 | 28 | 0.35 | CALL=13, PAUSE=14, STOP=1 |
| wake |  | 80 | 3 | 0.0375 | LIGHT_OFF=1, LIGHT_ON=1, WEATHER=1 |
| wake me |  | 80 | 6 | 0.075 | LIGHT_ON=2, VOLUME_UP=2, WEATHER=2 |
| wake me up |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| wake me up at |  | 80 | 0 | 0.0 | - |
| wake me up at eight |  | 80 | 2 | 0.025 | ALARM=2 |
| wake me up at nine |  | 72 | 0 | 0.0 | - |
| wake me up at six |  | 80 | 0 | 0.0 | - |
| what | YES | 80 | 14 | 0.175 | STOP=7, WEATHER=7 |
| what time |  | 80 | 8 | 0.1 | LIGHT_OFF=1, LIGHT_ON=3, STOP=4 |
| what time is |  | 80 | 8 | 0.1 | TIME=8 |
| what's |  | 80 | 32 | 0.4 | MESSAGE=1, STOP=2, WEATHER=29 |
| what's the |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2672 | 11 | 0.004116766467065869 |
| lead_in_room_tone | 798 | 85 | 0.10651629072681704 |
| tail_room_tone | 3828 | 472 | 0.12330198537095088 |
| digital_zero | 3390 | 614 | 0.18112094395280237 |

FRR (single-word commands): 0/16 (0.0)
FRR (strict-prefix commands): 0/16 (0.0)

Gate-caused FRR: 0/114 confidence-accepted target rows additionally rejected (0.0)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|

Babble/silence FAR:
- babble: 1/255 (0.00392156862745098)
- silence: 3/324 (0.009259259259259259)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| unknown | 129 | 114 | 0.8837209302325582 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2672 | 641 | 0.23989520958083832 |
| [0, 0.2)s | 2672 | 11 | 0.004116766467065869 |
| [0.2, 0.5)s | 2672 | 151 | 0.05651197604790419 |
| [1, 2)s | 2672 | 379 | 0.14184131736526945 |

## margin=4.0

Target exact accuracy: 112/129 (0.8682170542635659)

Slot exact match: n/a (no classifiable slot-bearing Option B target row)

Incomplete-prefix FAR: 719/10688 (0.06727170658682635)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 1 | 0.0125 | STOP=1 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 3 | 0.0375 | PAUSE=3 |
| alarm eight |  | 64 | 0 | 0.0 | - |
| alarm nine |  | 80 | 0 | 0.0 | - |
| alarm six |  | 80 | 0 | 0.0 | - |
| brightness |  | 80 | 4 | 0.05 | TIME=4 |
| brightness level |  | 80 | 0 | 0.0 | - |
| brightness level one |  | 80 | 0 | 0.0 | - |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 0 | 0.0 | - |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 0 | 0.0 | - |
| brightness one hundred |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |
| brightness sixty |  | 80 | 0 | 0.0 | - |
| brightness twenty |  | 80 | 0 | 0.0 | - |
| change |  | 80 | 4 | 0.05 | STOP=4 |
| change color |  | 80 | 0 | 0.0 | - |
| change color to |  | 80 | 0 | 0.0 | - |
| change the |  | 80 | 4 | 0.05 | PLAY_MUSIC=4 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 1 | 0.0125 | STOP=1 |
| countdown for |  | 80 | 0 | 0.0 | - |
| countdown for one |  | 80 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 0 | 0.0 | - |
| create |  | 80 | 1 | 0.0125 | CALL=1 |
| create a |  | 80 | 0 | 0.0 | - |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 0 | 0.0 | - |
| end |  | 72 | 25 | 0.3472222222222222 | STOP=25 |
| increase |  | 80 | 1 | 0.0125 | TIME=1 |
| increase the |  | 80 | 0 | 0.0 | - |
| kill |  | 80 | 32 | 0.4 | STOP=32 |
| kill the |  | 80 | 7 | 0.0875 | STOP=7 |
| lights |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |
| list |  | 80 | 1 | 0.0125 | WEATHER=1 |
| list my |  | 80 | 2 | 0.025 | LIGHT_OFF=2 |
| lower |  | 72 | 16 | 0.2222222222222222 | PAUSE=2, STOP=14 |
| lower the |  | 72 | 0 | 0.0 | - |
| make |  | 80 | 1 | 0.0125 | STOP=1 |
| make a |  | 80 | 0 | 0.0 | - |
| make a phone |  | 80 | 0 | 0.0 | - |
| next |  | 80 | 2 | 0.025 | STOP=2 |
| pause for |  | 72 | 34 | 0.4722222222222222 | PAUSE=34 |
| place |  | 80 | 1 | 0.0125 | STOP=1 |
| place a |  | 80 | 1 | 0.0125 | STOP=1 |
| play |  | 80 | 0 | 0.0 | - |
| play next |  | 80 | 0 | 0.0 | - |
| play some |  | 80 | 6 | 0.075 | PLAY_MUSIC=6 |
| power |  | 72 | 40 | 0.5555555555555556 | CALL=8, STOP=32 |
| power on |  | 72 | 4 | 0.05555555555555555 | PAUSE=1, STOP=2, VOLUME_UP=1 |
| power on the |  | 72 | 0 | 0.0 | - |
| remind | YES | 80 | 0 | 0.0 | - |
| remind me |  | 80 | 1 | 0.0125 | STOP=1 |
| remind me to |  | 80 | 0 | 0.0 | - |
| remind me to drink |  | 80 | 0 | 0.0 | - |
| reminder | YES | 80 | 0 | 0.0 | - |
| reminder drink |  | 80 | 0 | 0.0 | - |
| send |  | 80 | 27 | 0.3375 | STOP=27 |
| send a |  | 72 | 11 | 0.1527777777777778 | STOP=11 |
| send my |  | 80 | 2 | 0.025 | PAUSE=1, STOP=1 |
| set |  | 80 | 20 | 0.25 | STOP=20 |
| set an |  | 80 | 10 | 0.125 | STOP=10 |
| set an alarm |  | 80 | 0 | 0.0 | - |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 0 | 0.0 | - |
| set an alarm for nine |  | 80 | 0 | 0.0 | - |
| set an alarm for six |  | 72 | 0 | 0.0 | - |
| set color |  | 80 | 0 | 0.0 | - |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 7 | 0.0875 | STOP=7 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 44 | 0.55 | STOP=44 |
| show my |  | 80 | 0 | 0.0 | - |
| shut |  | 80 | 42 | 0.525 | STOP=42 |
| shut off |  | 80 | 0 | 0.0 | - |
| shut off the |  | 80 | 0 | 0.0 | - |
| skip |  | 80 | 28 | 0.35 | STOP=28 |
| start |  | 80 | 54 | 0.675 | STOP=54 |
| start a |  | 80 | 47 | 0.5875 | STOP=47 |
| start a timer |  | 80 | 1 | 0.0125 | PAUSE=1 |
| start a timer for |  | 80 | 0 | 0.0 | - |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 2 | 0.025 | STOP=2 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 46 | 0.575 | CALL=6, STOP=40 |
| tell me |  | 80 | 7 | 0.0875 | TIME=7 |
| tell me the |  | 80 | 0 | 0.0 | - |
| temperature |  | 80 | 2 | 0.025 | PAUSE=1, TIME=1 |
| temperature eighteen |  | 80 | 0 | 0.0 | - |
| temperature twenty |  | 80 | 1 | 0.0125 | TIME=1 |
| temperature twenty six |  | 80 | 0 | 0.0 | - |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 41 | 0.5125 | CALL=5, STOP=8, TIME=27, VOLUME_UP=1 |
| timer one |  | 64 | 0 | 0.0 | - |
| timer ten |  | 80 | 3 | 0.0375 | PLAY_MUSIC=1, STOP=1, VOLUME_UP=1 |
| timer thirty |  | 80 | 0 | 0.0 | - |
| turn |  | 80 | 29 | 0.3625 | STOP=14, TIME=15 |
| turn on |  | 80 | 12 | 0.15 | STOP=12 |
| turn on the |  | 80 | 5 | 0.0625 | STOP=5 |
| turn the |  | 80 | 7 | 0.0875 | STOP=1, TIME=6 |
| turn the volume |  | 80 | 0 | 0.0 | - |
| volume |  | 80 | 22 | 0.275 | CALL=9, PAUSE=12, STOP=1 |
| wake |  | 80 | 0 | 0.0 | - |
| wake me |  | 80 | 4 | 0.05 | LIGHT_ON=1, VOLUME_UP=2, WEATHER=1 |
| wake me up |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| wake me up at |  | 80 | 0 | 0.0 | - |
| wake me up at eight |  | 80 | 0 | 0.0 | - |
| wake me up at nine |  | 72 | 0 | 0.0 | - |
| wake me up at six |  | 80 | 0 | 0.0 | - |
| what | YES | 80 | 9 | 0.1125 | STOP=2, WEATHER=7 |
| what time |  | 80 | 8 | 0.1 | LIGHT_OFF=1, LIGHT_ON=3, STOP=4 |
| what time is |  | 80 | 4 | 0.05 | TIME=4 |
| what's |  | 80 | 28 | 0.35 | MESSAGE=1, STOP=1, WEATHER=26 |
| what's the |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2672 | 5 | 0.0018712574850299401 |
| lead_in_room_tone | 798 | 36 | 0.045112781954887216 |
| tail_room_tone | 3828 | 292 | 0.07628004179728318 |
| digital_zero | 3390 | 386 | 0.11386430678466077 |

FRR (single-word commands): 0/16 (0.0)
FRR (strict-prefix commands): 0/16 (0.0)

Gate-caused FRR: 2/114 confidence-accepted target rows additionally rejected (0.017543859649122806)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|
| play |  | 1 | PLAY_MUSIC=1 |
| turn the volume |  | 1 | VOLUME_DOWN=1 |

Babble/silence FAR:
- babble: 1/255 (0.00392156862745098)
- silence: 1/324 (0.0030864197530864196)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| unknown | 129 | 112 | 0.8682170542635659 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2672 | 352 | 0.1317365269461078 |
| [0, 0.2)s | 2672 | 5 | 0.0018712574850299401 |
| [0.2, 0.5)s | 2672 | 116 | 0.04341317365269461 |
| [1, 2)s | 2672 | 246 | 0.09206586826347306 |

## Checkpoint/ONNX hashes

Before: {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
After:  {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
Unchanged: True

