# Incomplete-prefix rejection gate -- held-out test report

Checkpoint: `out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt`  Beam width: 50
Margin (from val selection): 6.0  Threshold: -0.1
Val selection: `out/vcm/option-d-dataset-v2/metadata/incomplete_calibration_beam50/val_selection.json`

## baseline (gate off)

Target exact accuracy: 1887/1927 (0.9792423456149455)

Slot exact match (given intent correct): 1057/1057 (1.0), over 1068 slot-bearing target rows (0 unparseable ground truth)

Incomplete-prefix FAR: 1566/10688 (0.1465194610778443)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 5 | 0.0625 | PAUSE=4, STOP=1 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 11 | 0.1375 | CALL=3, PAUSE=7, STOP=1 |
| alarm eight |  | 64 | 14 | 0.21875 | ALARM=14 |
| alarm nine |  | 80 | 2 | 0.025 | PAUSE=1, VOLUME_DOWN=1 |
| alarm six |  | 80 | 1 | 0.0125 | PAUSE=1 |
| brightness |  | 80 | 7 | 0.0875 | LIGHT_ON=1, PAUSE=1, TIME=5 |
| brightness level |  | 80 | 1 | 0.0125 | PAUSE=1 |
| brightness level one |  | 80 | 0 | 0.0 | - |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 0 | 0.0 | - |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 4 | 0.05 | LIGHT_ON=4 |
| brightness one hundred |  | 80 | 3 | 0.0375 | LIGHT_ON=3 |
| brightness sixty |  | 80 | 0 | 0.0 | - |
| brightness twenty |  | 80 | 0 | 0.0 | - |
| change |  | 80 | 17 | 0.2125 | STOP=17 |
| change color |  | 80 | 0 | 0.0 | - |
| change color to |  | 80 | 0 | 0.0 | - |
| change the |  | 80 | 4 | 0.05 | PLAY_MUSIC=4 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 6 | 0.075 | STOP=6 |
| countdown for |  | 80 | 0 | 0.0 | - |
| countdown for one |  | 80 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 3 | 0.0375 | STOP=3 |
| create |  | 80 | 4 | 0.05 | CALL=4 |
| create a |  | 80 | 5 | 0.0625 | CALL=4, PAUSE=1 |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 0 | 0.0 | - |
| end |  | 72 | 38 | 0.5277777777777778 | STOP=38 |
| increase |  | 80 | 2 | 0.025 | TIME=2 |
| increase the |  | 80 | 0 | 0.0 | - |
| kill |  | 80 | 37 | 0.4625 | STOP=37 |
| kill the |  | 80 | 29 | 0.3625 | CALL=11, STOP=16, TIME=2 |
| lights |  | 80 | 24 | 0.3 | LIGHT_ON=24 |
| list |  | 80 | 11 | 0.1375 | CALL=2, PAUSE=1, STOP=6, WEATHER=2 |
| list my |  | 80 | 14 | 0.175 | LIGHT_OFF=7, LIGHT_ON=2, PAUSE=4, PLAY_MUSIC=1 |
| lower |  | 72 | 30 | 0.4166666666666667 | CALL=2, PAUSE=6, STOP=22 |
| lower the |  | 72 | 0 | 0.0 | - |
| make |  | 80 | 8 | 0.1 | STOP=4, TIME=4 |
| make a |  | 80 | 4 | 0.05 | STOP=1, TIME=3 |
| make a phone |  | 80 | 0 | 0.0 | - |
| next |  | 80 | 7 | 0.0875 | NEXT=1, STOP=6 |
| pause for |  | 72 | 51 | 0.7083333333333334 | PAUSE=51 |
| place |  | 80 | 4 | 0.05 | PAUSE=1, STOP=3 |
| place a |  | 80 | 6 | 0.075 | PAUSE=4, STOP=2 |
| play |  | 80 | 10 | 0.125 | CALL=3, PAUSE=5, STOP=2 |
| play next |  | 80 | 0 | 0.0 | - |
| play some |  | 80 | 6 | 0.075 | PLAY_MUSIC=6 |
| power |  | 72 | 48 | 0.6666666666666666 | CALL=11, STOP=37 |
| power on |  | 72 | 16 | 0.2222222222222222 | PAUSE=10, STOP=4, TIME=1, VOLUME_UP=1 |
| power on the |  | 72 | 11 | 0.1527777777777778 | PAUSE=8, STOP=3 |
| remind | YES | 80 | 10 | 0.125 | LIST_REMINDERS=9, TIME=1 |
| remind me |  | 80 | 30 | 0.375 | LIST_REMINDERS=29, STOP=1 |
| remind me to |  | 80 | 26 | 0.325 | LIST_REMINDERS=23, TIME=3 |
| remind me to drink |  | 80 | 3 | 0.0375 | CREATE_REMINDER=2, LIST_REMINDERS=1 |
| reminder | YES | 80 | 26 | 0.325 | LIST_REMINDERS=24, TIME=2 |
| reminder drink |  | 80 | 20 | 0.25 | LIST_REMINDERS=20 |
| send |  | 80 | 38 | 0.475 | STOP=38 |
| send a |  | 72 | 27 | 0.375 | CALL=1, PAUSE=1, STOP=25 |
| send my |  | 80 | 15 | 0.1875 | PAUSE=3, STOP=12 |
| set |  | 80 | 38 | 0.475 | STOP=38 |
| set an |  | 80 | 37 | 0.4625 | STOP=37 |
| set an alarm |  | 80 | 1 | 0.0125 | STOP=1 |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 18 | 0.225 | ALARM=18 |
| set an alarm for nine |  | 80 | 6 | 0.075 | ALARM=6 |
| set an alarm for six |  | 72 | 2 | 0.027777777777777776 | ALARM=2 |
| set color |  | 80 | 0 | 0.0 | - |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 38 | 0.475 | STOP=38 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 50 | 0.625 | STOP=50 |
| show my |  | 80 | 23 | 0.2875 | PAUSE=2, STOP=21 |
| shut |  | 80 | 53 | 0.6625 | STOP=53 |
| shut off |  | 80 | 2 | 0.025 | STOP=2 |
| shut off the |  | 80 | 0 | 0.0 | - |
| skip |  | 80 | 43 | 0.5375 | STOP=43 |
| start |  | 80 | 61 | 0.7625 | STOP=61 |
| start a |  | 80 | 58 | 0.725 | STOP=58 |
| start a timer |  | 80 | 2 | 0.025 | PAUSE=1, STOP=1 |
| start a timer for |  | 80 | 0 | 0.0 | - |
| start a timer for one |  | 80 | 1 | 0.0125 | TIMER=1 |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 22 | 0.275 | STOP=22 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 51 | 0.6375 | CALL=6, STOP=45 |
| tell me |  | 80 | 17 | 0.2125 | STOP=1, TIME=16 |
| tell me the |  | 80 | 4 | 0.05 | PLAY_MUSIC=1, TIME=3 |
| temperature |  | 80 | 5 | 0.0625 | PAUSE=4, TIME=1 |
| temperature eighteen |  | 80 | 0 | 0.0 | - |
| temperature twenty |  | 80 | 1 | 0.0125 | TIME=1 |
| temperature twenty six |  | 80 | 0 | 0.0 | - |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 63 | 0.7875 | CALL=7, STOP=12, TIME=42, VOLUME_UP=2 |
| timer one |  | 64 | 4 | 0.0625 | TIME=4 |
| timer ten |  | 80 | 21 | 0.2625 | PLAY_MUSIC=1, STOP=3, TIME=15, VOLUME_UP=2 |
| timer thirty |  | 80 | 20 | 0.25 | TIME=20 |
| turn |  | 80 | 45 | 0.5625 | STOP=21, TIME=24 |
| turn on |  | 80 | 25 | 0.3125 | STOP=22, TIME=3 |
| turn on the |  | 80 | 15 | 0.1875 | STOP=11, TIME=3, VOLUME_UP=1 |
| turn the |  | 80 | 33 | 0.4125 | STOP=4, TIME=29 |
| turn the volume |  | 80 | 7 | 0.0875 | VOLUME_UP=7 |
| volume |  | 80 | 36 | 0.45 | CALL=15, PAUSE=19, STOP=1, VOLUME_UP=1 |
| wake |  | 80 | 9 | 0.1125 | LIGHT_OFF=1, LIGHT_ON=4, WEATHER=4 |
| wake me |  | 80 | 9 | 0.1125 | LIGHT_ON=3, PLAY_MUSIC=2, VOLUME_UP=2, WEATHER=2 |
| wake me up |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| wake me up at |  | 80 | 0 | 0.0 | - |
| wake me up at eight |  | 80 | 10 | 0.125 | ALARM=10 |
| wake me up at nine |  | 72 | 8 | 0.1111111111111111 | ALARM=8 |
| wake me up at six |  | 80 | 1 | 0.0125 | ALARM=1 |
| what | YES | 80 | 20 | 0.25 | STOP=12, WEATHER=8 |
| what time |  | 80 | 11 | 0.1375 | LIGHT_OFF=2, LIGHT_ON=4, STOP=4, WEATHER=1 |
| what time is |  | 80 | 13 | 0.1625 | LIGHT_OFF=1, TIME=12 |
| what's |  | 80 | 39 | 0.4875 | MESSAGE=1, STOP=3, WEATHER=35 |
| what's the |  | 80 | 5 | 0.0625 | LIGHT_ON=2, NEXT=1, PAUSE=1, STOP=1 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2672 | 34 | 0.012724550898203593 |
| lead_in_room_tone | 798 | 119 | 0.14912280701754385 |
| tail_room_tone | 3828 | 650 | 0.1698014629049112 |
| digital_zero | 3390 | 763 | 0.22507374631268437 |

FRR (single-word commands): 0/130 (0.0)
FRR (strict-prefix commands): 0/50 (0.0)

Gate-caused FRR: 0/1889 confidence-accepted target rows additionally rejected (0.0)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|

Babble/silence FAR:
- babble: 3/255 (0.011764705882352941)
- silence: 5/324 (0.015432098765432098)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| [0, 0.2)s | 332 | 327 | 0.9849397590361446 |
| [0.2, 0.5)s | 1256 | 1240 | 0.9872611464968153 |
| [0.5, 1)s | 194 | 189 | 0.9742268041237113 |
| [1, 2)s | 16 | 15 | 0.9375 |
| unknown | 129 | 118 | 0.9147286821705426 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2672 | 813 | 0.30426646706586824 |
| [0, 0.2)s | 2672 | 34 | 0.012724550898203593 |
| [0.2, 0.5)s | 2672 | 204 | 0.07634730538922156 |
| [1, 2)s | 2672 | 515 | 0.19273952095808383 |

## margin=6.0

Target exact accuracy: 1870/1927 (0.9704203425012974)

Slot exact match (given intent correct): 1052/1052 (1.0), over 1068 slot-bearing target rows (0 unparseable ground truth)

Incomplete-prefix FAR: 690/10688 (0.06455838323353294)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 1 | 0.0125 | STOP=1 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 4 | 0.05 | PAUSE=3, STOP=1 |
| alarm eight |  | 64 | 0 | 0.0 | - |
| alarm nine |  | 80 | 1 | 0.0125 | VOLUME_DOWN=1 |
| alarm six |  | 80 | 0 | 0.0 | - |
| brightness |  | 80 | 5 | 0.0625 | TIME=5 |
| brightness level |  | 80 | 0 | 0.0 | - |
| brightness level one |  | 80 | 0 | 0.0 | - |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 0 | 0.0 | - |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |
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
| create |  | 80 | 2 | 0.025 | CALL=2 |
| create a |  | 80 | 0 | 0.0 | - |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 0 | 0.0 | - |
| end |  | 72 | 25 | 0.3472222222222222 | STOP=25 |
| increase |  | 80 | 0 | 0.0 | - |
| increase the |  | 80 | 0 | 0.0 | - |
| kill |  | 80 | 32 | 0.4 | STOP=32 |
| kill the |  | 80 | 7 | 0.0875 | STOP=7 |
| lights |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |
| list |  | 80 | 2 | 0.025 | STOP=1, WEATHER=1 |
| list my |  | 80 | 4 | 0.05 | LIGHT_OFF=2, PAUSE=1, PLAY_MUSIC=1 |
| lower |  | 72 | 16 | 0.2222222222222222 | PAUSE=1, STOP=15 |
| lower the |  | 72 | 0 | 0.0 | - |
| make |  | 80 | 1 | 0.0125 | STOP=1 |
| make a |  | 80 | 1 | 0.0125 | STOP=1 |
| make a phone |  | 80 | 0 | 0.0 | - |
| next |  | 80 | 3 | 0.0375 | STOP=3 |
| pause for |  | 72 | 32 | 0.4444444444444444 | PAUSE=32 |
| place |  | 80 | 1 | 0.0125 | STOP=1 |
| place a |  | 80 | 1 | 0.0125 | STOP=1 |
| play |  | 80 | 0 | 0.0 | - |
| play next |  | 80 | 0 | 0.0 | - |
| play some |  | 80 | 6 | 0.075 | PLAY_MUSIC=6 |
| power |  | 72 | 39 | 0.5416666666666666 | CALL=6, STOP=33 |
| power on |  | 72 | 2 | 0.027777777777777776 | STOP=2 |
| power on the |  | 72 | 0 | 0.0 | - |
| remind | YES | 80 | 0 | 0.0 | - |
| remind me |  | 80 | 0 | 0.0 | - |
| remind me to |  | 80 | 0 | 0.0 | - |
| remind me to drink |  | 80 | 0 | 0.0 | - |
| reminder | YES | 80 | 0 | 0.0 | - |
| reminder drink |  | 80 | 0 | 0.0 | - |
| send |  | 80 | 24 | 0.3 | STOP=24 |
| send a |  | 72 | 11 | 0.1527777777777778 | STOP=11 |
| send my |  | 80 | 1 | 0.0125 | PAUSE=1 |
| set |  | 80 | 19 | 0.2375 | STOP=19 |
| set an |  | 80 | 7 | 0.0875 | STOP=7 |
| set an alarm |  | 80 | 0 | 0.0 | - |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 0 | 0.0 | - |
| set an alarm for nine |  | 80 | 0 | 0.0 | - |
| set an alarm for six |  | 72 | 0 | 0.0 | - |
| set color |  | 80 | 0 | 0.0 | - |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 6 | 0.075 | STOP=6 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 42 | 0.525 | STOP=42 |
| show my |  | 80 | 0 | 0.0 | - |
| shut |  | 80 | 41 | 0.5125 | STOP=41 |
| shut off |  | 80 | 0 | 0.0 | - |
| shut off the |  | 80 | 0 | 0.0 | - |
| skip |  | 80 | 28 | 0.35 | STOP=28 |
| start |  | 80 | 55 | 0.6875 | STOP=55 |
| start a |  | 80 | 46 | 0.575 | STOP=46 |
| start a timer |  | 80 | 0 | 0.0 | - |
| start a timer for |  | 80 | 0 | 0.0 | - |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 1 | 0.0125 | STOP=1 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 48 | 0.6 | CALL=6, STOP=42 |
| tell me |  | 80 | 5 | 0.0625 | TIME=5 |
| tell me the |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| temperature |  | 80 | 2 | 0.025 | PAUSE=1, TIME=1 |
| temperature eighteen |  | 80 | 0 | 0.0 | - |
| temperature twenty |  | 80 | 1 | 0.0125 | TIME=1 |
| temperature twenty six |  | 80 | 0 | 0.0 | - |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 34 | 0.425 | CALL=3, STOP=10, TIME=20, VOLUME_UP=1 |
| timer one |  | 64 | 0 | 0.0 | - |
| timer ten |  | 80 | 2 | 0.025 | PLAY_MUSIC=1, VOLUME_UP=1 |
| timer thirty |  | 80 | 0 | 0.0 | - |
| turn |  | 80 | 21 | 0.2625 | STOP=13, TIME=8 |
| turn on |  | 80 | 10 | 0.125 | STOP=10 |
| turn on the |  | 80 | 5 | 0.0625 | STOP=4, VOLUME_UP=1 |
| turn the |  | 80 | 4 | 0.05 | TIME=4 |
| turn the volume |  | 80 | 0 | 0.0 | - |
| volume |  | 80 | 20 | 0.25 | CALL=8, PAUSE=12 |
| wake |  | 80 | 0 | 0.0 | - |
| wake me |  | 80 | 4 | 0.05 | LIGHT_ON=1, VOLUME_UP=2, WEATHER=1 |
| wake me up |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| wake me up at |  | 80 | 0 | 0.0 | - |
| wake me up at eight |  | 80 | 0 | 0.0 | - |
| wake me up at nine |  | 72 | 0 | 0.0 | - |
| wake me up at six |  | 80 | 0 | 0.0 | - |
| what | YES | 80 | 11 | 0.1375 | STOP=5, WEATHER=6 |
| what time |  | 80 | 7 | 0.0875 | LIGHT_OFF=1, LIGHT_ON=2, STOP=4 |
| what time is |  | 80 | 3 | 0.0375 | TIME=3 |
| what's |  | 80 | 30 | 0.375 | MESSAGE=1, STOP=2, WEATHER=27 |
| what's the |  | 80 | 3 | 0.0375 | LIGHT_ON=2, NEXT=1 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2672 | 8 | 0.0029940119760479044 |
| lead_in_room_tone | 798 | 34 | 0.042606516290726815 |
| tail_room_tone | 3828 | 274 | 0.07157784743991641 |
| digital_zero | 3390 | 374 | 0.11032448377581121 |

FRR (single-word commands): 5/130 (0.038461538461538464)
FRR (strict-prefix commands): 0/50 (0.0)

Gate-caused FRR: 18/1889 confidence-accepted target rows additionally rejected (0.009528851244044468)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|
| alarm eight |  | 1 | ALARM=1 |
| create a reminder to drink |  | 1 | CREATE_REMINDER=1 |
| place a |  | 1 | CALL=1 |
| play |  | 1 | PLAY_MUSIC=1 |
| remind me to |  | 2 | CREATE_REMINDER=2 |
| reminder | YES | 7 | LIST_REMINDERS=7 |
| set |  | 1 | CALL=1 |
| set color to |  | 1 | COLOR=1 |
| turn the volume |  | 1 | VOLUME_DOWN=1 |
| volume |  | 1 | VOLUME_UP=1 |
| wake me up at |  | 1 | ALARM=1 |

Babble/silence FAR:
- babble: 1/255 (0.00392156862745098)
- silence: 1/324 (0.0030864197530864196)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| [0, 0.2)s | 332 | 323 | 0.9728915662650602 |
| [0.2, 0.5)s | 1256 | 1234 | 0.982484076433121 |
| [0.5, 1)s | 194 | 184 | 0.9484536082474226 |
| [1, 2)s | 16 | 15 | 0.9375 |
| unknown | 129 | 115 | 0.8914728682170543 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2672 | 337 | 0.12612275449101795 |
| [0, 0.2)s | 2672 | 8 | 0.0029940119760479044 |
| [0.2, 0.5)s | 2672 | 111 | 0.04154191616766467 |
| [1, 2)s | 2672 | 234 | 0.0875748502994012 |

## Checkpoint/ONNX hashes

Before: {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
After:  {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
Unchanged: True

