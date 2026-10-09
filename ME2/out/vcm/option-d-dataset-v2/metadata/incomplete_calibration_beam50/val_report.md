# Incomplete-prefix rejection gate -- val calibration report

Checkpoint: `out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt`  Beam width: 50
Chosen threshold (tau): -0.1

Chosen margin: **6.0** (val target_exact_accuracy=0.9523, val incomplete_prefix_far_excl_digital_zero=0.0373 [selection criterion], val incomplete_prefix_far_all_probes=0.0664)

## Margin grid

Selection rule: Among margins whose val target exact-accuracy drop from the gate-disabled baseline is at most the regression budget, pick the lowest incomplete_prefix_far_excl_digital_zero (NOT incomplete_prefix_far, which includes digital-zero-padded probes); ties break toward the smallest margin. R2-6: digital-zero padding is not acoustically neutral and was found to understate the false-accept rate, biasing selection toward too small a margin -- excluded from the selection criterion. incomplete_prefix_far (all probes) and incomplete_prefix_far_digital_zero_only are still reported on every grid row for visibility. (regression_budget_pp=1.0)

| margin | target_exact_accuracy | far_excl_digital_zero (selection) | far_all_probes | far_digital_zero_only | gate_caused_frr |
|---|---|---|---|---|---|
| baseline (gate off) | 0.9622 | 0.1080 | 0.1538 | 0.2352 | 0.0 |
| -10.0 | 0.9622 | 0.0871 | 0.1347 | 0.2190 | 0.0 |
| -5.0 | 0.9622 | 0.0737 | 0.1202 | 0.2029 | 0.0 |
| -3.0 | 0.9613 | 0.0672 | 0.1118 | 0.1909 | 0.0010303967027305513 |
| -2.0 | 0.9608 | 0.0643 | 0.1077 | 0.1849 | 0.0015455950540958269 |
| -1.0 | 0.9593 | 0.0603 | 0.1025 | 0.1773 | 0.0030911901081916537 |
| -0.5 | 0.9588 | 0.0587 | 0.1003 | 0.1742 | 0.0036063884595569293 |
| 0.0 | 0.9578 | 0.0569 | 0.0976 | 0.1698 | 0.00463678516228748 |
| 0.5 | 0.9573 | 0.0549 | 0.0944 | 0.1646 | 0.005151983513652756 |
| 1.0 | 0.9573 | 0.0528 | 0.0921 | 0.1617 | 0.005151983513652756 |
| 2.0 | 0.9568 | 0.0499 | 0.0871 | 0.1531 | 0.005667181865018032 |
| 3.0 | 0.9563 | 0.0464 | 0.0819 | 0.1451 | 0.0061823802163833074 |
| 4.0 | 0.9558 | 0.0433 | 0.0770 | 0.1367 | 0.006697578567748583 |
| 4.5 | 0.9553 | 0.0414 | 0.0740 | 0.1320 | 0.0072127769191138585 |
| 5.0 | 0.9538 | 0.0405 | 0.0717 | 0.1271 | 0.008758371973209686 |
| 6.0 | 0.9523 | 0.0373 | 0.0664 | 0.1182 | 0.010303967027305513 |
| 7.0 | 0.9498 | 0.0343 | 0.0613 | 0.1091 | 0.01287995878413189 |
| 10.0 | 0.9374 | 0.0270 | 0.0497 | 0.0901 | 0.02627511591962906 |
| 15.0 | 0.9265 | 0.0167 | 0.0354 | 0.0685 | 0.03760947964966512 |
| 20.0 | 0.9131 | 0.0110 | 0.0254 | 0.0510 | 0.05151983513652756 |
| 30.0 | 0.8847 | 0.0054 | 0.0157 | 0.0339 | 0.08140133951571354 |
| 50.0 | 0.8132 | 0.0023 | 0.0093 | 0.0216 | 0.1561051004636785 |

## Baseline vs. chosen-margin metrics

### baseline (gate off)

Target exact accuracy: 1937/2013 (0.9622454048683556)

Slot exact match (given intent correct): 1014/1015 (0.9990147783251232), over 1068 slot-bearing target rows (0 unparseable ground truth)

Incomplete-prefix FAR: 1639/10656 (0.15381006006006007)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 6 | 0.075 | PAUSE=2, STOP=4 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 13 | 0.1625 | CALL=1, PAUSE=11, TIME=1 |
| alarm eight |  | 80 | 20 | 0.25 | ALARM=14, CALL=2, PAUSE=4 |
| alarm nine |  | 80 | 7 | 0.0875 | ALARM=4, CALL=1, VOLUME_UP=2 |
| alarm six |  | 72 | 5 | 0.06944444444444445 | PAUSE=4, PLAY_MUSIC=1 |
| brightness |  | 80 | 7 | 0.0875 | STOP=2, TIME=5 |
| brightness level |  | 80 | 5 | 0.0625 | LIGHT_OFF=3, LIGHT_ON=2 |
| brightness level one |  | 80 | 1 | 0.0125 | LIGHT_OFF=1 |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 4 | 0.05 | PLAY_MUSIC=4 |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 15 | 0.1875 | LIGHT_OFF=5, LIGHT_ON=2, PAUSE=7, TIME=1 |
| brightness one hundred |  | 80 | 11 | 0.1375 | LIGHT_ON=3, PAUSE=8 |
| brightness sixty |  | 80 | 2 | 0.025 | LIGHT_ON=1, PLAY_MUSIC=1 |
| brightness twenty |  | 80 | 3 | 0.0375 | TIME=2, VOLUME_UP=1 |
| change |  | 80 | 23 | 0.2875 | PAUSE=1, STOP=22 |
| change color |  | 80 | 2 | 0.025 | PAUSE=2 |
| change color to |  | 80 | 1 | 0.0125 | PAUSE=1 |
| change the |  | 80 | 13 | 0.1625 | PAUSE=6, PLAY_MUSIC=7 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 8 | 0.1 | STOP=8 |
| countdown for |  | 80 | 3 | 0.0375 | PAUSE=2, STOP=1 |
| countdown for one |  | 72 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 1 | 0.0125 | VOLUME_DOWN=1 |
| create |  | 80 | 3 | 0.0375 | CALL=3 |
| create a |  | 80 | 9 | 0.1125 | CALL=9 |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 2 | 0.025 | LIST_REMINDERS=2 |
| create a reminder to drink |  | 80 | 4 | 0.05 | CREATE_REMINDER=2, PLAY_MUSIC=2 |
| end |  | 72 | 32 | 0.4444444444444444 | STOP=32 |
| increase |  | 80 | 2 | 0.025 | PAUSE=2 |
| increase the |  | 80 | 2 | 0.025 | STOP=2 |
| kill |  | 80 | 46 | 0.575 | STOP=44, TIME=2 |
| kill the |  | 80 | 21 | 0.2625 | CALL=7, STOP=13, TIME=1 |
| lights |  | 80 | 15 | 0.1875 | LIGHT_ON=13, TIME=2 |
| list |  | 80 | 8 | 0.1 | STOP=8 |
| list my |  | 80 | 19 | 0.2375 | LIGHT_OFF=7, LIGHT_ON=5, STOP=7 |
| lower |  | 80 | 30 | 0.375 | STOP=30 |
| lower the |  | 80 | 9 | 0.1125 | PAUSE=1, STOP=3, WEATHER=5 |
| make |  | 72 | 7 | 0.09722222222222222 | STOP=4, TIME=3 |
| make a |  | 80 | 4 | 0.05 | NEXT=1, STOP=2, TIME=1 |
| make a phone |  | 80 | 3 | 0.0375 | LIGHT_ON=2, PLAY_MUSIC=1 |
| next |  | 80 | 16 | 0.2 | CALL=2, NEXT=2, PAUSE=4, STOP=8 |
| pause for |  | 80 | 40 | 0.5 | PAUSE=40 |
| place |  | 72 | 3 | 0.041666666666666664 | CALL=1, STOP=2 |
| place a |  | 72 | 2 | 0.027777777777777776 | PAUSE=2 |
| play |  | 80 | 9 | 0.1125 | CALL=4, PAUSE=1, STOP=4 |
| play next |  | 80 | 7 | 0.0875 | CALL=3, PAUSE=4 |
| play some |  | 80 | 0 | 0.0 | - |
| power |  | 80 | 47 | 0.5875 | CALL=1, STOP=46 |
| power on |  | 80 | 23 | 0.2875 | CALL=4, PAUSE=8, STOP=8, TIME=3 |
| power on the |  | 80 | 21 | 0.2625 | PAUSE=12, STOP=3, TIME=2, VOLUME_DOWN=4 |
| remind | YES | 80 | 16 | 0.2 | LIST_REMINDERS=11, STOP=3, TIME=1, WEATHER=1 |
| remind me |  | 80 | 23 | 0.2875 | LIST_REMINDERS=21, TIME=2 |
| remind me to |  | 80 | 22 | 0.275 | LIST_REMINDERS=20, TIME=1, WEATHER=1 |
| remind me to drink |  | 80 | 1 | 0.0125 | CREATE_REMINDER=1 |
| reminder | YES | 80 | 25 | 0.3125 | LIST_REMINDERS=22, STOP=1, TIME=2 |
| reminder drink |  | 72 | 28 | 0.3888888888888889 | LIST_REMINDERS=26, TIME=1, VOLUME_DOWN=1 |
| send |  | 80 | 34 | 0.425 | STOP=34 |
| send a |  | 80 | 36 | 0.45 | PAUSE=9, STOP=27 |
| send my |  | 80 | 10 | 0.125 | PAUSE=2, STOP=8 |
| set |  | 80 | 40 | 0.5 | PAUSE=1, STOP=39 |
| set an |  | 80 | 40 | 0.5 | PLAY_MUSIC=2, STOP=37, TIME=1 |
| set an alarm |  | 80 | 5 | 0.0625 | STOP=5 |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 25 | 0.3125 | ALARM=25 |
| set an alarm for nine |  | 72 | 5 | 0.06944444444444445 | ALARM=5 |
| set an alarm for six |  | 80 | 0 | 0.0 | - |
| set color |  | 80 | 5 | 0.0625 | STOP=5 |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 21 | 0.2625 | STOP=21 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 1 | 0.0125 | STOP=1 |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 46 | 0.575 | STOP=46 |
| show my |  | 80 | 28 | 0.35 | STOP=28 |
| shut |  | 72 | 38 | 0.5277777777777778 | STOP=38 |
| shut off |  | 72 | 3 | 0.041666666666666664 | STOP=3 |
| shut off the |  | 72 | 0 | 0.0 | - |
| skip |  | 80 | 39 | 0.4875 | STOP=39 |
| start |  | 80 | 59 | 0.7375 | STOP=59 |
| start a |  | 80 | 53 | 0.6625 | STOP=53 |
| start a timer |  | 80 | 8 | 0.1 | PAUSE=5, STOP=3 |
| start a timer for |  | 80 | 2 | 0.025 | PAUSE=1, TIME=1 |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 15 | 0.1875 | PAUSE=2, STOP=13 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 42 | 0.525 | STOP=41, TIME=1 |
| tell me |  | 80 | 23 | 0.2875 | STOP=11, TIME=12 |
| tell me the |  | 80 | 12 | 0.15 | CALL=1, PLAY_MUSIC=1, STOP=3, TIME=7 |
| temperature |  | 80 | 8 | 0.1 | PAUSE=7, PLAY_MUSIC=1 |
| temperature eighteen |  | 80 | 2 | 0.025 | STOP=2 |
| temperature twenty |  | 80 | 1 | 0.0125 | PAUSE=1 |
| temperature twenty six |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| temperature twenty two |  | 80 | 1 | 0.0125 | PAUSE=1 |
| timer |  | 80 | 58 | 0.725 | CALL=2, STOP=13, TIME=43 |
| timer one |  | 80 | 7 | 0.0875 | PAUSE=1, TIME=6 |
| timer ten |  | 80 | 17 | 0.2125 | PAUSE=3, STOP=1, TIME=13 |
| timer thirty |  | 72 | 25 | 0.3472222222222222 | PAUSE=1, TIME=24 |
| turn |  | 80 | 30 | 0.375 | STOP=23, TIME=7 |
| turn on |  | 80 | 31 | 0.3875 | STOP=29, TIME=2 |
| turn on the |  | 80 | 19 | 0.2375 | CALL=2, STOP=17 |
| turn the |  | 80 | 26 | 0.325 | STOP=4, TIME=22 |
| turn the volume |  | 80 | 5 | 0.0625 | VOLUME_UP=5 |
| volume |  | 80 | 30 | 0.375 | CALL=9, PAUSE=7, STOP=6, TIME=4, VOLUME_UP=4 |
| wake |  | 80 | 12 | 0.15 | STOP=1, WEATHER=11 |
| wake me |  | 80 | 12 | 0.15 | LIGHT_ON=2, PAUSE=2, VOLUME_UP=2, WEATHER=6 |
| wake me up |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| wake me up at |  | 80 | 5 | 0.0625 | PAUSE=2, PLAY_MUSIC=3 |
| wake me up at eight |  | 80 | 21 | 0.2625 | ALARM=21 |
| wake me up at nine |  | 80 | 8 | 0.1 | ALARM=7, LIGHT_ON=1 |
| wake me up at six |  | 80 | 5 | 0.0625 | ALARM=1, LIGHT_OFF=2, PLAY_MUSIC=2 |
| what | YES | 64 | 20 | 0.3125 | STOP=6, WEATHER=14 |
| what time |  | 72 | 7 | 0.09722222222222222 | LIGHT_OFF=1, LIGHT_ON=1, STOP=1, WEATHER=4 |
| what time is |  | 72 | 12 | 0.16666666666666666 | TIME=12 |
| what's |  | 72 | 32 | 0.4444444444444444 | LIGHT_ON=1, MESSAGE=2, PAUSE=3, TIME=1, WEATHER=25 |
| what's the |  | 72 | 4 | 0.05555555555555555 | LIGHT_ON=4 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2664 | 30 | 0.01126126126126126 |
| lead_in_room_tone | 654 | 130 | 0.19877675840978593 |
| tail_room_tone | 3498 | 576 | 0.1646655231560892 |
| digital_zero | 3840 | 903 | 0.23515625 |

FRR (single-word commands): 0/137 (0.0)
FRR (strict-prefix commands): 0/51 (0.0)

Gate-caused FRR: 0/1941 confidence-accepted target rows additionally rejected (0.0)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|

Babble/silence FAR:
- babble: 12/409 (0.029339853300733496)
- silence: 1/329 (0.00303951367781155)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 6 | 6 | 1.0 |
| [0, 0.2)s | 372 | 350 | 0.9408602150537635 |
| [0.2, 0.5)s | 1225 | 1181 | 0.9640816326530612 |
| [0.5, 1)s | 209 | 206 | 0.9856459330143541 |
| [1, 2)s | 6 | 6 | 1.0 |
| unknown | 195 | 192 | 0.9846153846153847 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2664 | 876 | 0.32882882882882886 |
| [0, 0.2)s | 2664 | 30 | 0.01126126126126126 |
| [0.2, 0.5)s | 2664 | 199 | 0.0746996996996997 |
| [1, 2)s | 2664 | 534 | 0.20045045045045046 |

### margin=6.0

Target exact accuracy: 1917/2013 (0.9523099850968704)

Slot exact match (given intent correct): 1011/1011 (1.0), over 1068 slot-bearing target rows (0 unparseable ground truth)

Incomplete-prefix FAR: 708/10656 (0.06644144144144144)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 4 | 0.05 | STOP=4 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 4 | 0.05 | PAUSE=4 |
| alarm eight |  | 80 | 2 | 0.025 | PAUSE=2 |
| alarm nine |  | 80 | 2 | 0.025 | VOLUME_UP=2 |
| alarm six |  | 72 | 5 | 0.06944444444444445 | PAUSE=4, PLAY_MUSIC=1 |
| brightness |  | 80 | 2 | 0.025 | STOP=1, TIME=1 |
| brightness level |  | 80 | 4 | 0.05 | LIGHT_OFF=2, LIGHT_ON=2 |
| brightness level one |  | 80 | 1 | 0.0125 | LIGHT_OFF=1 |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 4 | 0.05 | PLAY_MUSIC=4 |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 7 | 0.0875 | PAUSE=7 |
| brightness one hundred |  | 80 | 9 | 0.1125 | LIGHT_ON=1, PAUSE=8 |
| brightness sixty |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| brightness twenty |  | 80 | 0 | 0.0 | - |
| change |  | 80 | 8 | 0.1 | STOP=8 |
| change color |  | 80 | 0 | 0.0 | - |
| change color to |  | 80 | 0 | 0.0 | - |
| change the |  | 80 | 9 | 0.1125 | PAUSE=3, PLAY_MUSIC=6 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 3 | 0.0375 | STOP=3 |
| countdown for |  | 80 | 2 | 0.025 | PAUSE=2 |
| countdown for one |  | 72 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 0 | 0.0 | - |
| create |  | 80 | 0 | 0.0 | - |
| create a |  | 80 | 1 | 0.0125 | CALL=1 |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 2 | 0.025 | PLAY_MUSIC=2 |
| end |  | 72 | 16 | 0.2222222222222222 | STOP=16 |
| increase |  | 80 | 0 | 0.0 | - |
| increase the |  | 80 | 0 | 0.0 | - |
| kill |  | 80 | 32 | 0.4 | STOP=32 |
| kill the |  | 80 | 1 | 0.0125 | STOP=1 |
| lights |  | 80 | 2 | 0.025 | LIGHT_ON=2 |
| list |  | 80 | 3 | 0.0375 | STOP=3 |
| list my |  | 80 | 5 | 0.0625 | LIGHT_OFF=2, LIGHT_ON=3 |
| lower |  | 80 | 26 | 0.325 | STOP=26 |
| lower the |  | 80 | 0 | 0.0 | - |
| make |  | 72 | 1 | 0.013888888888888888 | STOP=1 |
| make a |  | 80 | 0 | 0.0 | - |
| make a phone |  | 80 | 2 | 0.025 | LIGHT_ON=2 |
| next |  | 80 | 8 | 0.1 | CALL=2, NEXT=1, PAUSE=4, STOP=1 |
| pause for |  | 80 | 21 | 0.2625 | PAUSE=21 |
| place |  | 72 | 0 | 0.0 | - |
| place a |  | 72 | 0 | 0.0 | - |
| play |  | 80 | 1 | 0.0125 | CALL=1 |
| play next |  | 80 | 0 | 0.0 | - |
| play some |  | 80 | 0 | 0.0 | - |
| power |  | 80 | 33 | 0.4125 | STOP=33 |
| power on |  | 80 | 8 | 0.1 | CALL=2, PAUSE=4, STOP=2 |
| power on the |  | 80 | 14 | 0.175 | PAUSE=10, TIME=1, VOLUME_DOWN=3 |
| remind | YES | 80 | 3 | 0.0375 | STOP=3 |
| remind me |  | 80 | 0 | 0.0 | - |
| remind me to |  | 80 | 0 | 0.0 | - |
| remind me to drink |  | 80 | 0 | 0.0 | - |
| reminder | YES | 80 | 0 | 0.0 | - |
| reminder drink |  | 72 | 0 | 0.0 | - |
| send |  | 80 | 21 | 0.2625 | STOP=21 |
| send a |  | 80 | 22 | 0.275 | PAUSE=7, STOP=15 |
| send my |  | 80 | 0 | 0.0 | - |
| set |  | 80 | 26 | 0.325 | STOP=26 |
| set an |  | 80 | 16 | 0.2 | STOP=16 |
| set an alarm |  | 80 | 1 | 0.0125 | STOP=1 |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 0 | 0.0 | - |
| set an alarm for nine |  | 72 | 0 | 0.0 | - |
| set an alarm for six |  | 80 | 0 | 0.0 | - |
| set color |  | 80 | 0 | 0.0 | - |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 6 | 0.075 | STOP=6 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 35 | 0.4375 | STOP=35 |
| show my |  | 80 | 4 | 0.05 | STOP=4 |
| shut |  | 72 | 25 | 0.3472222222222222 | STOP=25 |
| shut off |  | 72 | 0 | 0.0 | - |
| shut off the |  | 72 | 0 | 0.0 | - |
| skip |  | 80 | 23 | 0.2875 | STOP=23 |
| start |  | 80 | 56 | 0.7 | STOP=56 |
| start a |  | 80 | 45 | 0.5625 | STOP=45 |
| start a timer |  | 80 | 3 | 0.0375 | PAUSE=3 |
| start a timer for |  | 80 | 1 | 0.0125 | PAUSE=1 |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 0 | 0.0 | - |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 39 | 0.4875 | STOP=38, TIME=1 |
| tell me |  | 80 | 12 | 0.15 | STOP=6, TIME=6 |
| tell me the |  | 80 | 7 | 0.0875 | PLAY_MUSIC=1, STOP=3, TIME=3 |
| temperature |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| temperature eighteen |  | 80 | 0 | 0.0 | - |
| temperature twenty |  | 80 | 0 | 0.0 | - |
| temperature twenty six |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 25 | 0.3125 | CALL=2, STOP=9, TIME=14 |
| timer one |  | 80 | 0 | 0.0 | - |
| timer ten |  | 80 | 0 | 0.0 | - |
| timer thirty |  | 72 | 2 | 0.027777777777777776 | TIME=2 |
| turn |  | 80 | 19 | 0.2375 | STOP=13, TIME=6 |
| turn on |  | 80 | 11 | 0.1375 | STOP=11 |
| turn on the |  | 80 | 5 | 0.0625 | STOP=5 |
| turn the |  | 80 | 3 | 0.0375 | STOP=1, TIME=2 |
| turn the volume |  | 80 | 0 | 0.0 | - |
| volume |  | 80 | 9 | 0.1125 | CALL=2, PAUSE=3, STOP=4 |
| wake |  | 80 | 6 | 0.075 | WEATHER=6 |
| wake me |  | 80 | 4 | 0.05 | WEATHER=4 |
| wake me up |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| wake me up at |  | 80 | 2 | 0.025 | PLAY_MUSIC=2 |
| wake me up at eight |  | 80 | 0 | 0.0 | - |
| wake me up at nine |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |
| wake me up at six |  | 80 | 2 | 0.025 | PLAY_MUSIC=2 |
| what | YES | 64 | 7 | 0.109375 | WEATHER=7 |
| what time |  | 72 | 2 | 0.027777777777777776 | WEATHER=2 |
| what time is |  | 72 | 0 | 0.0 | - |
| what's |  | 72 | 16 | 0.2222222222222222 | WEATHER=16 |
| what's the |  | 72 | 3 | 0.041666666666666664 | LIGHT_ON=3 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2664 | 4 | 0.0015015015015015015 |
| lead_in_room_tone | 654 | 39 | 0.05963302752293578 |
| tail_room_tone | 3498 | 211 | 0.06032018296169239 |
| digital_zero | 3840 | 454 | 0.11822916666666666 |

FRR (single-word commands): 11/137 (0.08029197080291971)
FRR (strict-prefix commands): 1/51 (0.0196078431372549)

Gate-caused FRR: 20/1941 confidence-accepted target rows additionally rejected (0.010303967027305513)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|
| alarm |  | 1 | ALARM=1 |
| change color to |  | 1 | COLOR=1 |
| play |  | 1 | PLAY_MUSIC=1 |
| reminder | YES | 9 | LIST_REMINDERS=9 |
| set |  | 1 | PAUSE=1 |
| timer |  | 1 | TIME=1 |
| timer one |  | 2 | TIMER=2 |
| turn the volume |  | 1 | VOLUME_UP=1 |
| volume |  | 3 | VOLUME_UP=3 |

Babble/silence FAR:
- babble: 1/409 (0.0024449877750611247)
- silence: 0/329 (0.0)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 6 | 4 | 0.6666666666666666 |
| [0, 0.2)s | 372 | 344 | 0.9247311827956989 |
| [0.2, 0.5)s | 1225 | 1172 | 0.9567346938775511 |
| [0.5, 1)s | 209 | 206 | 0.9856459330143541 |
| [1, 2)s | 6 | 6 | 1.0 |
| unknown | 195 | 189 | 0.9692307692307692 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2664 | 362 | 0.13588588588588588 |
| [0, 0.2)s | 2664 | 4 | 0.0015015015015015015 |
| [0.2, 0.5)s | 2664 | 94 | 0.03528528528528529 |
| [1, 2)s | 2664 | 248 | 0.09309309309309309 |

## Decode latency/peak memory: gate off vs. chosen margin

Hardware: ai-n002.hpc.coe.upd.edu.ph (measured on this node)

| config | n | median latency (ms) | p95 latency (ms) | peak memory (bytes) |
|---|---|---|---|---|
| gate off | 50 | 239.0641886740923 | 322.8903179988265 | 46411 |
| margin=6.0 | 50 | 237.66062199138105 | 322.941817343235 | 46411 |

## Alignment / probe-generation failures

- target forced-alignment: 0/1818 failed
- probe generation: 6 failures: {'crop_too_short': 6, 'force_align_error': 0, 'missing_audio': 0, 'no_gap': 0}

## Checkpoint/ONNX hashes

Before: {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
After:  {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
Unchanged: True

