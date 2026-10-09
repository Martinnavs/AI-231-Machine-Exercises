# Incomplete-prefix rejection gate -- val calibration report

Checkpoint: `out/vcm/option-d-dataset-v2/checkpoints/checkpoint.pt`  Beam width: 50
Chosen threshold (tau): -0.075

Chosen margin: **4.0** (val target_exact_accuracy=0.9795, val incomplete_prefix_far_excl_digital_zero=0.0403 [selection criterion], val incomplete_prefix_far_all_probes=0.0690)

## Margin grid

Selection rule: Among margins whose val target exact-accuracy drop from the gate-disabled baseline is at most the regression budget, pick the lowest incomplete_prefix_far_excl_digital_zero (NOT incomplete_prefix_far, which includes digital-zero-padded probes); ties break toward the smallest margin. R2-6: digital-zero padding is not acoustically neutral and was found to understate the false-accept rate, biasing selection toward too small a margin -- excluded from the selection criterion. incomplete_prefix_far (all probes) and incomplete_prefix_far_digital_zero_only are still reported on every grid row for visibility. (regression_budget_pp=1.0)

| margin | target_exact_accuracy | far_excl_digital_zero (selection) | far_all_probes | far_digital_zero_only | gate_caused_frr |
|---|---|---|---|---|---|
| baseline (gate off) | 0.9846 | 0.0776 | 0.1137 | 0.1779 | 0.0 |
| -10.0 | 0.9846 | 0.0710 | 0.1076 | 0.1727 | 0.0 |
| -5.0 | 0.9846 | 0.0616 | 0.0985 | 0.1641 | 0.0 |
| -3.0 | 0.9846 | 0.0580 | 0.0942 | 0.1586 | 0.0 |
| -2.0 | 0.9846 | 0.0560 | 0.0910 | 0.1531 | 0.0 |
| -1.0 | 0.9846 | 0.0534 | 0.0876 | 0.1482 | 0.0 |
| -0.5 | 0.9846 | 0.0521 | 0.0859 | 0.1458 | 0.0 |
| 0.0 | 0.9795 | 0.0509 | 0.0842 | 0.1432 | 0.005208333333333333 |
| 0.5 | 0.9795 | 0.0496 | 0.0821 | 0.1398 | 0.005208333333333333 |
| 1.0 | 0.9795 | 0.0480 | 0.0807 | 0.1388 | 0.005208333333333333 |
| 2.0 | 0.9795 | 0.0461 | 0.0772 | 0.1326 | 0.005208333333333333 |
| 3.0 | 0.9795 | 0.0430 | 0.0732 | 0.1268 | 0.005208333333333333 |
| 4.0 | 0.9795 | 0.0403 | 0.0690 | 0.1198 | 0.005208333333333333 |
| 4.5 | 0.9744 | 0.0386 | 0.0669 | 0.1172 | 0.010416666666666666 |
| 5.0 | 0.9692 | 0.0379 | 0.0648 | 0.1128 | 0.015625 |
| 6.0 | 0.9692 | 0.0352 | 0.0607 | 0.1060 | 0.015625 |
| 7.0 | 0.9641 | 0.0326 | 0.0562 | 0.0982 | 0.020833333333333332 |
| 10.0 | 0.9590 | 0.0257 | 0.0461 | 0.0823 | 0.026041666666666668 |
| 15.0 | 0.9487 | 0.0160 | 0.0325 | 0.0617 | 0.036458333333333336 |
| 20.0 | 0.9385 | 0.0106 | 0.0229 | 0.0448 | 0.046875 |
| 30.0 | 0.8718 | 0.0051 | 0.0132 | 0.0276 | 0.11458333333333333 |
| 50.0 | 0.8256 | 0.0021 | 0.0071 | 0.0161 | 0.16145833333333334 |

## Baseline vs. chosen-margin metrics

### baseline (gate off)

Target exact accuracy: 192/195 (0.9846153846153847)

Slot exact match: n/a (no classifiable slot-bearing Option B target row)

Incomplete-prefix FAR: 1212/10656 (0.11373873873873874)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 4 | 0.05 | PAUSE=1, STOP=3 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 7 | 0.0875 | PAUSE=7 |
| alarm eight |  | 80 | 10 | 0.125 | ALARM=6, CALL=1, PAUSE=3 |
| alarm nine |  | 80 | 3 | 0.0375 | ALARM=2, VOLUME_UP=1 |
| alarm six |  | 72 | 4 | 0.05555555555555555 | PAUSE=3, PLAY_MUSIC=1 |
| brightness |  | 80 | 4 | 0.05 | STOP=2, TIME=2 |
| brightness level |  | 80 | 3 | 0.0375 | LIGHT_OFF=2, LIGHT_ON=1 |
| brightness level one |  | 80 | 0 | 0.0 | - |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 3 | 0.0375 | PLAY_MUSIC=3 |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 12 | 0.15 | LIGHT_OFF=4, PAUSE=7, TIME=1 |
| brightness one hundred |  | 80 | 10 | 0.125 | LIGHT_ON=3, PAUSE=7 |
| brightness sixty |  | 80 | 0 | 0.0 | - |
| brightness twenty |  | 80 | 2 | 0.025 | TIME=1, VOLUME_UP=1 |
| change |  | 80 | 17 | 0.2125 | PAUSE=1, STOP=16 |
| change color |  | 80 | 2 | 0.025 | PAUSE=2 |
| change color to |  | 80 | 0 | 0.0 | - |
| change the |  | 80 | 9 | 0.1125 | PAUSE=3, PLAY_MUSIC=6 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 6 | 0.075 | STOP=6 |
| countdown for |  | 80 | 1 | 0.0125 | PAUSE=1 |
| countdown for one |  | 72 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 1 | 0.0125 | VOLUME_DOWN=1 |
| create |  | 80 | 2 | 0.025 | CALL=2 |
| create a |  | 80 | 6 | 0.075 | CALL=6 |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 0 | 0.0 | - |
| end |  | 72 | 27 | 0.375 | STOP=27 |
| increase |  | 80 | 1 | 0.0125 | PAUSE=1 |
| increase the |  | 80 | 1 | 0.0125 | STOP=1 |
| kill |  | 80 | 34 | 0.425 | STOP=34 |
| kill the |  | 80 | 15 | 0.1875 | CALL=3, STOP=11, TIME=1 |
| lights |  | 80 | 8 | 0.1 | LIGHT_ON=8 |
| list |  | 80 | 3 | 0.0375 | STOP=3 |
| list my |  | 80 | 10 | 0.125 | LIGHT_OFF=4, LIGHT_ON=3, STOP=3 |
| lower |  | 80 | 25 | 0.3125 | STOP=25 |
| lower the |  | 80 | 6 | 0.075 | PAUSE=1, STOP=2, WEATHER=3 |
| make |  | 72 | 4 | 0.05555555555555555 | STOP=2, TIME=2 |
| make a |  | 80 | 1 | 0.0125 | STOP=1 |
| make a phone |  | 80 | 2 | 0.025 | LIGHT_ON=2 |
| next |  | 80 | 11 | 0.1375 | CALL=2, PAUSE=4, STOP=5 |
| pause for |  | 80 | 37 | 0.4625 | PAUSE=37 |
| place |  | 72 | 1 | 0.013888888888888888 | STOP=1 |
| place a |  | 72 | 0 | 0.0 | - |
| play |  | 80 | 5 | 0.0625 | CALL=3, STOP=2 |
| play next |  | 80 | 2 | 0.025 | CALL=1, PAUSE=1 |
| play some |  | 80 | 0 | 0.0 | - |
| power |  | 80 | 37 | 0.4625 | STOP=37 |
| power on |  | 80 | 16 | 0.2 | CALL=2, PAUSE=5, STOP=7, TIME=2 |
| power on the |  | 80 | 14 | 0.175 | PAUSE=10, STOP=1, TIME=2, VOLUME_DOWN=1 |
| remind | YES | 80 | 11 | 0.1375 | LIST_REMINDERS=9, STOP=1, TIME=1 |
| remind me |  | 80 | 11 | 0.1375 | LIST_REMINDERS=11 |
| remind me to |  | 80 | 15 | 0.1875 | LIST_REMINDERS=14, TIME=1 |
| remind me to drink |  | 80 | 0 | 0.0 | - |
| reminder | YES | 80 | 18 | 0.225 | LIST_REMINDERS=17, STOP=1 |
| reminder drink |  | 72 | 19 | 0.2638888888888889 | LIST_REMINDERS=19 |
| send |  | 80 | 31 | 0.3875 | STOP=31 |
| send a |  | 80 | 30 | 0.375 | PAUSE=7, STOP=23 |
| send my |  | 80 | 7 | 0.0875 | PAUSE=2, STOP=5 |
| set |  | 80 | 34 | 0.425 | PAUSE=1, STOP=33 |
| set an |  | 80 | 37 | 0.4625 | PLAY_MUSIC=1, STOP=35, TIME=1 |
| set an alarm |  | 80 | 3 | 0.0375 | STOP=3 |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 11 | 0.1375 | ALARM=11 |
| set an alarm for nine |  | 72 | 0 | 0.0 | - |
| set an alarm for six |  | 80 | 0 | 0.0 | - |
| set color |  | 80 | 4 | 0.05 | STOP=4 |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 16 | 0.2 | STOP=16 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 44 | 0.55 | STOP=44 |
| show my |  | 80 | 19 | 0.2375 | STOP=19 |
| shut |  | 72 | 34 | 0.4722222222222222 | STOP=34 |
| shut off |  | 72 | 1 | 0.013888888888888888 | STOP=1 |
| shut off the |  | 72 | 0 | 0.0 | - |
| skip |  | 80 | 33 | 0.4125 | STOP=33 |
| start |  | 80 | 55 | 0.6875 | STOP=55 |
| start a |  | 80 | 52 | 0.65 | STOP=52 |
| start a timer |  | 80 | 6 | 0.075 | PAUSE=4, STOP=2 |
| start a timer for |  | 80 | 2 | 0.025 | PAUSE=1, TIME=1 |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 14 | 0.175 | PAUSE=2, STOP=12 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 38 | 0.475 | STOP=38 |
| tell me |  | 80 | 19 | 0.2375 | STOP=9, TIME=10 |
| tell me the |  | 80 | 9 | 0.1125 | CALL=1, STOP=2, TIME=6 |
| temperature |  | 80 | 4 | 0.05 | PAUSE=3, PLAY_MUSIC=1 |
| temperature eighteen |  | 80 | 2 | 0.025 | STOP=2 |
| temperature twenty |  | 80 | 1 | 0.0125 | PAUSE=1 |
| temperature twenty six |  | 80 | 0 | 0.0 | - |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 52 | 0.65 | CALL=1, STOP=12, TIME=39 |
| timer one |  | 80 | 2 | 0.025 | TIME=2 |
| timer ten |  | 80 | 14 | 0.175 | PAUSE=3, STOP=1, TIME=10 |
| timer thirty |  | 72 | 19 | 0.2638888888888889 | PAUSE=1, TIME=18 |
| turn |  | 80 | 25 | 0.3125 | STOP=18, TIME=7 |
| turn on |  | 80 | 21 | 0.2625 | STOP=19, TIME=2 |
| turn on the |  | 80 | 10 | 0.125 | CALL=1, STOP=9 |
| turn the |  | 80 | 17 | 0.2125 | STOP=3, TIME=14 |
| turn the volume |  | 80 | 0 | 0.0 | - |
| volume |  | 80 | 22 | 0.275 | CALL=8, PAUSE=6, STOP=6, TIME=1, VOLUME_UP=1 |
| wake |  | 80 | 9 | 0.1125 | STOP=1, WEATHER=8 |
| wake me |  | 80 | 5 | 0.0625 | LIGHT_ON=2, VOLUME_UP=1, WEATHER=2 |
| wake me up |  | 80 | 0 | 0.0 | - |
| wake me up at |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| wake me up at eight |  | 80 | 12 | 0.15 | ALARM=12 |
| wake me up at nine |  | 80 | 2 | 0.025 | ALARM=1, LIGHT_ON=1 |
| wake me up at six |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| what | YES | 64 | 16 | 0.25 | STOP=4, WEATHER=12 |
| what time |  | 72 | 5 | 0.06944444444444445 | LIGHT_OFF=1, WEATHER=4 |
| what time is |  | 72 | 5 | 0.06944444444444445 | TIME=5 |
| what's |  | 72 | 26 | 0.3611111111111111 | MESSAGE=1, PAUSE=2, TIME=1, WEATHER=22 |
| what's the |  | 72 | 2 | 0.027777777777777776 | LIGHT_ON=2 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2664 | 10 | 0.0037537537537537537 |
| lead_in_room_tone | 654 | 90 | 0.13761467889908258 |
| tail_room_tone | 3498 | 429 | 0.12264150943396226 |
| digital_zero | 3840 | 683 | 0.17786458333333333 |

FRR (single-word commands): 0/13 (0.0)
FRR (strict-prefix commands): 0/13 (0.0)

Gate-caused FRR: 0/192 confidence-accepted target rows additionally rejected (0.0)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|

Babble/silence FAR:
- babble: 4/409 (0.009779951100244499)
- silence: 0/329 (0.0)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| unknown | 195 | 192 | 0.9846153846153847 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2664 | 678 | 0.2545045045045045 |
| [0, 0.2)s | 2664 | 10 | 0.0037537537537537537 |
| [0.2, 0.5)s | 2664 | 142 | 0.0533033033033033 |
| [1, 2)s | 2664 | 382 | 0.14339339339339338 |

### margin=4.0

Target exact accuracy: 191/195 (0.9794871794871794)

Slot exact match: n/a (no classifiable slot-bearing Option B target row)

Incomplete-prefix FAR: 735/10656 (0.06897522522522523)

| prefix | char_overlap | n | n_false_accept | far | false_accept_intents |
|---|---|---|---|---|---|
| adjust |  | 80 | 4 | 0.05 | PAUSE=1, STOP=3 |
| adjust brightness |  | 80 | 0 | 0.0 | - |
| adjust brightness to |  | 80 | 0 | 0.0 | - |
| adjust brightness to one |  | 80 | 0 | 0.0 | - |
| adjust brightness to one hundred |  | 80 | 0 | 0.0 | - |
| adjust brightness to sixty |  | 80 | 0 | 0.0 | - |
| adjust brightness to twenty |  | 80 | 0 | 0.0 | - |
| alarm |  | 80 | 6 | 0.075 | PAUSE=6 |
| alarm eight |  | 80 | 2 | 0.025 | PAUSE=2 |
| alarm nine |  | 80 | 1 | 0.0125 | VOLUME_UP=1 |
| alarm six |  | 72 | 4 | 0.05555555555555555 | PAUSE=3, PLAY_MUSIC=1 |
| brightness |  | 80 | 2 | 0.025 | STOP=2 |
| brightness level |  | 80 | 3 | 0.0375 | LIGHT_OFF=2, LIGHT_ON=1 |
| brightness level one |  | 80 | 0 | 0.0 | - |
| brightness level one hundred |  | 80 | 0 | 0.0 | - |
| brightness level sixty |  | 80 | 3 | 0.0375 | PLAY_MUSIC=3 |
| brightness level twenty |  | 80 | 0 | 0.0 | - |
| brightness one |  | 80 | 7 | 0.0875 | PAUSE=7 |
| brightness one hundred |  | 80 | 10 | 0.125 | LIGHT_ON=3, PAUSE=7 |
| brightness sixty |  | 80 | 0 | 0.0 | - |
| brightness twenty |  | 80 | 0 | 0.0 | - |
| change |  | 80 | 10 | 0.125 | STOP=10 |
| change color |  | 80 | 0 | 0.0 | - |
| change color to |  | 80 | 0 | 0.0 | - |
| change the |  | 80 | 8 | 0.1 | PAUSE=2, PLAY_MUSIC=6 |
| change the temperature |  | 80 | 0 | 0.0 | - |
| change the temperature to |  | 80 | 0 | 0.0 | - |
| change the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| change the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| countdown |  | 80 | 3 | 0.0375 | STOP=3 |
| countdown for |  | 80 | 1 | 0.0125 | PAUSE=1 |
| countdown for one |  | 72 | 0 | 0.0 | - |
| countdown for ten |  | 80 | 0 | 0.0 | - |
| countdown for thirty |  | 80 | 0 | 0.0 | - |
| create |  | 80 | 0 | 0.0 | - |
| create a |  | 80 | 1 | 0.0125 | CALL=1 |
| create a reminder |  | 80 | 0 | 0.0 | - |
| create a reminder to |  | 80 | 0 | 0.0 | - |
| create a reminder to drink |  | 80 | 0 | 0.0 | - |
| end |  | 72 | 17 | 0.2361111111111111 | STOP=17 |
| increase |  | 80 | 0 | 0.0 | - |
| increase the |  | 80 | 0 | 0.0 | - |
| kill |  | 80 | 30 | 0.375 | STOP=30 |
| kill the |  | 80 | 2 | 0.025 | STOP=2 |
| lights |  | 80 | 3 | 0.0375 | LIGHT_ON=3 |
| list |  | 80 | 2 | 0.025 | STOP=2 |
| list my |  | 80 | 5 | 0.0625 | LIGHT_OFF=2, LIGHT_ON=3 |
| lower |  | 80 | 24 | 0.3 | STOP=24 |
| lower the |  | 80 | 0 | 0.0 | - |
| make |  | 72 | 1 | 0.013888888888888888 | STOP=1 |
| make a |  | 80 | 0 | 0.0 | - |
| make a phone |  | 80 | 2 | 0.025 | LIGHT_ON=2 |
| next |  | 80 | 7 | 0.0875 | CALL=2, PAUSE=4, STOP=1 |
| pause for |  | 80 | 22 | 0.275 | PAUSE=22 |
| place |  | 72 | 0 | 0.0 | - |
| place a |  | 72 | 0 | 0.0 | - |
| play |  | 80 | 3 | 0.0375 | CALL=3 |
| play next |  | 80 | 0 | 0.0 | - |
| play some |  | 80 | 0 | 0.0 | - |
| power |  | 80 | 36 | 0.45 | STOP=36 |
| power on |  | 80 | 5 | 0.0625 | PAUSE=2, STOP=2, TIME=1 |
| power on the |  | 80 | 12 | 0.15 | PAUSE=9, TIME=2, VOLUME_DOWN=1 |
| remind | YES | 80 | 1 | 0.0125 | STOP=1 |
| remind me |  | 80 | 0 | 0.0 | - |
| remind me to |  | 80 | 0 | 0.0 | - |
| remind me to drink |  | 80 | 0 | 0.0 | - |
| reminder | YES | 80 | 0 | 0.0 | - |
| reminder drink |  | 72 | 0 | 0.0 | - |
| send |  | 80 | 22 | 0.275 | STOP=22 |
| send a |  | 80 | 23 | 0.2875 | PAUSE=7, STOP=16 |
| send my |  | 80 | 0 | 0.0 | - |
| set |  | 80 | 27 | 0.3375 | STOP=27 |
| set an |  | 80 | 19 | 0.2375 | STOP=19 |
| set an alarm |  | 80 | 2 | 0.025 | STOP=2 |
| set an alarm for |  | 80 | 0 | 0.0 | - |
| set an alarm for eight |  | 80 | 0 | 0.0 | - |
| set an alarm for nine |  | 72 | 0 | 0.0 | - |
| set an alarm for six |  | 80 | 0 | 0.0 | - |
| set color |  | 80 | 1 | 0.0125 | STOP=1 |
| set color to |  | 80 | 0 | 0.0 | - |
| set the |  | 80 | 7 | 0.0875 | STOP=7 |
| set the temperature |  | 80 | 0 | 0.0 | - |
| set the temperature to |  | 80 | 0 | 0.0 | - |
| set the temperature to eighteen |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty six |  | 80 | 0 | 0.0 | - |
| set the temperature to twenty two |  | 80 | 0 | 0.0 | - |
| show |  | 80 | 38 | 0.475 | STOP=38 |
| show my |  | 80 | 2 | 0.025 | STOP=2 |
| shut |  | 72 | 28 | 0.3888888888888889 | STOP=28 |
| shut off |  | 72 | 0 | 0.0 | - |
| shut off the |  | 72 | 0 | 0.0 | - |
| skip |  | 80 | 24 | 0.3 | STOP=24 |
| start |  | 80 | 55 | 0.6875 | STOP=55 |
| start a |  | 80 | 48 | 0.6 | STOP=48 |
| start a timer |  | 80 | 3 | 0.0375 | PAUSE=3 |
| start a timer for |  | 80 | 1 | 0.0125 | PAUSE=1 |
| start a timer for one |  | 80 | 0 | 0.0 | - |
| start a timer for ten |  | 80 | 0 | 0.0 | - |
| start a timer for thirty |  | 80 | 0 | 0.0 | - |
| switch |  | 80 | 1 | 0.0125 | STOP=1 |
| switch color |  | 80 | 0 | 0.0 | - |
| switch color to |  | 80 | 0 | 0.0 | - |
| tell |  | 80 | 38 | 0.475 | STOP=38 |
| tell me |  | 80 | 12 | 0.15 | STOP=6, TIME=6 |
| tell me the |  | 80 | 8 | 0.1 | STOP=2, TIME=6 |
| temperature |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| temperature eighteen |  | 80 | 1 | 0.0125 | STOP=1 |
| temperature twenty |  | 80 | 0 | 0.0 | - |
| temperature twenty six |  | 80 | 0 | 0.0 | - |
| temperature twenty two |  | 80 | 0 | 0.0 | - |
| timer |  | 80 | 32 | 0.4 | CALL=1, STOP=11, TIME=20 |
| timer one |  | 80 | 0 | 0.0 | - |
| timer ten |  | 80 | 1 | 0.0125 | PAUSE=1 |
| timer thirty |  | 72 | 2 | 0.027777777777777776 | TIME=2 |
| turn |  | 80 | 19 | 0.2375 | STOP=13, TIME=6 |
| turn on |  | 80 | 12 | 0.15 | STOP=12 |
| turn on the |  | 80 | 5 | 0.0625 | STOP=5 |
| turn the |  | 80 | 9 | 0.1125 | STOP=2, TIME=7 |
| turn the volume |  | 80 | 0 | 0.0 | - |
| volume |  | 80 | 12 | 0.15 | CALL=4, PAUSE=4, STOP=4 |
| wake |  | 80 | 7 | 0.0875 | WEATHER=7 |
| wake me |  | 80 | 3 | 0.0375 | LIGHT_ON=1, WEATHER=2 |
| wake me up |  | 80 | 0 | 0.0 | - |
| wake me up at |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| wake me up at eight |  | 80 | 0 | 0.0 | - |
| wake me up at nine |  | 80 | 1 | 0.0125 | LIGHT_ON=1 |
| wake me up at six |  | 80 | 1 | 0.0125 | PLAY_MUSIC=1 |
| what | YES | 64 | 10 | 0.15625 | STOP=1, WEATHER=9 |
| what time |  | 72 | 2 | 0.027777777777777776 | WEATHER=2 |
| what time is |  | 72 | 0 | 0.0 | - |
| what's |  | 72 | 18 | 0.25 | MESSAGE=1, WEATHER=17 |
| what's the |  | 72 | 2 | 0.027777777777777776 | LIGHT_ON=2 |

Incomplete-prefix FAR by silence_source (R2-5: `digital_zero` is a dataset-limitation fallback, not a neutral padding choice -- its share and FAR must stay visible on their own, never averaged away):

| silence_source | n | n_false_accept | far |
|---|---|---|---|
| none | 2664 | 2 | 0.0007507507507507507 |
| lead_in_room_tone | 654 | 42 | 0.06422018348623854 |
| tail_room_tone | 3498 | 231 | 0.0660377358490566 |
| digital_zero | 3840 | 460 | 0.11979166666666667 |

FRR (single-word commands): 0/13 (0.0)
FRR (strict-prefix commands): 0/13 (0.0)

Gate-caused FRR: 1/192 confidence-accepted target rows additionally rejected (0.005208333333333333)

| competitor | char_overlap | n | rejected true-label distribution |
|---|---|---|---|
| play |  | 1 | PLAY_MUSIC=1 |

Babble/silence FAR:
- babble: 1/409 (0.0024449877750611247)
- silence: 0/329 (0.0)

Target rows by trailing-silence bucket (natural, force-aligned):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| unknown | 195 | 191 | 0.9794871794871794 |

Probe rows by trailing-silence bucket (crafted):

| bucket | n | n_accepted | accept_rate |
|---|---|---|---|
| >=2s | 2664 | 381 | 0.14301801801801803 |
| [0, 0.2)s | 2664 | 2 | 0.0007507507507507507 |
| [0.2, 0.5)s | 2664 | 105 | 0.039414414414414414 |
| [1, 2)s | 2664 | 247 | 0.09271771771771772 |

## Decode latency/peak memory: gate off vs. chosen margin

Hardware: ai-n002.hpc.coe.upd.edu.ph (measured on this node)

| config | n | median latency (ms) | p95 latency (ms) | peak memory (bytes) |
|---|---|---|---|---|
| gate off | 0 | None | None | None |
| margin=4.0 | 0 | None | None | None |

## Alignment / probe-generation failures

- target forced-alignment: 0/0 failed
- probe generation: 6 failures: {'crop_too_short': 6, 'force_align_error': 0, 'missing_audio': 0, 'no_gap': 0}

## Checkpoint/ONNX hashes

Before: {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
After:  {'checkpoint': 'bdd99f37b911fd653128956c86d2abcfdc151e699bba1332daee8711eacab55a', 'vcm_model.fp32.onnx': '54c4b26bcae17d14969ed9604ec9c5e6aaa30c0240618c9b4120a936568762a2', 'vcm_model.fp32.preprocessed.onnx': '35473fc1a050cf385b1d435f5ae01ab1972943a5cd0fce2c49a78bcc8a8e2f6d', 'vcm_model.int8.onnx': 'a8d3c648792f75ea57e14fbcdb8fe291c56eec010d827fb2bcf1e7889722f11f'}
Unchanged: True

