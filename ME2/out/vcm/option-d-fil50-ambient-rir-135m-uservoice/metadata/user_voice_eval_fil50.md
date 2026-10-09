# user-voice retrain vs production (same val/test, each at its own chosen threshold, margin off)

eval manifest: `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv`; clean-val-chosen thresholds: old -0.1, new -0.075

| set | old exact | new exact | old babble/silence FA | new babble/silence FA |
|---|---|---|---|---|
| val/clean | 3263/3330 (0.9799) | 3227/3330 (0.9691) | 13/409 , 0/329 | 7/409 , 1/329 |
| val/noisy_s0 | 3062/3330 (0.9195) | 2989/3330 (0.8976) | 30/409 , 4/329 | 14/409 , 3/329 |
| test/clean | 3448/3494 (0.9868) | 3424/3494 (0.9800) | 3/255 , 1/324 | 1/255 , 1/324 |
| test/noisy_s0 | 3345/3494 (0.9574) | 3266/3494 (0.9347) | 10/255 , 6/324 | 0/255 , 4/324 |

## Per-intent exact-correct, val+test x clean+noisy pooled (weak intents first)

| intent | n | old | new | delta |
|---|---|---|---|---|
| **LIGHT_OFF** | 450 | 424 (94.22%) | 424 (94.22%) | +0 (+0.00pp) |
| **COLOR** | 1282 | 1273 (99.30%) | 1268 (98.91%) | -5 (-0.39pp) |
| **TIME** | 432 | 415 (96.06%) | 406 (93.98%) | -9 (-2.08pp) |
| **CALL** | 380 | 361 (95.00%) | 359 (94.47%) | -2 (-0.53pp) |
| **MESSAGE** | 420 | 407 (96.90%) | 398 (94.76%) | -9 (-2.14pp) |
| **LIST_REMINDERS** | 428 | 414 (96.73%) | 412 (96.26%) | -2 (-0.47pp) |
| ALARM | 1288 | 1237 (96.04%) | 1199 (93.09%) | -38 (-2.95pp) |
| BRIGHTNESS | 1022 | 979 (95.79%) | 962 (94.13%) | -17 (-1.66pp) |
| CREATE_REMINDER | 1148 | 1095 (95.38%) | 1073 (93.47%) | -22 (-1.92pp) |
| LIGHT_ON | 474 | 456 (96.20%) | 447 (94.30%) | -9 (-1.90pp) |
| NEXT | 1086 | 1007 (92.73%) | 1007 (92.73%) | +0 (+0.00pp) |
| PAUSE | 370 | 352 (95.14%) | 337 (91.08%) | -15 (-4.05pp) |
| PLAY_MUSIC | 506 | 488 (96.44%) | 475 (93.87%) | -13 (-2.57pp) |
| STOP | 534 | 522 (97.75%) | 518 (97.00%) | -4 (-0.75pp) |
| TEMPERATURE | 1268 | 1219 (96.14%) | 1183 (93.30%) | -36 (-2.84pp) |
| TIMER | 1212 | 1157 (95.46%) | 1143 (94.31%) | -14 (-1.16pp) |
| VOLUME_DOWN | 460 | 450 (97.83%) | 443 (96.30%) | -7 (-1.52pp) |
| VOLUME_UP | 498 | 486 (97.59%) | 480 (96.39%) | -6 (-1.20pp) |
| WEATHER | 390 | 376 (96.41%) | 372 (95.38%) | -4 (-1.03pp) |

## Your 20 raw recordings (prosody was used in training via conversion; timbre unseen)

| clip | label | old -> (conf) | new -> (conf) |
|---|---|---|---|
| uv_call_1 | CALL | CALL (-0.089) OK | CALL (-0.001) OK |
| uv_change_color_to_blue_1 | COLOR | COLOR (-0.002) OK | COLOR (-0.000) OK |
| uv_change_color_to_green_1 | COLOR | COLOR (-0.001) OK | COLOR (-0.000) OK |
| uv_change_color_to_green_2 | COLOR | COLOR (-0.009) OK | COLOR (-0.003) OK |
| uv_change_color_to_red_1 | COLOR | COLOR (-0.000) OK | COLOR (-0.001) OK |
| uv_make_a_phone_call_1 | CALL | CALL (-0.006) OK | CALL (-0.000) OK |
| uv_place_a_call_1 | CALL | CALL (-0.003) OK | CALL (-0.000) OK |
| uv_send_my_message_1 | MESSAGE | MESSAGE (-0.000) OK | MESSAGE (-0.000) OK |
| uv_send_my_message_2 | MESSAGE | MESSAGE (-0.000) OK | MESSAGE (-0.000) OK |
| uv_send_my_message_3 | MESSAGE | MESSAGE (-0.010) OK | MESSAGE (-0.000) OK |
| uv_set_color_to_green_1 | COLOR | COLOR (-0.004) OK | COLOR (-0.000) OK |
| uv_set_color_to_red_1 | COLOR | COLOR (-0.003) OK | COLOR (-0.002) OK |
| uv_show_my_reminders_1 | LIST_REMINDERS | LIST_REMINDERS (-0.004) OK | LIST_REMINDERS (-0.000) OK |
| uv_shut_off_the_lights_1 | LIGHT_OFF | LIGHT_OFF (-0.001) OK | LIGHT_OFF (-0.001) OK |
| uv_shut_off_the_lights_2 | LIGHT_OFF | LIGHT_OFF (-0.001) OK | LIGHT_OFF (-0.000) OK |
| uv_shut_off_the_lights_3 | LIGHT_OFF | LIGHT_OFF (-0.002) OK | LIGHT_OFF (-0.000) OK |
| uv_switch_color_to_green_1 | COLOR | COLOR (-0.004) OK | COLOR (-0.000) OK |
| uv_switch_color_to_red_1 | COLOR | COLOR (-0.017) OK | COLOR (-0.002) OK |
| uv_what_time_is_it_1 | TIME | TIME (-0.000) OK | TIME (-0.000) OK |
| uv_what_time_is_it_2 | TIME | TIME (-0.073) OK | TIME (-0.000) OK |

raw clips correct: old 20/20, new 20/20
