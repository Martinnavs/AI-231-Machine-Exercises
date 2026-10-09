CTC path: threshold -0.1, whole clip. Classifier: max-softmax threshold chosen on clean val to match the CTC clean-val false-accept rate of `baseline` (0/121 = 0.00%); it is never re-swept on perturbed audio. Test split, `exact` target rows; babble FA is on the 47 babble rows, 'ESC-50 silence' is the noise-pool rows (an extra probe, not part of the ai231 test).

## Headline: unperturbed (clean) audio

| model | path | intent | intent+slot | babble FA | ESC-50 silence FA |
|---|---|---|---|---|---|
| baseline | CTC | 88.3% (3377/3823) | 88.3% (3374/3823) | 4/226 | 0/100 |
| heads-A | CTC | 84.6% (3233/3823) | 84.4% (3227/3823) | 2/226 | 0/100 |
| rir-baseline | CTC | 94.9% (3629/3823) | 94.9% (3629/3823) | 3/226 | 0/100 |
| fil50-s1 | CTC | 98.5% (3764/3823) | 98.5% (3764/3823) | 9/226 | 0/100 |
| fil50-s2 | CTC | 97.0% (3709/3823) | 96.9% (3706/3823) | 13/226 | 2/100 |
| fil50-s2-seed1 | CTC | 98.1% (3752/3823) | 98.1% (3752/3823) | 14/226 | 6/100 |
| heads-A | classifier (thr 0.9878) | 87.0% (3327/3823) | 86.7% (3313/3823) | 4/226 | 0/100 |

Classifier without rejection (plain argmax):

| model | intent | intent+slot |
|---|---|---|
| heads-A | 97.3% (3718/3823) | 96.8% (3701/3823) |

Top-k coverage of the true intent, and agreement with the CTC decode:

| model | top-1 | top-3 | classifier top-3 on CTC-correct clips | classifier == CTC (when CTC accepted) |
|---|---|---|---|---|
| heads-A | 97.3% (3718/3823) | 99.5% (3802/3823) | 99.9% (3231/3233) | 98.9% (3204/3238) |

Confusable pairs (true intent -> where it went):

- **heads-A / classifier**: TIME: TIME 69, reject 39; TIMER: TIMER 388, reject 34; PAUSE: PAUSE 99, reject 14; STOP: STOP 60, reject 23; LIGHT_ON: LIGHT_ON 106, reject 20; LIGHT_OFF: LIGHT_ON 2, LIGHT_OFF 20, reject 76
- **heads-A / ctc**: TIME: TIME 84, reject 24; TIMER: TIMER 359, reject 63; PAUSE: PAUSE 103, reject 10; STOP: STOP 75, reject 8; LIGHT_ON: LIGHT_ON 112, reject 14; LIGHT_OFF: LIGHT_ON 2, LIGHT_OFF 87, reject 9

### By source (unperturbed, target rows; intent+slot)

| model | path | optionb |
|---|---|---|
| baseline | CTC | 88.3% (3374/3823) |
| heads-A | CTC | 84.4% (3227/3823) |
| heads-A | classifier | 86.7% (3313/3823) |
| rir-baseline | CTC | 94.9% (3629/3823) |
| fil50-s1 | CTC | 98.5% (3764/3823) |
| fil50-s2 | CTC | 96.9% (3706/3823) |
| fil50-s2-seed1 | CTC | 98.1% (3752/3823) |

## Robustness: perturbed audio (fixed-seed RIR + ESC-50 noise, seed 0)

| model | path | intent | intent+slot | babble FA | ESC-50 silence FA |
|---|---|---|---|---|---|
| baseline | CTC | 74.8% (2859/3823) | 74.5% (2848/3823) | 3/226 | 0/100 |
| heads-A | CTC | 67.1% (2564/3823) | 66.7% (2549/3823) | 3/226 | 0/100 |
| rir-baseline | CTC | 76.1% (2909/3823) | 75.8% (2898/3823) | 1/226 | 0/100 |
| fil50-s1 | CTC | 95.2% (3639/3823) | 95.0% (3633/3823) | 13/226 | 0/100 |
| fil50-s2 | CTC | 93.6% (3580/3823) | 93.5% (3576/3823) | 6/226 | 10/100 |
| fil50-s2-seed1 | CTC | 95.1% (3636/3823) | 95.0% (3632/3823) | 10/226 | 27/100 |
| heads-A | classifier (thr 0.9878) | 73.3% (2802/3823) | 72.1% (2755/3823) | 3/226 | 0/100 |

Classifier without rejection (plain argmax):

| model | intent | intent+slot |
|---|---|---|
| heads-A | 92.0% (3519/3823) | 90.1% (3443/3823) |

Top-k coverage of the true intent, and agreement with the CTC decode:

| model | top-1 | top-3 | classifier top-3 on CTC-correct clips | classifier == CTC (when CTC accepted) |
|---|---|---|---|---|
| heads-A | 92.0% (3519/3823) | 97.4% (3724/3823) | 100.0% (2563/2564) | 98.8% (2540/2571) |

Confusable pairs (true intent -> where it went):

- **heads-A / classifier**: TIME: TIME 54, reject 54; TIMER: TIMER 325, reject 97; PAUSE: PAUSE 88, reject 25; STOP: STOP 44, reject 39; LIGHT_ON: LIGHT_ON 85, reject 41; LIGHT_OFF: LIGHT_ON 1, LIGHT_OFF 10, reject 87
- **heads-A / ctc**: TIME: TIME 66, reject 42; TIMER: TIMER 258, other 1, reject 163; PAUSE: PAUSE 92, other 1, reject 20; STOP: STOP 63, reject 20; LIGHT_ON: LIGHT_ON 93, reject 33; LIGHT_OFF: LIGHT_ON 2, LIGHT_OFF 69, reject 27

### By source (perturbed, target rows; intent+slot)

| model | path | optionb |
|---|---|---|
| baseline | CTC | 74.5% (2848/3823) |
| heads-A | CTC | 66.7% (2549/3823) |
| heads-A | classifier | 72.1% (2755/3823) |
| rir-baseline | CTC | 75.8% (2898/3823) |
| fil50-s1 | CTC | 95.0% (3633/3823) |
| fil50-s2 | CTC | 93.5% (3576/3823) |
| fil50-s2-seed1 | CTC | 95.0% (3632/3823) |

## Gate A (heads-A CTC path vs baseline)

- clean_exact: baseline 3374, joint 3227, floor 3354 -> FAIL
- noisy_exact: baseline 2848, joint 2549, floor 2828 -> FAIL
- noisy_babble_fa: baseline 3, joint 3, allowed 3 -> PASS
- **Gate A: FAIL**
