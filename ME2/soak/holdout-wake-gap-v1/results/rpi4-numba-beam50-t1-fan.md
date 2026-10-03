# Soak results: rpi4-numba-beam50-t1-fan

`{"sessions": "soak/holdout-wake-gap-v1", "name": "rpi4-numba-beam50-t1-fan", "models": "/home/rpi-anavarez/Documents/AI-222-Machine-Exercises/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/home/rpi-anavarez/Documents/AI-222-Machine-Exercises/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate": "wakeword", "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.25, "beam_width": 50, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **151 correct first trigger (81.2%)**, 26 missed (5 never opened a period), 9 wrong-action first triggers (intent or slot), 4 wrong-intent triggers in all, 17 early first triggers, 18 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 58/68, 0.4-0.6 s 46/57, 0.7-1.0 s 47/61
- latency after end of speech (correct): p50 0.379 s, p95 0.654 s
- decode per window: mean 63.2 ms, p50 56.31, p95 202.42, max 268.17 ms (2122 windows)
- wake-word gate per decoded window: mean 36.87 ms, p95 37.92 ms
- RTF (p95 gate+decode over the stride): 0.957
