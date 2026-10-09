# Soak results: beam10

`{"sessions": "soak/holdout-wake-gap-v1", "name": "beam10", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "beam_width": 10, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **151 correct first trigger (81.2%)**, 26 missed (5 never opened a period), 9 wrong-action first triggers (intent or slot), 4 wrong-intent triggers in all, 18 early first triggers, 20 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 57/68, 0.4-0.6 s 46/57, 0.7-1.0 s 48/61
- latency after end of speech (correct): p50 0.375 s, p95 0.661 s
- decode per window: mean 28.23 ms, p50 27.02, p95 65.04, max 100.23 ms (2123 windows)
- wake-word gate per decoded window: mean 6.31 ms, p95 7.49 ms
- RTF (p95 gate+decode over the stride): 0.287
