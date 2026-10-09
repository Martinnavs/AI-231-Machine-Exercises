# Soak results: beam15

`{"sessions": "soak/holdout-wake-gap-v1", "name": "beam15", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "beam_width": 15, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **152 correct first trigger (81.7%)**, 26 missed (5 never opened a period), 8 wrong-action first triggers (intent or slot), 3 wrong-intent triggers in all, 17 early first triggers, 19 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 58/68, 0.4-0.6 s 46/57, 0.7-1.0 s 48/61
- latency after end of speech (correct): p50 0.377 s, p95 0.654 s
- decode per window: mean 33.67 ms, p50 32.66, p95 70.89, max 104.57 ms (2126 windows)
- wake-word gate per decoded window: mean 5.72 ms, p95 6.72 ms
- RTF (p95 gate+decode over the stride): 0.309
