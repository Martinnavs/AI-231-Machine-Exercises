# Soak results: wake0.6

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/val-wake-gap-v1", "name": "wake0.6", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.0, "gate_period": 3.0, "wakeword_threshold": 0.6, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 221: **174 correct first trigger (78.7%)**, 40 missed (7 never opened a period), 4 wrong-intent triggers, 6 early first triggers, 28 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 19: 4 false accepts
- correct by gap: 0.0-0.3 s 59/76, 0.4-0.6 s 47/55, 0.7-1.0 s 68/90
- latency after end of speech (correct): p50 0.361 s, p95 0.699 s
- decode per window: mean 74.45 ms, p50 73.66, p95 126.87, max 177.19 ms (2570 windows)
- wake-word gate per decoded window: mean 5.94 ms, p95 6.83 ms
- RTF (p95 gate+decode over the stride): 0.531
