# Soak results: base

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/val-wake-gap-v1", "name": "base", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.0, "gate_period": 3.0, "wakeword_threshold": 0.9, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 221: **169 correct first trigger (76.5%)**, 45 missed (18 never opened a period), 4 wrong-intent triggers, 6 early first triggers, 26 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 19: 4 false accepts
- correct by gap: 0.0-0.3 s 56/76, 0.4-0.6 s 47/55, 0.7-1.0 s 66/90
- latency after end of speech (correct): p50 0.36 s, p95 0.685 s
- decode per window: mean 70.93 ms, p50 69.59, p95 121.5, max 159.38 ms (2257 windows)
- wake-word gate per decoded window: mean 5.59 ms, p95 6.54 ms
- RTF (p95 gate+decode over the stride): 0.508
