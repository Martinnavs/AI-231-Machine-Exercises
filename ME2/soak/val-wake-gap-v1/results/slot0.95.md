# Soak results: slot0.95

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/val-wake-gap-v1", "name": "slot0.95", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.95, "gate_period": 3.0, "wakeword_threshold": 0.9, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 221: **164 correct first trigger (74.2%)**, 52 missed (18 never opened a period), 4 wrong-intent triggers, 2 early first triggers, 20 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 19: 4 false accepts
- correct by gap: 0.0-0.3 s 56/76, 0.4-0.6 s 45/55, 0.7-1.0 s 63/90
- latency after end of speech (correct): p50 0.361 s, p95 0.672 s
- decode per window: mean 72.88 ms, p50 71.45, p95 125.69, max 163.67 ms (2294 windows)
- wake-word gate per decoded window: mean 5.52 ms, p95 6.37 ms
- RTF (p95 gate+decode over the stride): 0.525
