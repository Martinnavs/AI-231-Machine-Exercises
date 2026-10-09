# Soak results: period4

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/val-wake-gap-v1", "name": "period4", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.0, "gate_period": 4.0, "wakeword_threshold": 0.9, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 221: **175 correct first trigger (79.2%)**, 37 missed (18 never opened a period), 6 wrong-intent triggers, 6 early first triggers, 28 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 19: 4 false accepts
- correct by gap: 0.0-0.3 s 56/76, 0.4-0.6 s 48/55, 0.7-1.0 s 71/90
- latency after end of speech (correct): p50 0.364 s, p95 0.697 s
- decode per window: mean 76.83 ms, p50 77.12, p95 132.69, max 164.77 ms (2420 windows)
- wake-word gate per decoded window: mean 5.7 ms, p95 6.38 ms
- RTF (p95 gate+decode over the stride): 0.555
