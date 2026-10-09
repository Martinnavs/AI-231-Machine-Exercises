# Soak results: tuned-cpu-onnx-int8-hybrid

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "tuned-cpu-onnx-int8-hybrid", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **151 correct first trigger (81.2%)**, 26 missed (5 never opened a period), 4 wrong-intent triggers, 17 early first triggers, 18 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 58/68, 0.4-0.6 s 46/57, 0.7-1.0 s 47/61
- latency after end of speech (correct): p50 0.379 s, p95 0.654 s
- decode per window: mean 70.19 ms, p50 69.9, p95 125.96, max 184.03 ms (2122 windows)
- wake-word gate per decoded window: mean 5.61 ms, p95 6.6 ms
- RTF (p95 gate+decode over the stride): 0.527
