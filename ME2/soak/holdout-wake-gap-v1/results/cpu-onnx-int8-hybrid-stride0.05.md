# Soak results: cpu-onnx-int8-hybrid-stride0.05

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "cpu-onnx-int8-hybrid-stride0.05", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "poll_s": 0.05, "stride_s": 0.05, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **149 correct first trigger (80.1%)**, 20 missed (8 never opened a period), 6 wrong-intent triggers, 27 early first triggers, 20 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 56/68, 0.4-0.6 s 48/57, 0.7-1.0 s 45/61
- latency after end of speech (correct): p50 0.256 s, p95 0.536 s
- decode per window: mean 72.58 ms, p50 70.52, p95 131.99, max 430.9 ms (10013 windows)
- wake-word gate per decoded window: mean 1.6 ms, p95 2.15 ms
- RTF (p95 gate+decode over the stride): 2.677
