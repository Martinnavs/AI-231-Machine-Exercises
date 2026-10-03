# Soak results: cpu-onnx-int8-hybrid

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "cpu-onnx-int8-hybrid", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "poll_s": 0.05, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **147 correct first trigger (79.0%)**, 27 missed (8 never opened a period), 4 wrong-intent triggers, 20 early first triggers, 18 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 56/68, 0.4-0.6 s 46/57, 0.7-1.0 s 45/61
- latency after end of speech (correct): p50 0.379 s, p95 0.656 s
- decode per window: mean 74.39 ms, p50 73.53, p95 123.31, max 160.05 ms (2029 windows)
- wake-word gate per decoded window: mean 5.61 ms, p95 6.36 ms
- RTF (p95 gate+decode over the 0.25 s stride): 0.52
