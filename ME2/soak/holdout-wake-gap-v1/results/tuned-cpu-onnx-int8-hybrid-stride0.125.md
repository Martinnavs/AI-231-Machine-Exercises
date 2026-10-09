# Soak results: tuned-cpu-onnx-int8-hybrid-stride0.125

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "tuned-cpu-onnx-int8-hybrid-stride0.125", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.125, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **154 correct first trigger (82.8%)**, 22 missed (5 never opened a period), 4 wrong-intent triggers, 21 early first triggers, 19 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 57/68, 0.4-0.6 s 48/57, 0.7-1.0 s 49/61
- latency after end of speech (correct): p50 0.295 s, p95 0.554 s
- decode per window: mean 69.03 ms, p50 68.52, p95 116.05, max 166.72 ms (4285 windows)
- wake-word gate per decoded window: mean 3.47 ms, p95 4.25 ms
- RTF (p95 gate+decode over the stride): 0.958
