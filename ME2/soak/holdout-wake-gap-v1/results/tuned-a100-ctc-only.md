# Soak results: tuned-a100-ctc-only

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "tuned-a100-ctc-only", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "torch", "device": "cuda:0", "gpu": 5, "no_cls": true, "cls_threshold": 0.8787, "cls_slot_threshold": 0.0, "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **139 correct first trigger (74.7%)**, 45 missed (5 never opened a period), 2 wrong-intent triggers, 10 early first triggers, 0 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 55/68, 0.4-0.6 s 40/57, 0.7-1.0 s 44/61
- latency after end of speech (correct): p50 0.368 s, p95 0.48 s
- decode per window: mean 65.54 ms, p50 67.74, p95 102.61, max 117.18 ms (2202 windows)
- wake-word gate per decoded window: mean 9.35 ms, p95 10.51 ms
- RTF (p95 gate+decode over the stride): 0.448
