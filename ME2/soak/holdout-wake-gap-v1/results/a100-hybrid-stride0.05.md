# Soak results: a100-hybrid-stride0.05

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "a100-hybrid-stride0.05", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "torch", "device": "cuda:0", "gpu": 5, "no_cls": false, "cls_threshold": 0.8787, "poll_s": 0.05, "stride_s": 0.05, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **150 correct first trigger (80.7%)**, 20 missed (8 never opened a period), 5 wrong-intent triggers, 28 early first triggers, 22 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 4 false accepts
- correct by gap: 0.0-0.3 s 57/68, 0.4-0.6 s 49/57, 0.7-1.0 s 44/61
- latency after end of speech (correct): p50 0.256 s, p95 0.536 s
- decode per window: mean 64.57 ms, p50 63.87, p95 104.72, max 708.21 ms (9988 windows)
- wake-word gate per decoded window: mean 2.16 ms, p95 3.1 ms
- RTF (p95 gate+decode over the stride): 2.139
