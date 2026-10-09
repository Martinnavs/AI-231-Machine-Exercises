# Soak results: a100-ctc-only

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "a100-ctc-only", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "torch", "device": "cuda:0", "gpu": 5, "no_cls": true, "cls_threshold": 0.8787, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **137 correct first trigger (73.7%)**, 47 missed (8 never opened a period), 2 wrong-intent triggers, 10 early first triggers, 0 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 3 false accepts
- correct by gap: 0.0-0.3 s 53/68, 0.4-0.6 s 40/57, 0.7-1.0 s 44/61
- latency after end of speech (correct): p50 0.37 s, p95 0.482 s
- decode per window: mean 67.58 ms, p50 68.99, p95 107.36, max 132.01 ms (2103 windows)
- wake-word gate per decoded window: mean 9.88 ms, p95 11.51 ms
- RTF (p95 gate+decode over the stride): 0.471
