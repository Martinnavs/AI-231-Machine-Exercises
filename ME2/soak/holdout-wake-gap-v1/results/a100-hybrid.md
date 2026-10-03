# Soak results: a100-hybrid

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "a100-hybrid", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "torch", "device": "cuda:0", "gpu": 5, "no_cls": false, "cls_threshold": 0.8787, "poll_s": 0.05, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **148 correct first trigger (79.6%)**, 26 missed (8 never opened a period), 3 wrong-intent triggers, 21 early first triggers, 20 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 4 false accepts
- correct by gap: 0.0-0.3 s 56/68, 0.4-0.6 s 46/57, 0.7-1.0 s 46/61
- latency after end of speech (correct): p50 0.38 s, p95 0.661 s
- decode per window: mean 67.6 ms, p50 67.51, p95 106.99, max 931.35 ms (2021 windows)
- wake-word gate per decoded window: mean 9.64 ms, p95 11.16 ms
- RTF (p95 gate+decode over the 0.25 s stride): 0.466
