# Soak results: tuned-a100-hybrid

`{"sessions": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/out/soak/holdout-wake-gap-v1", "name": "tuned-a100-hybrid", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "torch", "device": "cuda:0", "gpu": 5, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate_period": 3.0, "wakeword_threshold": 0.8, "poll_s": 0.05, "stride_s": 0.25, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **152 correct first trigger (81.7%)**, 25 missed (5 never opened a period), 3 wrong-intent triggers, 18 early first triggers, 20 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 0
- out-of-scope sessions 16: 4 false accepts
- correct by gap: 0.0-0.3 s 58/68, 0.4-0.6 s 46/57, 0.7-1.0 s 48/61
- latency after end of speech (correct): p50 0.38 s, p95 0.661 s
- decode per window: mean 63.79 ms, p50 65.05, p95 101.79, max 359.98 ms (2115 windows)
- wake-word gate per decoded window: mean 9.29 ms, p95 10.69 ms
- RTF (p95 gate+decode over the stride): 0.445
