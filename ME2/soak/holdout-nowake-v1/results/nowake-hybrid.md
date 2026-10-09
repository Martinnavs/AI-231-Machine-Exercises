# Soak results: nowake-hybrid

`{"sessions": "out/soak/holdout-nowake-v1", "name": "nowake-hybrid", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": false, "cls_threshold": 0.8787, "cls_slot_threshold": 0.6, "gate": "always", "gate_period": 3.0, "wakeword_threshold": 0.9, "poll_s": 0.05, "stride_s": 0.25, "beam_width": 50, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **165 correct first trigger (88.7%)**, 14 missed (0 never opened a period), 7 wrong-action first triggers (intent or slot), 8 wrong-intent triggers in all, 21 early first triggers, 28 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 4
- out-of-scope sessions 16: 4 false accepts
- correct by gap: 0.0-0.3 s 165/186, 0.4-0.6 s 0/0, 0.7-1.0 s 0/0
- latency after end of speech (correct): p50 0.339 s, p95 0.718 s
- decode per window: mean 93.16 ms, p50 99.04, p95 119.15, max 273.77 ms (6279 windows)
- wake-word gate per decoded window: mean 0.05 ms, p95 0.07 ms
- RTF (p95 gate+decode over the stride): 0.477
