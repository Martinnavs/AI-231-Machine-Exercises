# Soak results: nowake-ctc-only

`{"sessions": "out/soak/holdout-nowake-v1", "name": "nowake-ctc-only", "models": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/vcm/hybrid-ctcwide-clsxl", "wakeword": "/mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/me2-ctc-attention/ME2/out/wakeword-sesame-ambient-rir-45m", "backend": "onnx", "device": "cpu", "gpu": null, "no_cls": true, "cls_threshold": 0.8787, "cls_slot_threshold": 0.0, "gate": "always", "gate_period": 3.0, "wakeword_threshold": 0.9, "poll_s": 0.05, "stride_s": 0.25, "beam_width": 50, "threads": 1, "workers": 4, "limit": null, "continuous": true}`

- commands 186: **143 correct first trigger (76.9%)**, 38 missed (0 never opened a period), 5 wrong-action first triggers (intent or slot), 5 wrong-intent triggers in all, 12 early first triggers, 0 answered by the classifier
- triggers in the ambient gaps (more than 1.5 s after a unit ended): 2
- out-of-scope sessions 16: 4 false accepts
- correct by gap: 0.0-0.3 s 143/186, 0.4-0.6 s 0/0, 0.7-1.0 s 0/0
- latency after end of speech (correct): p50 0.333 s, p95 0.468 s
- decode per window: mean 91.96 ms, p50 97.69, p95 108.52, max 254.32 ms (6333 windows)
- wake-word gate per decoded window: mean 0.05 ms, p95 0.07 ms
- RTF (p95 gate+decode over the stride): 0.434
