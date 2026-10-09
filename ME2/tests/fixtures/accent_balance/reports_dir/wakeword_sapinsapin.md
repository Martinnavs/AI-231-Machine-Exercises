# QA Report: wakeword_sapinsapin

- **Source**: /mnt/jfs_hpc/home/anthony.martin.navarez/AI-222-Machine-Exercises/ME2/tests/fixtures/accent_balance/shim_fixture/wakeword_sapinsapin
- **Files**: 4
- **Model**: small
- **Threshold**: 0.8
- **Device index**: 0
- **Workers**: 2
- **Date**: 2026-09-25

---

## Flagged

| file | expected | transcribed | score | exact? |
|---|---|---|---|---|
| Xylophone parade. - job_3.wav | Xylophone parade. | Computer. | 0.333 | no |

## Passed

| file | expected | transcribed | score | exact? |
|---|---|---|---|---|
| Computer. - job_0.wav | Computer. | Computer | 1.000 | yes |
| Computer. - job_1.wav | Computer. | Computer. | 1.000 | yes |
| Computer. - job_2.wav | Computer. | Computer | 1.000 | yes |

**Summary**: 3 passed, 1 flagged, 0 error(s) of 4 file(s).
