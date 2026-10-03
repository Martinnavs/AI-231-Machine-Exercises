## Accuracy, intent+slot (%)

| set (targets) | old-production CTC | H-int CTC | H-int cls | H-wide CTC | H-wide cls | H-xl CTC | H-xl cls | wide+wide A | xl+xl A | wide CTC + xl cls A | wide CTC + xl cls B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ai231 test, original rows, clean (3823) | 96.9 | 96.7 | 97.8 | 95.0 | 97.5 | 94.0 | 97.7 | 98.5 | 98.4 | 98.8 | 98.7 |
| ai231 test, original rows, perturbed (3823) | 93.5 | 91.8 | 92.6 | 87.8 | 92.4 | 83.3 | 92.9 | 93.9 | 93.6 | 94.9 | 94.3 |
| ai231 persona rows (voice-disjoint), clean (2946) | 98.4 | n/a | n/a | 98.7 | 99.4 | 98.0 | 99.2 | 99.8 | 99.5 | 99.7 | 99.6 |
| old internal test, non-persona, clean (1927) | 97.2 | n/a | n/a | 96.2 | 98.5 | 95.0 | 97.5 | 99.4 | 98.2 | 99.1 | 98.5 |
| old internal test, 129 real recordings (129) | 96.9 | n/a | n/a | 96.1 | 86.0 | 83.7 | 84.5 | 96.9 | 90.7 | 98.4 | 94.6 |
| holdout (186 real + 16 OOS) (186) | n/a | 84.9 | 84.4 | 66.7 | 69.4 | 71.5 | 76.9 | 72.6 | 79.0 | 77.4 | 78.0 |
| leak-free internal held-out, clean (638) | 95.8 | 94.5 | 96.7 | 93.4 | 96.6 | 90.8 | 94.2 | 98.4 | 96.1 | 97.8 | 96.6 |
| leak-free internal held-out, perturbed (638) | 89.3 | 87.6 | 87.8 | 80.7 | 85.4 | 75.9 | 85.1 | 87.0 | 86.7 | 87.3 | 86.4 |
| user voice raw (20), clean (20) | 90.0 | 90.0 | 95.0 | 95.0 | 100.0 | 95.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| user voice raw (20), perturbed (20) | 90.0 | 90.0 | 90.0 | 95.0 | 100.0 | 95.0 | 95.0 | 100.0 | 95.0 | 95.0 | 95.0 |
| user voice converted (648), clean (648) | 89.7 | 90.9 | 93.2 | 98.8 | 98.0 | 96.6 | 99.7 | 99.8 | 99.7 | 100.0 | 100.0 |

## False accepts on babble + noise/silence rows

| set | old-production CTC | H-int CTC | H-int cls | H-wide CTC | H-wide cls | H-xl CTC | H-xl cls | wide+wide A | xl+xl A | wide CTC + xl cls A | wide CTC + xl cls B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ai231 test, original rows, clean | 0/76 | 0/76 | 1/76 | 0/76 | 3/76 | 1/76 | 4/76 | 3/76 | 5/76 | 4/76 | 4/76 |
| ai231 test, original rows, perturbed | 0/76 | 0/76 | 1/76 | 0/76 | 3/76 | 0/76 | 4/76 | 3/76 | 4/76 | 4/76 | 4/76 |
| ai231 persona rows (voice-disjoint), clean | 2/50 | n/a | n/a | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| old internal test, non-persona, clean | 0/579 | n/a | n/a | 1/579 | 8/579 | 0/579 | 5/579 | 8/579 | 5/579 | 6/579 | 5/579 |
| old internal test, 129 real recordings | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| holdout (186 real + 16 OOS) | n/a | 0/16 | 0/16 | 0/16 | 1/16 | 0/16 | 1/16 | 1/16 | 1/16 | 1/16 | 1/16 |
| leak-free internal held-out, clean | 2/117 | 2/117 | 2/117 | 0/117 | 2/117 | 0/117 | 0/117 | 2/117 | 0/117 | 0/117 | 0/117 |
| leak-free internal held-out, perturbed | 4/117 | 3/117 | 1/117 | 0/117 | 2/117 | 0/117 | 1/117 | 2/117 | 1/117 | 1/117 | 1/117 |
| user voice raw (20), clean | 2/50 | 2/50 | 1/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 |
| user voice raw (20), perturbed | 5/50 | 2/50 | 1/50 | 0/50 | 0/50 | 2/50 | 0/50 | 0/50 | 2/50 | 0/50 | 0/50 |
| user voice converted (648), clean | 2/50 | 2/50 | 1/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 |
