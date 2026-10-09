| model | test clean | test perturbed | holdout all | holdout real Filipino | test FA % |
|---|---|---|---|---|---|
| baseline CTC (seed 0) | 88.3 | 74.5 | 54.8 | 16.7 | 1.8 |
| baseline CTC (seed 1) | 87.3 | 73.2 | 53.8 | 7.1 | 2.2 |
| heads-A CTC (seed 0) | 84.4 | 66.7 | 52.2 | 16.7 | 0.9 |
| heads-A CTC (seed 1) | 87.5 | 76.1 | 52.2 | 6.0 | 0.4 |
| heads-A classifier T=0.9 (seed 0) | 93.6 | 82.9 | 64.5 | 27.4 | 4.4 |
| heads-A classifier T=0.9 (seed 1) | 95.4 | 88.9 | 58.1 | 8.3 | 4.4 |
| rir-only baseline (LEAKY: old split) | 94.9 | 75.8 | 68.3 | 32.1 | 1.3 |
| fil50 s1 (internal data) | 98.5 | 95.0 | 86.6 | 71.4 | 4.0 |
| fil50 s2 | 96.9 | 93.5 | 83.9 | 67.9 | 5.8 |
| fil50 s2-seed1 | 98.1 | 95.0 | 85.5 | 69.0 | 6.2 |
