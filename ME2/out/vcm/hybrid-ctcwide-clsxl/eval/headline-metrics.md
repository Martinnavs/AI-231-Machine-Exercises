## Headline (test split, exact-variation rows)

| Metric | Hybrid (INT8 ONNX) | Wide CTC alone (fp32) |
|---|---:|---:|
| Variation balanced accuracy (93 variations + OOS) | 0.9877 | 0.9524 |
| ... on human voices only | 91.8% (214/233) | 84.5% (197/233) |
| ... on synthetic voices only | 99.3% (3,564/3,590) | 95.7% (3,436/3,590) |
| Command + slot accuracy (in scope) | 98.8% (3,778/3,823) | 95.0% (3,633/3,823) |
| Command accuracy | 98.9% (3,780/3,823) | 95.0% (3,633/3,823) |
| Slot accuracy (command right, slotted) | 99.9% (2,342/2,344) | 100.0% (2,232/2,232) |
| Out-of-scope false accept | 3.9% (3/76) | 0.0% (0/76) |
| In-scope false reject | 1.0% (40/3,823) | 4.9% (187/3,823) |
| Synthetic-negative misfire | 2.8% (7/250) | 1.2% (3/250) |

## By split (hybrid, INT8 ONNX; accuracy = command and slot both right)

| Split | In-scope clips | Accuracy | Command acc. | Human voices | Synthetic voices | OOS accepted | In scope rejected | Synthetic negatives accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 1,001 | 99.1% (992/1,001) | 99.5% (996/1,001) | 100.0% (57/57) | 99.0% (935/944) | 5.6% (1/18) | 0.5% (5/1,001) | 1.9% (2/103) |
| test | 3,823 | 98.8% (3,778/3,823) | 98.9% (3,780/3,823) | 91.8% (214/233) | 99.3% (3,564/3,590) | 3.9% (3/76) | 1.0% (40/3,823) | 2.8% (7/250) |
| holdout | 186 | 76.9% (143/186) | 78.0% (145/186) | 50.0% (43/86) | 100.0% (100/100) | 6.2% (1/16) | 20.4% (38/186) | n/a |

## By split (wide CTC alone, fp32)

| Split | Accuracy | Human voices | Synthetic voices | OOS accepted | In scope rejected |
|---|---:|---:|---:|---:|---:|
| val | 95.8% (959/1,001) | 96.5% (55/57) | 95.8% (904/944) | 5.6% (1/18) | 4.0% (40/1,001) |
| test | 95.0% (3,633/3,823) | 84.5% (197/233) | 95.7% (3,436/3,590) | 0.0% (0/76) | 4.9% (187/3,823) |
| holdout | 66.7% (124/186) | 30.2% (26/86) | 98.0% (98/100) | 0.0% (0/16) | 31.7% (59/186) |
