# STL10 Experiment Comparison

| run | model | aug | bn | dropout | optimizer | lr | best_val_acc(%) | test_acc(%) | macro_f1 | weighted_f1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | baseline | False | False | 0.0 | adam | 0.001 | 61.43 | 54.70 | 0.5464 | 0.5464 |
| exp1_aug | baseline | True | False | 0.0 | adam | 0.001 | 71.57 | 66.50 | 0.6688 | 0.6688 |
| exp2_aug_bn | configurable | True | True | 0.0 | adam | 0.001 | 77.14 | 71.70 | 0.7178 | 0.7178 |
| exp3_aug_dropout | configurable | True | False | 0.3 | adam | 0.001 | 74.14 | 69.00 | 0.6918 | 0.6918 |
| exp4_aug_bn_sgd | configurable | True | True | 0.0 | sgd | 0.01 | 74.57 | 70.30 | 0.7052 | 0.7052 |
| exp5_aug_bn_lr | configurable | True | True | 0.0 | adam | 0.0003 | 74.43 | 69.80 | 0.6959 | 0.6959 |
| exp6_improved_long | improved_long | True | True | 0.3 | adam | 0.001 | 82.71 | 80.40 | 0.8055 | 0.8055 |
| exp7_improved_longer | improved_longer | True | True | 0.3 | adam | 0.001 | 81.57 | 81.60 | 0.8153 | 0.8153 |
| exp8_improved_longer_12 | improved_longer_12 | True | True | 0.3 | adam | 0.001 | 70.29 | 69.50 | 0.6948 | 0.6948 |

- Best by test accuracy: `exp7_improved_longer` (81.60%).