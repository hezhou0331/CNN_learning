# STL10 Experiment Comparison

| run | model | aug | bn | dropout | optimizer | lr | best_val_acc(%) | test_acc(%) | macro_f1 | weighted_f1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | baseline | False | False | 0.0 | adam | 0.001 | 61.29 | 55.90 | 0.5594 | 0.5594 |
| exp1_aug | baseline | True | False | 0.0 | adam | 0.001 | 74.00 | 68.30 | 0.6858 | 0.6858 |
| exp2_aug_bn | configurable | True | True | 0.0 | adam | 0.001 | 77.14 | 74.60 | 0.7461 | 0.7461 |
| exp3_aug_dropout | configurable | True | False | 0.3 | adam | 0.001 | 74.71 | 72.20 | 0.7239 | 0.7239 |
| exp4_aug_bn_sgd | configurable | True | True | 0.0 | sgd | 0.01 | 76.71 | 71.10 | 0.7115 | 0.7115 |
| exp5_aug_bn_lr | configurable | True | True | 0.0 | adam | 0.0003 | 76.43 | 73.30 | 0.7316 | 0.7316 |
| exp6_improved_long_6 | improved_long | True | True | 0.3 | adam | 0.001 | 86.00 | 81.80 | 0.8196 | 0.8196 |
| exp7_improved_long_9 | improved_longer | True | True | 0.3 | adam | 0.001 | 83.71 | 81.40 | 0.8150 | 0.8150 |
| exp8_improved_longer_12 | improved_longer_12 | True | True | 0.3 | adam | 0.001 | 84.57 | 81.20 | 0.8120 | 0.8120 |
| exp9_aug_bn_15conv | configurable_15conv | True | True | 0.0 | adam | 0.001 | 80.86 | 77.30 | 0.7744 | 0.7744 |

- Best by test accuracy: `exp6_improved_long_6` (81.80%).