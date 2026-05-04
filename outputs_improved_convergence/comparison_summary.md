# STL10 Experiment Comparison

| run | model | aug | bn | dropout | optimizer | lr | best_val_acc(%) | test_acc(%) | macro_f1 | weighted_f1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| exp4_aug_bn_sgd | configurable | True | True | 0.0 | sgd | 0.01 | 76.71 | 71.10 | 0.7115 | 0.7115 |
| exp5_aug_bn_lr | configurable | True | True | 0.0 | adam | 0.0003 | 76.43 | 73.30 | 0.7316 | 0.7316 |
| exp8_improved_longer_12 | improved_longer_12 | True | True | 0.3 | adam | 0.001 | 84.57 | 81.20 | 0.8120 | 0.8120 |
| exp1_aug_conv150 | baseline | True | False | 0.0 | adam | 0.001 | 74.00 | 68.30 | 0.6858 | 0.6858 |

- Best by test accuracy: `exp8_improved_longer_12` (81.20%).