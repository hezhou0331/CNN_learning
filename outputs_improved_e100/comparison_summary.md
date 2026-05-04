# STL10 Experiment Comparison

| run | model | aug | bn | dropout | optimizer | lr | best_val_acc(%) | test_acc(%) | macro_f1 | weighted_f1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| exp2_aug_bn_e100p20 | configurable | True | True | 0.0 | adam | 0.001 | 77.14 | 74.60 | 0.7461 | 0.7461 |
| exp3_aug_dropout_e100p20 | configurable | True | False | 0.3 | adam | 0.001 | 74.71 | 72.20 | 0.7239 | 0.7239 |
| exp6_improved_long_6_e100p20 | improved_long | True | True | 0.3 | adam | 0.001 | 86.00 | 81.80 | 0.8196 | 0.8196 |
| exp7_improved_long_9_e100p20 | improved_longer | True | True | 0.3 | adam | 0.001 | 83.71 | 81.40 | 0.8150 | 0.8150 |
| exp9_aug_bn_15conv_e100p20 | configurable_15conv | True | True | 0.0 | adam | 0.001 | 80.86 | 77.30 | 0.7744 | 0.7744 |

- Best by test accuracy: `exp6_improved_long_6_e100p20` (81.80%).