# STL10 Experiment Comparison

| run | model | aug | bn | dropout | optimizer | lr | best_val_acc(%) | test_acc(%) | macro_f1 | weighted_f1 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | baseline | False | False | 0.0 | adam | 0.001 | 61.43 | 54.70 | 0.5464 | 0.5464 |
| exp1_aug | baseline | True | False | 0.0 | adam | 0.001 | 71.57 | 66.50 | 0.6688 | 0.6688 |
| exp2_aug_bn | configurable | True | True | 0.0 | adam | 0.001 | 77.14 | 71.70 | 0.7178 | 0.7178 |
| exp3_aug_bn_dropout | configurable | True | True | 0.3 | adam | 0.001 | 73.29 | 71.00 | 0.7099 | 0.7099 |
| exp4_opt_adam | configurable | True | True | 0.3 | adam | 0.001 | 73.57 | 70.50 | 0.7053 | 0.7053 |
| exp4_opt_sgd | configurable | True | True | 0.3 | sgd | 0.01 | 74.57 | 71.60 | 0.7171 | 0.7171 |
| exp5_best_combo | configurable | True | True | 0.3 | adamw | 0.0008 | 75.00 | 70.70 | 0.7067 | 0.7067 |

- Best by test accuracy: `exp2_aug_bn` (71.70%).