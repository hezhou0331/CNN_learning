# Convergence rerun vs previous STL10 runs

- Previous root: `outputs_improved`
- Convergence root: `outputs_improved_convergence`
- Reruns used `--epochs 150 --patience 20` (val-acc early stop).

| experiment_id | prev run (under previous root) | prev best_epoch | prev best_val_acc(%) | prev test_acc(%) | new run (under convergence root) | new best_epoch | new best_val_acc(%) | new test_acc(%) | Δ test_acc (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1_aug | exp1_aug | 21 | 71.57 | 66.50 | exp1_aug_conv150 | 64 | 74.00 | 68.30 | 1.80 |
| exp4_aug_bn_sgd | exp4_aug_bn_sgd | 30 | 74.57 | 70.30 | exp4_aug_bn_sgd | 86 | 76.71 | 71.10 | 0.80 |
| exp5_aug_bn_lr | exp5_aug_bn_lr | 28 | 74.43 | 69.80 | exp5_aug_bn_lr | 78 | 76.43 | 73.30 | 3.50 |
| exp8_improved_longer_12 | exp8_improved_longer_12 | 14 | 70.29 | 69.50 | exp8_improved_longer_12 | 143 | 84.57 | 81.20 | 11.70 |

Use this table to decide whether to cite convergence reruns in the report; original `outputs_improved` directories are unchanged.
