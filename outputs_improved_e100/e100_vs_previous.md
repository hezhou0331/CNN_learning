# 100-epoch capped runs vs previous `outputs_improved`

- Previous pool: `outputs_improved` (excludes run dirs containing `e100p20` or `conv150`).
- New runs: `outputs_improved_e100` (`*_e100p20`, `--epochs 100 --patience 20`, val-acc early stop).

| experiment_id | prev run | prev test_acc(%) | new run | new last_epoch | stop_reason | new test_acc(%) | Δ test (pp) |
|---|---:|---:|---|---:|---|---:|---:|
| exp2_aug_bn | exp2_aug_bn | 71.70 | exp2_aug_bn_e100p20 | 97 | early_stop_val_acc_patience | 74.60 | 2.90 |
| exp3_aug_dropout | (none) |  | exp3_aug_dropout_e100p20 | 88 | early_stop_val_acc_patience | 72.20 |  |
| exp6_improved_long_6 | exp6_improved_long_6 | 80.40 | exp6_improved_long_6_e100p20 | 100 | max_epochs_reached | 81.80 | 1.40 |
| exp7_improved_long_9 | exp7_improved_long_9 | 81.60 | exp7_improved_long_9_e100p20 | 93 | early_stop_val_acc_patience | 81.40 | -0.20 |
| exp9_aug_bn_15conv | exp9_aug_bn_15conv | 73.10 | exp9_aug_bn_15conv_e100p20 | 100 | max_epochs_reached | 77.30 | 4.20 |
