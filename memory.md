# outputs_improved 实验记录（STL-10 分类）

## 任务与数据

在 **STL-10**（96×96 彩色图、10 类）上训练卷积网络。验证集由训练集按固定比例划分；指标以 **测试集准确率** 为主，并记录 **macro / weighted F1**。各次实验除下表所列变量外，其余训练流程一致（如 Cosine 学习率调度、早停耐心等，以 `src/train.py` 中 `EXPERIMENT_PRESETS` 为准）。

**数据增强（`use_augmentation=True`）**：Resize(110) → RandomCrop(96) → RandomHorizontalFlip → ColorJitter → 归一化；关闭时训练集用 CenterCrop，无随机几何/颜色扰动（见 `src/dataset.py`）。

**模型类型**：

- **baseline**：固定结构的小 CNN，无 BN、无 Dropout（`BaselineCNN`）。
- **configurable**：与 baseline **同深度、同通道规模**，但可开关 **BatchNorm**，全连接前可设 **Dropout**（`ConfigurableCNN`），用于消融。

---

## 各次实验「改了什么」

| 目录 / run | 相对上一阶段的变量 | 设计意图 |
|------------|-------------------|----------|
| **baseline** | 无增强、baseline 模型、无 BN、无 Dropout、**Adam** lr=1e-3、weight_decay=5e-4 | 对照组，衡量「最简设置」上限 |
| **exp1_aug** | 仅打开 **数据增强**，其余同 baseline 模型与优化器 | 看增强单独带来的收益 |
| **exp2_aug_bn** | 换为 **configurable** 并 **use_bn=True**，仍无 Dropout、Adam lr=1e-3 | 在增强基础上加 **批归一化**，稳定训练、提高容量利用 |
| **exp3_aug_bn_dropout** | 在 exp2 上 **Dropout=0.3**（分类头前） | 抑制过拟合；验证集与测试表现可能分化 |
| **exp4_opt_adam** | 与 exp3 **超参相同**（Aug+BN+Dropout+Adam），独立一次运行 | 作为 **优化器对比** 的 Adam 支路（与 exp4_opt_sgd 对照） |
| **exp4_opt_sgd** | 相对 exp4_opt_adam：**SGD**（momentum=0.9）**lr=0.01**，其余同 | 经典大 lr SGD + BN 组合是否更优 |
| **exp5_best_combo** | **AdamW**、**lr=8e-4**、**weight_decay=1e-3**（其余仍为 Aug+BN+Dropout=0.3） | 在先前结果上尝试「更常用」的正则化与学习率组合 |

---

## 结果汇总（来自 `outputs_improved/comparison_summary`）

| run | 模型 | 增强 | BN | Dropout | 优化器 | lr | 最佳 epoch | 验证准确率 (%) | 测试准确率 (%) | macro F1 | weighted F1 |
|-----|------|:----:|:--:|:-------:|--------|-----|:----------:|---------------:|---------------:|-----------:|------------:|
| baseline | baseline | 否 | 否 | 0 | adam | 0.001 | 13 | 61.43 | 54.70 | 0.5464 | 0.5464 |
| exp1_aug | baseline | 是 | 否 | 0 | adam | 0.001 | 21 | 71.57 | 66.50 | 0.6688 | 0.6688 |
| exp2_aug_bn | configurable | 是 | 是 | 0 | adam | 0.001 | 43 | **77.14** | **71.70** | **0.7178** | **0.7178** |
| exp3_aug_bn_dropout | configurable | 是 | 是 | 0.3 | adam | 0.001 | 41 | 73.29 | 71.00 | 0.7099 | 0.7099 |
| exp4_opt_adam | configurable | 是 | 是 | 0.3 | adam | 0.001 | 48 | 73.57 | 70.50 | 0.7053 | 0.7053 |
| exp4_opt_sgd | configurable | 是 | 是 | 0.3 | sgd | 0.01 | 43 | 74.57 | 71.60 | 0.7171 | 0.7171 |
| exp5_best_combo | configurable | 是 | 是 | 0.3 | adamw | 0.0008 | 43 | 75.00 | 70.70 | 0.7067 | 0.7067 |

**结论摘要**：

- **测试准确率最高**：`exp2_aug_bn`（**71.70%**）。说明在本设置下，**增强 + BN** 已很强；再加 Dropout 后验证峰值下降，测试仍高但略低于无 Dropout 的最佳点。
- **Dropout** 使最佳验证准确率从 77.14% 降到约 73% 档，反映更强的正则与略低的拟合度；测试集上 exp3 与 exp2 差距不大（71.00 vs 71.70）。
- **SGD（lr=0.01）** 在本组中测试准确率 **71.60%**，与 exp2 接近，优于同结构下的 Adam 重复跑（exp4_opt_adam 70.50%），说明优化器与学习率匹配很重要；不同随机种子下 Adam 复跑也会有波动。
- **AdamW + 较小 lr**（exp5）测试 **70.70%**，未超过 exp2 / exp4_sgd；可继续调 wd 或训练轮数再试。

详细曲线与混淆矩阵见各子目录下 `figures/` 与 `reports/`。
