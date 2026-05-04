# outputs_improved 实验记录（STL-10 分类）

## 任务与数据

在 **STL-10**（96×96 彩色图、10 类）上训练卷积网络。验证集由训练集按固定比例划分；指标以 **测试集准确率** 为主，并记录 **macro / weighted F1**。各次实验除下表所列变量外，其余训练流程一致（如 Cosine 学习率调度、早停耐心等，以 `src/train.py` 中 `EXPERIMENT_PRESETS` 为准）。

**数据增强（`use_augmentation=True`）**：Resize(110) → RandomCrop(96) → RandomHorizontalFlip → ColorJitter → 归一化。

**模型类型**：

- **baseline** / **configurable**：见 `src/models.py`。
- **improved_long**：6 个 `Conv2d`，五段 `MaxPool2d(2)` + `AdaptiveAvgPool2d(1,1)`。
- **improved_longer**：**9** 个 `Conv2d`，与 exp6（`exp6_improved_long_6`）相同下采样节奏但在 24×24 与 12×12 多叠卷积；**无残差**。
- **improved_longer_12**：在 **improved_longer** 基础上再叠 **3** 个 128 通道 3×3 卷积（24×24 多 1 层、12×12 多 2 层），共 **12** 个 `Conv2d`，**无残差**。

---

## 各次实验「改了什么」（节选）

| run | 要点 |
|-----|------|
| **exp6_improved_long_6** | 增强 + `improved_long` + Adam `1e-3` + 头 Dropout 0.3 |
| **exp7_improved_long_9** | 与 exp6 **相同训练配方**，模型换为 **`improved_longer`（9 层卷积）** |
| **exp8_improved_longer_12** | 与 exp7 **相同训练配方**，模型换为 **`improved_longer_12`（12 层卷积）** |

---

## 结果汇总（`comparison_summary`）

| run | 模型 | 测试准确率 (%) | 备注 |
|-----|------|----------------|------|
| exp6_improved_long_6 | improved_long | 80.40 | 验证峰值最高 82.71 |
| **exp7_improved_long_9** | improved_longer | **81.60** | **当前测试最佳** |
| exp8_improved_longer_12 | improved_longer_12 | 69.50 | 验证峰值约 70.29%，第 22 轮早停；12 层在同配方下低于 exp7 |

完整表格见 `outputs_improved/comparison_summary.md`（已运行 `python src/summarize_results.py` 更新）。
