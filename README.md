# CNN_learning

人工智能原理课程项目二：基于 STL-10 数据集的 CNN 图像分类实验。

本仓库包含训练代码、实验结果、可解释性分析脚本和最终实验报告。最终报告见 `homework/homework_report.pdf`，报告源码见 `homework/homework_report.tex`。

## 目录说明

- `src/`：训练、测试、结果汇总、曲线绘制和 Grad-CAM 可视化代码。
- `homework/`：最终实验报告、报告中引用的对比曲线和 Grad-CAM 图片。
- `outputs_baseline/`：baseline 模型的原始训练输出，包含 checkpoint、训练曲线、分类报告和混淆矩阵。
- `outputs_improved/`：最终报告采用的实验结果，包括汇总表、训练曲线、分类报告和混淆矩阵。
- `人工智能原理课程项目2.pdf`：课程项目要求原文。
- `STL10/`：本地数据集目录，已在 `.gitignore` 中忽略，不提交到 GitHub。

## 环境安装

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

如果需要使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch 版本。

## 常用命令

训练单组实验：

```bash
python src/train.py --data_root STL10 --experiment_id exp6_improved_long_6 --output_root outputs_improved
```

汇总实验结果：

```bash
python src/summarize_results.py --output_root outputs_improved
```

绘制报告中的对比曲线：

```bash
python src/plot_comparison_curves.py
```

生成 Grad-CAM 可视化：

```bash
python src/run_gradcam.py --checkpoint outputs_improved/exp6_improved_long_6/checkpoints/exp6_improved_long_6_best.pth --data_root STL10
```

## 提交说明

本项目使用 Python、PyTorch 与 torchvision 完成。报告中已经说明本次作业使用 OpenAI Codex 进行辅助，主要用于模型搭建、实验脚本调整、批量运行实验、结果汇总和报告整理。
