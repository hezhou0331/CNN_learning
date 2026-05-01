我现在要完成作业任务1：
(1) 分类模型构建与评估（60 分）
请基于 STL‑10 数据集，完整实现一套深度学习图像分类流程。
• 自行设计适用于该数据集的卷积神经网络结构，说明设计思路。（要求至少包含 3 个及以
上卷积层、至少 1 层池化层）
• 选用合适的深度学习框架（推荐使用 PyTorch）完成模型的代码实现；
• 设计合理的实验方案完成模型训练，记录并绘制训练集与验证集的 Loss 及 Accuracy 曲
线。分析模型是否存在过拟合，并展示在最终测试集上的分类报告（包括 Precision、Recall、
F1-score 以及混淆矩阵等多分类指标，以及你考虑到的其它合理指标）。

请你完成以下任务：
1. 使用 PyTorch 训练 baselineCNN
   - 数据集：STL-10
   - 图像尺寸：保持 STL-10 默认尺寸或按代码中已有 transform 设置
   - 数据划分：使用训练集训练，划分一部分训练集作为验证集，最终在 test set 上评估
   - 训练轮数：尽量跑久一些，例如 50~100 epochs；如果验证集 loss 长时间不再下降，可以使用 early stopping
   - 优化器：Adam 或 SGD 均可，请记录具体设置
   - 学习率、batch size、weight decay、scheduler 等超参数请写清楚
2. 训练过程中记录：
   - train loss
   - val loss
   - train accuracy
   - val accuracy

3. 绘制并保存曲线：
   - Loss 曲线
   - Accuracy 曲线

4. 在测试集上输出并保存：
   - overall accuracy
   - precision
   - recall
   - F1-score
   - classification report
   - confusion matrix 图像

5. 保存内容到：
D:\Users\ljy\Desktop\课程学习\2026-春\人工智能原理\作业\人工智能原理课程项目2\outputs_baseline