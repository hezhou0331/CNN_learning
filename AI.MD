基于我现在的 CNN 分类模型（STL-10），帮我实现 Grad-CAM 可解释性分析。

要求：

1. 使用 PyTorch 实现

2. 选择最后一个卷积层作为 target layer

3. 输入一张图片，输出：

   - 原图

   - Grad-CAM 热力图

   - 叠加图

4. 代码尽量独立成一个文件，例如 [gradcam.py](http://gradcam.py)

5. 兼容我现在的模型 forward（不要改原模型结构）

请解释：

- Grad-CAM 的核心原理

- 每一步 tensor shape 在干嘛



@[model.py](http://model.py)

@[train.py](http://train.py)

请帮我把 Grad-CAM 接入到我已经训练好的模型中：

1. 加载 checkpoint

2. 从 test 数据集中选一张图片

3. 生成 Grad-CAM 可视化结果

4. 保存为图片（例如 gradcam_result.png）

不要修改现有训练流程，只新增代码



基于我生成的 Grad-CAM 可视化结果，请帮我写“可解释性分析”部分D:\Users\ljy\Desktop\课程学习\2026-春\人工智能原理\作业\人工智能原理课程项目2\homework\homework_report.tex：

要求：

完成"D:\Users\ljy\Desktop\课程学习\2026-春\人工智能原理\作业\人工智能原理课程项目2\人工智能原理课程项目2.pdf"内的实验报告所有要求，达到可以提交的程度，排版参考一下"D:\Users\ljy\Desktop\课程学习\2026-春\人工智能原理\作业\6\6.tex"，内容用我们自己的图并且做一下图表 ，并且自己做好检查和纠错

1. 用中文

2. 不要像 AI，写得自然一点

3. 结合 STL-10 数据集

4. 分析：

   - 模型关注了图像哪些区域

   - 是否合理

   - 是否有错误关注（背景等）

5. 给出结论：模型是否真正学到了语义特征



