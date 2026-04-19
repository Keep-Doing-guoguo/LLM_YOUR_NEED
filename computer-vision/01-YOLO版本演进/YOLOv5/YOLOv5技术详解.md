# YOLOv5 技术详解

YOLOv5 是 Ultralytics 开源的 YOLO 系列工程实现。它没有正式论文，但在工程落地、训练易用性、模型导出和部署支持方面影响很大。

YOLOv5 的核心特点可以概括为：

```text
CSP-style backbone + PAN/FPN neck + anchor-based YOLO head
Mosaic / AutoAnchor / 多尺度训练
CIoU loss + BCE loss + NMS
完善的 PyTorch 训练、推理、导出工具链
```

## 一、整体结构

以 YOLOv5s 为例，模型一般由三部分组成：

| 部分 | 典型模块 | 作用 |
|---|---|---|
| Backbone | Focus/Conv、CSP/C3、SPPF | 特征提取 |
| Neck | FPN + PAN | 多尺度特征融合 |
| Head | YOLO Detect | 三尺度预测 |

整体流程：

```text
输入图像
  -> Backbone 提取特征
  -> SPPF 扩大感受野
  -> FPN/PAN 融合多尺度特征
  -> Detect Head 输出预测
  -> 解码 bbox
  -> NMS
```

## 二、模型结构

YOLOv5 早期使用 Focus 模块进行下采样，后续版本中常用普通卷积替代。C3 模块是 YOLOv5 中很重要的 CSP-style 结构，用于在较低计算成本下保留梯度流。

常见结构：

| 模块 | 作用 |
|---|---|
| Conv | 卷积、BN、SiLU 激活 |
| C3 | CSP-style 特征提取 |
| SPPF | 快速空间金字塔池化 |
| Upsample | 上采样 |
| Concat | 特征拼接 |
| Detect | anchor-based 检测头 |

需要注意：YOLOv5 的检测头通常不是现代意义上的解耦头。它仍然是 anchor-based Detect head，分类、objectness 和 bbox 预测在同一个检测输出中组织。

## 三、输出格式

以 COCO 80 类、输入 640x640 为例，YOLOv5 通常在三个尺度输出：

| 层级 | stride | 输出尺寸 | 负责目标 |
|---|---:|---:|---|
| P3 | 8 | 80x80 | 小目标 |
| P4 | 16 | 40x40 | 中等目标 |
| P5 | 32 | 20x20 | 大目标 |

每个 anchor 输出：

```text
x, y, w, h, objectness, class_1, ..., class_80
```

维度为 `85`。如果每个尺度有 3 个 anchor，则每个尺度每个网格预测 3 个候选框。

## 四、数据增强

YOLOv5 的训练工程中常见增强包括：

| 增强 | 作用 |
|---|---|
| Mosaic | 四图拼接，提升上下文和小目标样本丰富度 |
| MixUp | 图像混合，增强泛化 |
| HSV 增强 | 调整色彩分布 |
| 随机翻转 | 增强空间变化 |
| 多尺度训练 | 提升输入尺寸鲁棒性 |

这些增强是 YOLOv5 工程效果的重要组成部分。

## 五、Anchor 与 AutoAnchor

YOLOv5 仍然是 anchor-based 检测器。它支持 AutoAnchor，用于检查默认 anchor 是否适合当前数据集，并在需要时重新聚类。

AutoAnchor 的基本流程：

```text
读取训练集标注
  -> 提取 bbox 宽高
  -> 评估默认 anchor 匹配程度
  -> 必要时执行 K-Means / 遗传优化
  -> 更新 anchor
```

自定义数据集目标尺寸和 COCO 差异较大时，AutoAnchor 很有价值。

## 六、正负样本分配

YOLOv5 的传统标签分配不是 SimOTA，而是基于 anchor 宽高比例和网格邻域扩展的静态匹配。

简化流程：

```text
遍历 ground truth
  -> 与每层 anchor 计算宽高比例
  -> 满足 anchor_t 阈值的 anchor 成为候选正样本
  -> 将 GT 分配到对应 grid cell
  -> 对靠近网格边界的 GT 扩展到相邻 grid
```

这与 YOLOv3/YOLOv4 “每个 GT 主要匹配最佳 anchor” 不完全相同。YOLOv5 可能让一个 GT 匹配多个 anchor 或邻近网格，从而增加正样本数量。

关于 SimOTA：SimOTA 主要来自 YOLOX。标准 Ultralytics YOLOv5 训练流程通常不使用 SimOTA。部分第三方改版可能引入动态标签分配，但不能作为 YOLOv5 标准机制。

## 七、损失函数

YOLOv5 的损失一般包括：

```text
box loss + objectness loss + classification loss
```

常见设置：

| 损失 | 说明 |
|---|---|
| box loss | 通常基于 CIoU 等 IoU 系列损失 |
| objectness loss | BCEWithLogitsLoss |
| classification loss | BCEWithLogitsLoss |

YOLOv5 不使用 DFL 作为标准框回归损失。DFL 是后续 YOLOv8 等 anchor-free 检测器中更典型的设计。

## 八、NMS 后处理

YOLOv5 推理阶段仍然依赖 NMS。

流程：

```text
模型输出
  -> 解码预测框
  -> score = objectness * class probability
  -> 置信度过滤
  -> 按类别或类别无关方式执行 NMS
  -> 输出最终检测结果
```

常用参数包括 `conf_thres` 和 `iou_thres`。

## 九、部署与导出

YOLOv5 的一个重要优势是部署工具链完整。常见导出格式：

| 格式 | 用途 |
|---|---|
| TorchScript | PyTorch 生态部署 |
| ONNX | 跨框架中间表示 |
| TensorRT | NVIDIA GPU 加速 |
| OpenVINO | Intel 设备推理 |
| CoreML | Apple 生态 |
| TFLite | 移动端或边缘端 |

示例命令：

```bash
python export.py --weights yolov5s.pt --include onnx
```

部署时要保证预处理、输入尺寸、anchor、类别数和 NMS 参数一致。

## 十、训练与推理流程

训练流程：

```text
加载数据
  -> Mosaic/MixUp/HSV 等增强
  -> 前向传播
  -> anchor 匹配与邻域扩展
  -> 计算 box/objectness/class loss
  -> 反向传播
```

推理流程：

```text
letterbox resize
  -> 归一化
  -> 模型前向
  -> 解码输出
  -> NMS
  -> 坐标映射回原图
```

## 十一、优点与局限

优点：

| 优点 | 说明 |
|---|---|
| 工程成熟 | 训练、推理、导出工具完整 |
| 易用性强 | 配置简单，社区资料多 |
| 部署友好 | 支持多种导出格式 |
| 自定义数据集方便 | AutoAnchor 和数据配置较完善 |

局限：

| 局限 | 说明 |
|---|---|
| 仍依赖 anchor | 需要考虑 anchor 适配问题 |
| 仍依赖 NMS | 不是端到端检测 |
| 标签分配不如后续动态方法灵活 | 没有标准 TAL/SimOTA |
| 没有官方论文 | 理论描述主要来自代码和文档 |

## 十二、学习重点

学习 YOLOv5 时重点掌握：

1. C3/SPPF/PAN/Detect 的作用。
2. YOLOv5 的 anchor 匹配与 YOLOv3/YOLOv4 有什么不同。
3. AutoAnchor 什么时候有必要。
4. box/objectness/class 三类 loss 如何作用。
5. 导出 ONNX 或 TensorRT 时，哪些后处理细节容易出错。

## 十三、结论

YOLOv5 的价值主要体现在工程化。它把 YOLO 训练、部署、模型缩放、自定义数据集支持做得非常完整，是很多实际项目选择 YOLO 系列的关键原因。
