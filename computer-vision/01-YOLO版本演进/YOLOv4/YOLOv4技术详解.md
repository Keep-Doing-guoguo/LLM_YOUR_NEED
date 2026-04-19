# YOLOv4 技术详解

YOLOv4 是 YOLO 系列中非常重要的工程化版本。它没有把目标检测改造成全新的范式，而是在 YOLOv3 的基础上系统组合了当时成熟有效的结构、训练策略和后处理方法，目标是让普通 GPU 也能训练出高精度、实时的检测器。

YOLOv4 的核心可以概括为：

```text
CSPDarknet53 + SPP + PANet + YOLO Head
Bag of Freebies + Bag of Specials
CIoU Loss + DIoU-NMS
```

## 一、整体结构

YOLOv4 的模型结构通常可以拆成四部分：

| 部分 | 结构 | 作用 |
|---|---|---|
| Backbone | CSPDarknet53 | 提取图像特征 |
| Neck | SPP + PANet | 扩大感受野并融合多尺度特征 |
| Head | YOLO detection head | 三尺度目标预测 |
| Postprocess | NMS / DIoU-NMS | 去除重复框 |

整体流程如下：

```text
输入图像
  -> CSPDarknet53
  -> SPP
  -> PANet
  -> 三尺度 YOLO Head
  -> bbox 解码
  -> 置信度筛选
  -> NMS / DIoU-NMS
```

YOLOv4 延续了 YOLOv3 的 anchor-based 设计，仍然依赖 anchor 匹配和 NMS 后处理。

## 二、CSPDarknet53

CSPDarknet53 是 YOLOv4 的主干网络。CSP 来自 Cross Stage Partial Network，核心思想是把特征分成两部分：一部分经过残差/卷积堆叠，另一部分跨阶段连接，最后再融合。

简化结构如下：

```text
输入特征
  -> 分支 A: 多个残差块
  -> 分支 B: shortcut / partial connection
  -> concat
  -> conv 融合
```

相比 YOLOv3 的 Darknet-53，CSPDarknet53 的优势是：

| 优势 | 说明 |
|---|---|
| 减少重复梯度信息 | 降低冗余计算 |
| 提升训练稳定性 | 梯度流更顺畅 |
| 精度和速度更平衡 | 适合实时检测 |

## 三、SPP 模块

YOLOv4 在 neck 中使用 SPP。它通过不同大小的最大池化核增强深层特征的感受野。

常见池化核：

```text
5x5, 9x9, 13x13
```

SPP 的作用：

| 作用 | 说明 |
|---|---|
| 扩大感受野 | 捕获更大范围上下文 |
| 不改变空间尺寸 | stride=1 且 padding 合理 |
| 增强深层语义 | 对大目标和复杂背景更有帮助 |

## 四、PANet 特征融合

YOLOv4 使用 PANet 思想做多尺度特征融合。它不仅有自顶向下的语义传递，也有自底向上的定位信息回流。

简化理解：

```text
高层语义特征 -> 上采样 -> 与中低层特征融合
低层定位特征 -> 下采样 -> 再向高层融合
```

这比单纯 FPN 更强调不同尺度之间的双向信息传递。

## 五、五大核心改进

YOLOv4 经常被概括为五个核心技术点：

| 技术 | 类型 | 作用 |
|---|---|---|
| Mosaic | Bag of Freebies | 数据增强，提升鲁棒性 |
| CSPDarknet53 | Backbone | 降低冗余计算，增强特征提取 |
| SPP | Neck | 扩大感受野 |
| PANet | Neck | 多尺度特征融合 |
| CIoU Loss / DIoU-NMS | Loss / Postprocess | 提升定位和后处理质量 |

其中 Mosaic、CIoU、DIoU-NMS 是训练和后处理层面的提升；CSPDarknet53、SPP、PANet 是结构层面的提升。

## 六、Mosaic 数据增强

Mosaic 会把四张图拼接成一张训练图像，同时调整对应标注框。

简化流程：

```text
随机选择 4 张图
  -> 随机缩放和裁剪
  -> 拼接到一张画布
  -> 修正 bbox 坐标
  -> 输入模型训练
```

它的主要价值是：

| 价值 | 说明 |
|---|---|
| 增加上下文变化 | 一张图里出现更多场景组合 |
| 增加小目标样本 | 缩放拼接后小目标更多 |
| 降低 batch size 依赖 | 单张训练图包含更丰富信息 |

## 七、正负样本划分

YOLOv4 仍然是 anchor-based 检测器。每个 ground truth 通常会根据宽高形状与 anchor 的匹配程度，分配到最合适的检测层和 anchor。

流程如下：

```text
读取 ground truth
  -> 计算 GT 与 anchor 的形状匹配
  -> 选择最佳 anchor
  -> 根据 GT 中心点找到 grid cell
  -> 标记为正样本
  -> 高 IoU 但非最佳预测可设为 ignore
  -> 其余候选作为负样本训练 objectness
```

各类样本作用：

| 样本 | 参与定位损失 | 参与 objectness | 参与分类损失 |
|---|---|---|---|
| 正样本 | 是 | 是，目标为 1 | 是 |
| 负样本 | 否 | 是，目标为 0 | 否 |
| ignore 样本 | 否 | 通常忽略 | 否 |

## 八、损失函数

YOLOv4 的损失通常包括：

```text
定位损失 + objectness 损失 + 分类损失
```

定位损失常使用 CIoU。CIoU 相比普通 IoU，不只关注重叠面积，还考虑中心点距离和宽高比。

简化理解：

```text
CIoU = IoU 项 + 中心距离惩罚 + 宽高比惩罚
```

CIoU 的优势是：当预测框和真实框重叠较少时，仍然能提供更明确的优化方向。

## 九、DIoU-NMS

普通 NMS 只根据 IoU 判断两个框是否重复。DIoU-NMS 会额外考虑框中心点距离，使得重叠较高但中心相距较远的框更有机会被保留。

它适合缓解部分密集目标场景中的误删问题，但仍然属于后处理策略，不改变模型本身的预测方式。

## 十、训练与推理流程

训练流程：

```text
数据增强
  -> 前向传播
  -> anchor 匹配
  -> 构造正负样本
  -> 计算 CIoU / objectness / class loss
  -> 反向传播
```

推理流程：

```text
输入图像
  -> 前向传播
  -> 三尺度输出解码
  -> score = objectness * class probability
  -> 置信度筛选
  -> NMS / DIoU-NMS
  -> 输出最终检测框
```

## 十一、YOLOv4 的贡献与局限

贡献：

| 贡献 | 说明 |
|---|---|
| 工程组合能力强 | 把大量有效技巧整合成稳定检测器 |
| 训练策略成熟 | Mosaic、label smoothing、CIoU 等提高训练效果 |
| 多尺度能力强 | SPP + PANet 改善不同尺寸目标检测 |
| 仍保持实时性 | 适合 GPU 实时部署 |

局限：

| 局限 | 说明 |
|---|---|
| 仍依赖 anchor | 自定义数据集可能需要重新聚类 |
| 仍依赖 NMS | 密集目标可能误删 |
| 标签分配较传统 | 没有 SimOTA/TAL 这类动态分配 |
| 检测头较旧 | 不是现代解耦头结构 |

## 十二、学习重点

学习 YOLOv4 时，重点掌握：

1. CSPDarknet53 为什么能减少冗余计算。
2. SPP 和 PANet 分别解决什么问题。
3. Mosaic 为什么能提高训练鲁棒性。
4. CIoU 相比普通 IoU 多考虑了什么。
5. YOLOv4 的正负样本划分为什么仍然属于传统 anchor-based 机制。
6. 哪些改进属于模型结构，哪些属于训练策略或后处理。

## 十三、结论

YOLOv4 是一个典型的工程集成型检测器。它的重要性不在于提出单一革命性结构，而在于把 backbone、neck、loss、数据增强和后处理组合成了一个性能很强、可训练、可部署的目标检测系统。

理解 YOLOv4 后，再看 YOLOv5、YOLOv6、YOLOv7 中的 CSP、PAN、Mosaic、CIoU、动态标签分配和部署优化，会更容易看出技术演进的连续性。
