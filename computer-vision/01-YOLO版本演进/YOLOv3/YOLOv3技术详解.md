# YOLOv3 技术详解

YOLOv3 出自论文《YOLOv3: An Incremental Improvement》，它不是一次彻底推翻 YOLOv1/YOLOv2 的重构，而是在 YOLOv2 的 anchor-based 检测框架上，把主干网络、多尺度检测、分类方式和训练细节做了系统升级。

从学习路线看，YOLOv3 最值得掌握的不是某一个孤立模块，而是下面这条完整链路：

```text
输入图像
  -> Darknet-53 提取特征
  -> 多尺度特征融合
  -> 三个检测头输出预测
  -> anchor 与 ground truth 匹配
  -> 计算定位、目标置信度、分类损失
  -> 推理时解码预测框
  -> 置信度筛选
  -> NMS 去重
  -> 得到最终检测结果
```

因此，把网络结构、正负样本划分、IoU、NMS 放在同一篇文档里更合适。它们不是四个互不相关的知识点，而是 YOLOv3 训练和推理流程中的连续环节。

## 一、YOLOv3 解决了什么问题

相比 YOLOv1 和 YOLOv2，YOLOv3 的核心改进主要有四点。

第一，主干网络从 Darknet-19 升级为 Darknet-53。Darknet-53 引入了残差连接，网络更深，特征表达能力更强。

第二，YOLOv3 使用三尺度检测。它分别在 13x13、26x26、52x52 三个尺度上预测目标，使模型对大目标、中等目标、小目标都有更好的覆盖。

第三，YOLOv3 不再使用 softmax 做多类别互斥分类，而是对每个类别使用独立的 logistic 分类器。这样更适合多标签场景，虽然常规目标检测数据集大多仍然是单标签目标。

第四，YOLOv3 的检测头仍然依赖 anchor 和 NMS，但整体预测质量相比 YOLOv2 更稳定，尤其是小目标检测能力明显提升。

需要注意：YOLOv3 仍然是 anchor-based、NMS-based 的检测器。它还没有后续版本中的解耦头、动态标签分配、DFL、anchor-free 或端到端去 NMS 设计。

## 二、整体架构

YOLOv3 可以拆成三个部分：

```text
Backbone: Darknet-53
Neck:     类 FPN 的多尺度特征融合
Head:     三个 YOLO detection head
```

以输入尺寸 416x416 为例，YOLOv3 的三个输出尺度通常是：

| 检测尺度 | stride | 特征图尺寸 | 主要负责目标 |
|---|---:|---:|---|
| 大尺度特征图 | 8 | 52x52 | 小目标 |
| 中尺度特征图 | 16 | 26x26 | 中等目标 |
| 小尺度特征图 | 32 | 13x13 | 大目标 |

这里的“大尺度特征图”指特征图分辨率更大，不是目标尺寸更大。52x52 的网格更密，适合定位小目标；13x13 的网格更粗，但语义更强，适合检测大目标。

## 三、Darknet-53 主干网络

Darknet-53 是 YOLOv3 的 backbone。它由卷积层、批归一化、Leaky ReLU 和残差块组成。

一个典型残差块可以理解为：

```text
输入 x
  -> 1x1 Conv 降维
  -> 3x3 Conv 提取空间特征
  -> 与输入 x 相加
输出 y
```

简化代码如下：

```python
import torch
import torch.nn as nn


class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DarknetResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.layers = nn.Sequential(
            ConvBNLeaky(channels, hidden, kernel_size=1),
            ConvBNLeaky(hidden, channels, kernel_size=3),
        )

    def forward(self, x):
        return x + self.layers(x)
```

Darknet-53 的优势是：

| 特点 | 说明 |
|---|---|
| 更深 | 相比 Darknet-19 表达能力更强 |
| 残差连接 | 缓解深层网络训练困难 |
| 全卷积结构 | 可以适配不同输入尺寸 |
| 速度友好 | 主要由标准卷积构成，工程实现简单 |

## 四、多尺度检测结构

YOLOv2 主要在单个尺度上做检测，YOLOv3 引入三尺度预测。它的思路和 FPN 类似：深层特征语义强，浅层特征细节多，把深层特征上采样后与浅层特征拼接，从而增强小目标检测。

简化流程如下：

```text
Darknet-53 输出深层特征
  -> 13x13 检测头预测大目标
  -> 上采样，与 26x26 中层特征拼接
  -> 26x26 检测头预测中等目标
  -> 再上采样，与 52x52 浅层特征拼接
  -> 52x52 检测头预测小目标
```

这也是 YOLOv3 相比 YOLOv2 小目标能力提升的重要原因。

## 五、输出张量格式

YOLOv3 每个尺度通常分配 3 个 anchor。对于 COCO 数据集，类别数为 80，因此每个 anchor 输出：

```text
tx, ty, tw, th, objectness, class_1, class_2, ..., class_80
```

也就是：

```text
4 + 1 + 80 = 85
```

每个尺度有 3 个 anchor，因此检测头输出通道数为：

```text
3 x 85 = 255
```

以 416x416 输入为例：

| 检测尺度 | 输出特征图 | 输出通道 | 展开后预测框数量 |
|---|---:|---:|---:|
| 大目标检测 | 13x13 | 255 | 13 x 13 x 3 = 507 |
| 中目标检测 | 26x26 | 255 | 26 x 26 x 3 = 2028 |
| 小目标检测 | 52x52 | 255 | 52 x 52 x 3 = 8112 |

总预测框数量为：

```text
507 + 2028 + 8112 = 10647
```

也就是说，对于一张 416x416 图像，YOLOv3 推理时会先产生 10647 个候选框，再通过置信度筛选和 NMS 得到最终结果。

## 六、Anchor 设计

YOLOv3 继承了 YOLOv2 的 anchor 思路。anchor 的尺寸通常通过训练集标注框聚类得到。

COCO 上常见的 9 个 anchor 为：

| 尺度 | anchor 尺寸 |
|---|---|
| 52x52 | 10x13, 16x30, 33x23 |
| 26x26 | 30x61, 62x45, 59x119 |
| 13x13 | 116x90, 156x198, 373x326 |

小 anchor 分配给高分辨率特征图，大 anchor 分配给低分辨率特征图。这符合目标检测中的基本直觉：

```text
小目标 -> 需要更密的网格 -> 52x52
大目标 -> 需要更大的感受野 -> 13x13
```

## 七、边界框解码

YOLOv3 的检测头不会直接输出最终框坐标，而是输出相对于网格和 anchor 的偏移量。

假设当前网格左上角坐标为 `(cx, cy)`，anchor 尺寸为 `(pw, ph)`，网络输出为 `(tx, ty, tw, th)`，则解码方式为：

```text
bx = sigmoid(tx) + cx
by = sigmoid(ty) + cy
bw = pw * exp(tw)
bh = ph * exp(th)
```

其中：

| 符号 | 含义 |
|---|---|
| `tx, ty` | 模型预测的中心点偏移 |
| `tw, th` | 模型预测的宽高缩放 |
| `cx, cy` | 当前 grid cell 的坐标 |
| `pw, ph` | anchor 的宽高 |
| `bx, by, bw, bh` | 解码后的预测框 |

`sigmoid(tx)` 和 `sigmoid(ty)` 会把中心点限制在当前网格附近，这使训练更稳定。`exp(tw)` 和 `exp(th)` 用于在 anchor 尺寸基础上做比例缩放。

## 八、正负样本划分

YOLOv3 的样本划分围绕 anchor 展开。每个 ground truth 框通常只会分配给一个最佳 anchor。

整体流程如下：

```text
遍历每个 ground truth
  -> 计算它与 9 个 anchor 的形状 IoU
  -> 选择 IoU 最大的 anchor
  -> 根据该 anchor 所属尺度找到检测头
  -> 根据 ground truth 中心点找到 grid cell
  -> 将该 grid cell 上的该 anchor 标记为正样本
```

这里计算 anchor 匹配时通常只比较宽高形状，不关心图像中的绝对位置。因为 anchor 的作用是匹配目标尺寸，而不是匹配目标坐标。

### 1. 正样本

正样本是被某个 ground truth 分配到的 anchor 位置。

正样本需要参与：

| 损失项 | 是否参与 |
|---|---|
| 定位损失 | 是 |
| objectness 正样本损失 | 是 |
| 分类损失 | 是 |

### 2. 负样本

负样本是没有被任何 ground truth 分配，并且不满足 ignore 条件的预测框。

负样本主要参与 objectness 的背景损失，训练目标是让它的目标置信度接近 0。

### 3. Ignore 样本

ignore 样本既不是正样本，也不作为普通负样本惩罚。

典型情况是：某个预测框虽然不是最佳 anchor，但它与某个 ground truth 的 IoU 较高。如果强行把它当负样本，会惩罚一个其实比较合理的预测框，因此实现中常把它从 no-object loss 中忽略。

常见配置中会出现类似：

```text
ignore_thresh = 0.5
truth_thresh = 1
```

这些阈值是 Darknet 实现细节，不同代码库可能略有差异。

## 九、损失函数

YOLOv3 的 loss 可以分为三类：

```text
定位损失
objectness 损失
分类损失
```

### 1. 定位损失

定位损失只对正样本计算，用来约束预测框的位置和尺寸。

原始 YOLOv3 论文并不是用后来常见的 CIoU、DFL 这类现代框回归损失来训练。许多 Darknet 系实现会对网络输出的坐标参数进行回归，后来的一些改进版 Darknet 或复现工程才加入 GIoU、DIoU、CIoU 等选项。

因此学习时要区分：

| 类型 | 是否属于原始 YOLOv3 核心设计 |
|---|---|
| anchor 匹配中的 IoU | 是 |
| NMS 中的 IoU | 是 |
| 评估指标中的 IoU | 是 |
| GIoU/DIoU/CIoU loss | 不是原始论文核心设计，属于后续实现扩展 |
| DFL | 不是 YOLOv3 设计 |

### 2. Objectness 损失

objectness 表示当前 anchor 位置是否负责一个真实目标。

正样本的 objectness 目标为 1，普通负样本的 objectness 目标为 0，ignore 样本通常不参与 no-object 惩罚。

### 3. 分类损失

YOLOv3 使用独立 logistic 分类器，而不是 softmax。

这意味着每个类别都会独立预测一个概率：

```text
P(class_1), P(class_2), ..., P(class_C)
```

这种设计允许一个框同时属于多个类别。虽然在 COCO 这类常规检测任务中，一个实例通常只有一个类别，但独立 logistic 分类器让 YOLOv3 的分类头更灵活。

## 十、IoU 在 YOLOv3 中的作用

IoU 是目标检测中的基础度量。它表示预测框和真实框的重叠程度：

```text
IoU = 交集面积 / 并集面积
```

在 YOLOv3 中，IoU 主要出现在四个位置。

| 位置 | 作用 |
|---|---|
| anchor 匹配 | 判断 ground truth 最适合哪个 anchor |
| ignore 机制 | 高 IoU 但非最佳预测可忽略 no-object 惩罚 |
| NMS | 判断两个候选框是否重复 |
| 模型评估 | mAP 计算依赖 IoU 阈值 |

后续实现中常见的 IoU 改进包括：

| 方法 | 核心思想 |
|---|---|
| GIoU | 在无重叠时也提供优化方向 |
| DIoU | 额外考虑中心点距离 |
| CIoU | 同时考虑重叠、中心距离和宽高比 |

这些方法对理解现代 YOLO 很重要，但不应误认为是 YOLOv3 原始论文的标准配置。

## 十一、NMS 推理后处理

YOLOv3 推理阶段会产生大量候选框，因此需要 NMS 去除重复框。

标准流程如下：

```text
1. 解码三个检测头的所有预测框
2. 计算每个框的最终得分
3. 根据置信度阈值过滤低分框
4. 按类别分别执行 NMS
5. 输出保留下来的检测框
```

YOLOv3 中常用的最终得分为：

```text
score = objectness * class_probability
```

标准贪心 NMS 的过程是：

```text
按 score 从高到低排序
选择当前最高分框
删除与它 IoU 大于阈值的同类框
继续处理剩余框
直到没有候选框
```

简化代码示意：

```python
def greedy_nms(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return keep
```

需要注意：YOLOv3 原始论文没有提出新的 NMS 算法。Soft-NMS、DIoU-NMS、batched NMS 等属于后续工程或其他论文中的改进。

## 十二、训练流程

YOLOv3 的训练可以概括为：

```text
读取图像和标注
  -> 数据增强与 resize
  -> 前向传播得到三个尺度预测
  -> 对每个 ground truth 分配最佳 anchor
  -> 构造正样本、负样本、ignore 样本
  -> 计算定位、objectness、分类损失
  -> 反向传播更新模型
```

训练时最容易混淆的是：模型输出的 10647 个候选位置，并不是全部都要回归真实框。只有正样本负责定位和分类；大量普通负样本主要用于训练 objectness；ignore 样本则避免模型过度惩罚一些合理但非最佳的预测。

## 十三、推理流程

YOLOv3 的推理流程如下：

```text
输入图像 resize 到指定尺寸
  -> 模型前向传播
  -> 得到 13x13、26x26、52x52 三个输出
  -> 解码 anchor box
  -> 计算 objectness 与类别概率
  -> 置信度筛选
  -> 每个类别分别 NMS
  -> 坐标映射回原图
  -> 输出最终检测结果
```

推理阶段不再需要正负样本划分。正负样本划分只发生在训练阶段，用于构造 loss。

## 十四、YOLOv3 的优点

YOLOv3 的优点主要包括：

| 优点 | 说明 |
|---|---|
| 速度快 | 保持 YOLO 系列单阶段检测优势 |
| 多尺度检测 | 小目标能力相比 YOLOv2 明显提升 |
| backbone 更强 | Darknet-53 提供更强特征表达 |
| 工程简单 | anchor、head、NMS 逻辑清晰 |
| 影响深远 | 后续 YOLOv4、YOLOv5 等都继承了很多思路 |

## 十五、YOLOv3 的局限性

YOLOv3 也有明显局限。

| 局限 | 说明 |
|---|---|
| 依赖 anchor | 需要聚类或手动设计 anchor |
| 依赖 NMS | 密集目标、遮挡目标下可能误删 |
| 标签分配较简单 | 一个 GT 主要匹配一个最佳 anchor |
| 检测头未解耦 | 分类和回归仍然耦合较强 |
| 框回归不够现代 | 没有 DFL、任务对齐分配等后续设计 |
| 端到端能力不足 | 推理仍需要后处理 |

这些问题正是后续 YOLOv4、YOLOv5、YOLOv7、YOLOv8、YOLOv10 等版本持续改进的方向。

## 十六、与后续 YOLO 的关系

理解 YOLOv3 后，再看后续版本会更清楚。

| 后续方向 | 与 YOLOv3 的关系 |
|---|---|
| 更强 backbone | 在 Darknet-53 思路上继续提升特征提取能力 |
| PAN/FPN 改进 | 强化多尺度特征融合 |
| 解耦头 | 将分类和回归分支拆开 |
| anchor-free | 减少 anchor 设计依赖 |
| 动态标签分配 | 替代固定最佳 anchor 分配 |
| IoU 系列 loss | 改进边界框回归目标 |
| DFL | 使用分布建模提升框定位精度 |
| NMS-free | 试图减少或移除后处理依赖 |

因此，YOLOv3 是从早期 YOLO 走向现代 YOLO 的关键中间版本。

## 十七、学习重点总结

学习 YOLOv3 时，建议重点掌握下面几个问题：

1. 为什么 YOLOv3 要做三尺度检测？
2. 13x13、26x26、52x52 分别负责什么目标？
3. 每个检测头为什么输出 255 个通道？
4. anchor 是如何和 ground truth 匹配的？
5. 正样本、负样本、ignore 样本分别参与哪些 loss？
6. IoU 在 anchor 匹配、ignore、NMS 和评估中分别起什么作用？
7. 推理阶段为什么需要 NMS？
8. 原始 YOLOv3 和后续 Darknet 改进版在 IoU loss 上有什么区别？

如果能把这些问题串起来，就基本理解了 YOLOv3 的主干逻辑。

## 十八、结论

YOLOv3 的核心价值在于：它把更强的残差主干网络、多尺度检测、anchor 机制和高效后处理组合成了一个实用的单阶段目标检测系统。

它不是最现代的 YOLO，但它非常适合作为理解现代 YOLO 的分水岭版本。YOLOv1 让检测变成单阶段回归问题，YOLOv2 引入 anchor 和更稳定的训练方式，YOLOv3 则把多尺度检测和更强 backbone 变成 YOLO 系列的重要基础。

后续版本中的很多概念，例如 FPN/PAN、anchor-free、解耦头、IoU loss、动态标签分配、NMS-free，都可以从 YOLOv3 的设计局限中找到演进动机。
