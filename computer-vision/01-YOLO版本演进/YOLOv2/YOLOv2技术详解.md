# YOLOv2 技术详解：Anchor、正负样本与 NMS

## 一、前言

YOLOv2，也称 **YOLO9000: Better, Faster, Stronger**，是 YOLOv1 之后的重要版本。它在保持实时检测速度的同时，明显提升了定位精度和召回率。

YOLOv2 相比 YOLOv1 的关键变化包括：

- 引入 Batch Normalization
- 使用高分辨率分类预训练
- 引入 Anchor Boxes
- 使用 K-Means 聚类生成 anchors
- 采用直接位置预测
- 支持多尺度训练
- 改进分类体系，扩展到 YOLO9000

本文重点合并说明 YOLOv2 中最关键的几个机制：

```text
Anchor Box
正负样本分配
Bounding Box 解码
NMS 后处理
与 YOLOv1 的区别
```

---

## 二、YOLOv2 解决了 YOLOv1 的哪些问题

YOLOv1 的核心问题：

| 问题 | 说明 |
|------|------|
| 没有 anchor | 每个 grid cell 直接预测固定数量框，形状先验不足 |
| 定位不够准 | 框回归难度较高 |
| 召回率有限 | 每个 cell 只能预测 2 个框 |
| 小目标较弱 | 网格较粗，密集目标处理困难 |
| 训练不够稳定 | 没有 BN 等现代训练技巧 |

YOLOv2 的改进方向：

```text
用 anchor 提供形状先验
用 K-Means 聚类得到更合理的 anchor
用 sigmoid 限制中心点预测范围
用多尺度训练增强不同输入尺寸适应能力
```

---

## 三、YOLOv2 的输入与输出

YOLOv2 通常使用：

```text
输入尺寸：416 x 416
输出网格：13 x 13
每个 grid cell：预测 5 个 anchor boxes
```

如果类别数为 `C`，每个 anchor 输出：

```text
tx, ty, tw, th, objectness, class_probs
```

最终输出通道数：

```text
B * (5 + C)
```

如果使用 VOC 20 类：

```text
5 * (5 + 20) = 125
```

输出张量：

```text
13 x 13 x 125
```

如果使用 COCO 80 类：

```text
5 * (5 + 80) = 425
```

---

## 四、Anchor Box 是什么

Anchor Box 是模型预设的一组参考框模板，用于覆盖数据集中常见目标的宽高比例。

简单理解：

```text
Anchor Box 是起点
Bounding Box 是模型基于 anchor 调整后的最终预测框
```

对比：

| 名称 | 说明 |
|------|------|
| Anchor Box | 预定义的参考框模板 |
| Ground Truth Box | 数据集人工标注的真实框 |
| Predicted Bounding Box | 模型最终预测出的目标框 |

YOLOv2 中，每个 grid cell 默认预测 5 个 anchor：

```text
cell (i, j)
  -> anchor 1
  -> anchor 2
  -> anchor 3
  -> anchor 4
  -> anchor 5
```

每个 anchor 会预测一个 bounding box。

---

## 五、为什么 YOLOv2 引入 Anchor

YOLOv1 直接预测框的位置和大小，没有显式先验。

这会带来问题：

- 不同目标形状差异大，直接回归难度高。
- 每个 cell 只能预测少量框。
- 多目标、密集目标召回率低。

Anchor 的好处：

| 好处 | 说明 |
|------|------|
| 降低回归难度 | 模型只需要预测相对 anchor 的偏移 |
| 提升召回率 | 每个 cell 可以预测多个不同形状的框 |
| 更适合多尺度目标 | 不同 anchor 负责不同宽高比例 |
| 训练更稳定 | 框预测有更明确的初始化参考 |

---

## 六、YOLOv2 的 Anchor 如何得到

YOLOv2 使用 K-Means 对训练集中的真实框宽高进行聚类。

不同于普通 K-Means 使用欧氏距离，YOLOv2 使用基于 IoU 的距离：

```text
distance(box, centroid) = 1 - IoU(box, centroid)
```

这样聚类结果更符合检测任务。

流程：

```text
收集训练集中所有 GT 框的宽高
  -> 归一化到相同尺度
  -> 使用 K-Means 聚类
  -> 得到 K 个 anchor 尺寸
```

论文中常用：

```text
K = 5
```

---

## 七、YOLOv2 的边界框解码

YOLOv2 不直接预测最终框坐标，而是预测：

```text
tx, ty, tw, th
```

设：

- 当前 grid cell 左上角坐标为 `(cx, cy)`
- anchor 宽高为 `(pw, ph)`
- 模型输出为 `(tx, ty, tw, th)`

解码公式：

```text
bx = sigmoid(tx) + cx
by = sigmoid(ty) + cy
bw = pw * exp(tw)
bh = ph * exp(th)
```

归一化到整图时：

```text
x = bx / grid_width
y = by / grid_height
w = bw / image_width
h = bh / image_height
```

其中：

- `sigmoid(tx), sigmoid(ty)` 让中心点落在当前 cell 内
- `exp(tw), exp(th)` 让宽高相对于 anchor 缩放

这就是 YOLOv2 中所谓的 **direct location prediction**。

---

## 八、YOLOv2 正负样本分配

正负样本分配决定哪些预测框参与哪些损失。

### 8.1 正样本

满足以下条件的 anchor 被视为正样本：

1. 某个 ground truth 的中心点落在某个 grid cell 内。
2. 在该 cell 的多个 anchors 中，选择与该 ground truth 形状最匹配的 anchor。
3. 通常选择 IoU 最大的 anchor 作为正样本。

也就是说：

```text
每个 GT 通常只分配给一个最佳 anchor
```

### 8.2 负样本

负样本通常是：

```text
没有负责任何 GT 的 anchor
并且与所有 GT 的 IoU 低于 ignore 阈值
```

这些 anchor 主要用于训练 objectness 为 0。

### 8.3 Ignore 样本

如果某个 anchor 没有被分配为正样本，但它与某个 GT 的 IoU 较高，通常不会强行作为负样本。

这类样本可以视为 ignore：

```text
既不作为正样本
也不作为负样本参与 objectness 负样本损失
```

目的：

- 避免惩罚潜在合理预测框
- 让训练更稳定
- 减少正负样本冲突

---

## 九、样本分配示意

```text
Grid Cell (13 x 13)
  |
  |-- Anchor 1: 与 GT IoU 最大 -> 正样本
  |-- Anchor 2: IoU 很低 -> 负样本
  |-- Anchor 3: IoU 中等偏高 -> ignore
  |-- Anchor 4: IoU 很低 -> 负样本
  |-- Anchor 5: IoU 很低 -> 负样本
```

与 YOLOv1 对比：

| 项目 | YOLOv1 | YOLOv2 |
|------|--------|--------|
| Anchor | 无 | 有 |
| 每个 cell 预测框数量 | 2 | 5 |
| 正样本选择 | cell 内 IoU 最大框 | cell 内最佳 anchor |
| 框回归方式 | 直接预测框 | 基于 anchor 预测偏移 |
| 忽略样本 | 较少强调 | 更明确 |

---

## 十、YOLOv2 的损失函数概念

YOLOv2 的损失可以粗略分为：

```text
坐标损失
objectness 损失
分类损失
```

其中：

| 损失项 | 作用 |
|--------|------|
| 坐标损失 | 约束预测框位置和尺寸 |
| objectness 损失 | 判断该 anchor 是否负责目标 |
| 分类损失 | 判断目标类别 |

正样本参与：

```text
坐标损失 + objectness 损失 + 分类损失
```

负样本主要参与：

```text
objectness 损失
```

ignore 样本：

```text
通常不参与负样本 objectness 惩罚
```

---

## 十一、什么是 NMS

NMS，全称 Non-Maximum Suppression，中文叫非极大值抑制。

它的作用是：

```text
去掉同一目标的重复预测框
保留得分最高的检测框
```

原因：

目标检测模型通常会对同一个物体预测多个框，如果不做 NMS，最终结果会有大量重复框。

---

## 十二、YOLOv2 的 NMS 流程

YOLOv2 推理后会得到大量候选框：

```text
13 x 13 x 5 = 845 个候选框
```

每个候选框有：

```text
box coordinates
objectness
class probabilities
```

常见类别得分：

```text
score_class = objectness * class_probability
```

NMS 通常按类别执行：

```text
对每个类别：
  1. 取出该类别得分超过阈值的框
  2. 按 score 从高到低排序
  3. 保留最高分框
  4. 删除与它 IoU 高于阈值的其他框
  5. 重复直到没有候选框
```

简化代码：

```python
def nms(boxes, scores, iou_threshold=0.5):
    indices = scores.argsort(descending=True)
    keep = []

    while indices.numel() > 0:
        current = indices[0]
        keep.append(current)

        if indices.numel() == 1:
            break

        rest = indices[1:]
        ious = box_iou(boxes[current][None], boxes[rest]).squeeze(0)
        indices = rest[ious < iou_threshold]

    return keep
```

---

## 十三、YOLOv1 与 YOLOv2 的 NMS 差异

YOLOv2 的 NMS 本质仍是 IoU-NMS，但候选框来源变了。

| 对比项 | YOLOv1 | YOLOv2 |
|--------|--------|--------|
| 候选框数量 | `7 x 7 x 2 = 98` | `13 x 13 x 5 = 845` |
| 框生成方式 | 直接预测框 | anchor + offset |
| 排序分数 | confidence × class probability | objectness × class probability |
| 小目标召回 | 较弱 | 更好 |
| NMS 类型 | IoU-NMS | IoU-NMS |

需要注意：

> YOLOv2 并不是提出了一种全新的 NMS，而是因为 anchor 和更多候选框让 NMS 输入更丰富，最终检测效果更好。

---

## 十四、YOLOv2 的训练流程

```text
输入图像
  -> resize / 多尺度训练
  -> Backbone 提取特征
  -> 检测头输出 13 x 13 x anchors
  -> 根据 GT 分配正负样本
  -> 计算坐标、objectness、分类损失
  -> 反向传播
```

关键点：

- 正样本由最佳 anchor 决定。
- 负样本只训练 objectness。
- ignore 样本避免误惩罚合理预测。
- 多尺度训练让模型适应不同输入尺寸。

---

## 十五、YOLOv2 的推理流程

```text
输入图像
  -> resize 到当前输入尺寸
  -> 模型输出 tx, ty, tw, th, objectness, class_probs
  -> 解码得到真实框坐标
  -> 计算 class score
  -> 过滤低分框
  -> 按类别执行 NMS
  -> 输出最终检测结果
```

---

## 十六、YOLOv2 的主要改进总结

| 改进点 | 说明 |
|--------|------|
| Batch Normalization | 提升训练稳定性和精度 |
| High Resolution Classifier | 使用更高分辨率微调分类网络 |
| Anchor Boxes | 提升召回率 |
| Dimension Clusters | 使用 K-Means 得到更合理 anchors |
| Direct Location Prediction | 限制中心点在当前 cell 内，训练更稳定 |
| Fine-Grained Features | 融合更高分辨率特征改善小目标 |
| Multi-Scale Training | 支持不同输入尺寸 |
| YOLO9000 | 结合检测和分类数据扩展类别数 |

---

## 十七、YOLOv2 的优点

| 优点 | 说明 |
|------|------|
| 比 YOLOv1 召回率更高 | anchor 机制带来更多候选框 |
| 定位更稳定 | 直接位置预测降低训练难度 |
| 小目标更好 | 特征融合和更多候选框有帮助 |
| 多尺度适应性更强 | 支持多尺度训练 |
| 速度仍然较快 | 保持单阶段检测优势 |

---

## 十八、YOLOv2 的局限

| 局限 | 说明 |
|------|------|
| 仍依赖 NMS | 推理后处理不可避免 |
| anchor 需要适配数据集 | 新数据集最好重新聚类 anchor |
| 密集小目标仍有挑战 | 比 YOLOv1 好，但仍不是强项 |
| 框回归和损失仍较早期 | 后续版本继续改进 IoU Loss 等机制 |

---

## 十九、YOLOv2 与后续版本的关系

YOLOv2 的 anchor 机制影响了后续很多检测器。

| 版本 | 继承或改进点 |
|------|--------------|
| YOLOv3 | 继续使用 anchor，并引入多尺度预测 |
| YOLOv4 | 引入更多训练技巧和 IoU 系列损失 |
| YOLOv5 | 工程化 anchor 检查和 AutoAnchor |
| YOLOX / YOLOv8 | 转向 anchor-free，并改进标签分配 |

---

## 二十、总结

YOLOv2 的核心可以概括为：

1. 在 YOLOv1 的基础上引入 Anchor Boxes。
2. 使用 K-Means 聚类获得更合理的 anchor 尺寸。
3. 通过 direct location prediction 提高训练稳定性。
4. 通过正负样本分配决定每个 anchor 的训练目标。
5. 推理阶段解码 anchor 偏移后，再按类别执行 NMS。
6. 多尺度训练和特征融合提升了模型泛化能力。

如果说 YOLOv1 开创了单阶段实时检测，那么 YOLOv2 则把 YOLO 推向了更实用、更稳定的 anchor-based 检测框架。

