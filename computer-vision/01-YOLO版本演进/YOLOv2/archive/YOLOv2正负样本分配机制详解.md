
# YOLOv2 正负样本分配机制详解

在目标检测任务中，**正负样本的定义**决定了哪些预测框用于训练，以及如何计算损失函数。YOLOv2 在 YOLOv1 的基础上，引入了 **Anchor Boxes（锚框）** 机制，正负样本的判断方式也发生了重要变化。

---

## 一、YOLOv2 中的 Anchor Box 引入

YOLOv2 将图像划分为 $S \times S$ 网格，每个网格预测 $B$ 个 Anchor Box（默认 5 个），每个 Anchor 预测：
- 位置偏移 $(t_x, t_y, t_w, t_h)$
- 置信度（objectness score）
- 多个类别的概率（softmax 或 sigmoid）

---

## 二、正负样本的定义

### 1. 正样本（Positive Samples）

满足以下条件的预测框被视为正样本：

- **某个 Ground Truth Box** 被分配给 **与其 IOU 最大的 Anchor Box**。
- 分配发生在该目标中心点所在的网格 cell 内。
- 每个 Ground Truth 只分配给 **一个 Anchor（最佳匹配）**。

**即：一张图中有多个目标，每个目标仅分配给一个最合适的 Anchor。**

---

### 2. 负样本（Negative Samples）

- 没有被任何 GT 分配的 Anchor 预测框。
- 与所有 Ground Truth 的最大 IOU **低于阈值（通常为 0.5）**。

这些 Anchor 被作为负样本，仅用于训练置信度（objectness）为 0。

---

## 三、样本分配机制图示

```text
Grid Cell (7x7)
  └─ 每个 Cell 预测 5 个 Anchor Box
        ├─ 与某个 GT IOU 最大 → 正样本
        ├─ IOU 太低 → 负样本
        └─ 其他 Anchor 忽略


## 四、与 YOLOv1 的区别

| 项目                 | YOLOv1                  | YOLOv2                          |
|----------------------|--------------------------|----------------------------------|
| Anchor               |  无                    |  有                            |
| 每个 GT 分配框数量   | 多个（B 个）            | 仅一个（最佳 Anchor）          |
| 负样本定义           | 其余所有预测框          | 未分配且 IOU < 阈值             |
| 正样本位置           | GT 中心落入的 Cell      | 同上                            |


## 五、 总结

YOLOv2 中的正负样本策略，主要围绕 Anchor Box 与 Ground Truth 的匹配关系：
	•	正样本：与 GT 匹配 IOU 最大的 Anchor
	•	负样本：其余 Anchor 且 IOU 低
	•	忽略：其他 IOU 不高不低者（可选处理）

# Bounding Box 与 Anchor 的关系详解

在目标检测中，`Anchor Box` 是模型预设的一组**参考框模板**，而 `Bounding Box` 是模型预测的**最终目标框**，两者之间的关系如下：

| 名称          | 说明                                                         |
|---------------|--------------------------------------------------------------|
| Anchor Box    | 预定义的固定尺寸框，用于覆盖不同尺寸、宽高比的目标           |
| Bounding Box  | 模型输出的框，用于拟合真实物体（Ground Truth Box）的位置     |

## 🔁 关系说明

1. 模型以 Anchor 为**起点**，通过预测 **偏移量（offset）** 对其进行微调；
2. 每个 Anchor 会输出一个预测框（Bounding Box）；
3. 训练过程中，选出与 Ground Truth 重合度最高（IoU 最大）的 Anchor，作为正样本；
4. 最终的 Bounding Box 是：
   $$ B_{\text{pred}} = \text{Anchor} + \text{偏移量} $$

##  举例

- Anchor: `[w=100, h=200]`（模型预定义的框）
- 偏移预测: `[dx, dy, dw, dh]`
- 最终预测框 (Bounding Box): 根据 anchor + 偏移解码得到

> Anchor 是起点，Bounding Box 是终点。













