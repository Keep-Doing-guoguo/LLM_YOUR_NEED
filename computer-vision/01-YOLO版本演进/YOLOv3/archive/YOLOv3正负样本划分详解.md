
# YOLOv3 正负样本划分详解（基于论文与 Darknet 实现）

## 一、前言

在目标检测任务中，**正负样本的划分** 是训练过程中的关键环节。它决定了哪些预测框参与位置回归、分类损失和置信度损失。

YOLOv3 在 YOLOv2 的基础上引入了 **多尺度预测** 和 **更精细的 Anchor Boxes 匹配策略**，使得正样本的选择更加合理，提高了模型的召回率和定位精度。

本文将基于以下来源进行解析：

- [YOLOv3: An Incremental Improvement (论文原文)](https://arxiv.org/abs/1804.02767)
- [AlexeyAB/darknet 开源实现](https://github.com/AlexeyAB/darknet)

---

## 二、YOLOv3 的输出结构回顾

YOLOv3 输出三个不同层级的特征图，分别用于检测不同尺度的目标：

|特征图大小|每个 cell 预测的 bounding box 数量|对应 anchor boxes|
|------------|----------------------------------|------------------|
|13×13|3|大目标|
|26×26|3|中目标|
|52×52|3|小目标|

每个 bounding box 的输出维度为：
```
(tx, ty, tw, th, confidence, class_0, ..., class_C)
```

---

## 三、YOLOv3 的正负样本划分机制详解

### 核心思想：

YOLOv3 中的正样本由以下两个条件共同决定：

1. **该 bounding box 的 anchor 与 ground truth 的 IoU 最大**
2. **该 bounding box 所属的 grid cell 包含物体中心点**

> 注意：YOLOv3 不像 Faster R-CNN 使用 RoI Pooling 或 Proposal 机制，而是直接对每个 grid cell 的多个 anchor 进行匹配。

---

### 正样本划分流程如下（来自 Darknet 实现）：

#### Step 1: 获取所有 ground truth 框
```python
ground_truth_boxes = [...]  # list of (x_center, y_center, w, h) 归一化坐标
```

#### Step 2: 遍历每个 ground truth 框

对于每个真实框：

1. 计算其相对于图像的归一化中心坐标；
2. 确定其落在哪个 grid cell；
3. 使用 K-Means 聚类得到的 9 个 anchors，计算每个 anchor 与该真实框的 IoU；
4. 找出 IoU 最大的那个 anchor；
5. 确定这个 anchor 属于哪个层级（13×13 / 26×26 / 52×52）；
6. 将该 anchor 所在的 bounding box 设为正样本；
7. 其余 anchor 若与该真实框的 IoU > iou_thresh（如 0.5），也设为正样本（可选）；

---

### 正样本选择规则总结：

|条件|是否为正样本|
|------|--------------|
|anchor 与 GT 的 IoU 最大|强制标记为正样本|
|anchor 与 GT 的 IoU > iou_threshold（如 0.5）|可选标记为正样本|
|无任何 GT 与其重叠|负样本|
|仅部分重叠但不是最大匹配|忽略（不参与位置损失）|

> 注意：只有负责预测真实框的 bounding box 才会参与位置回归（tx, ty, tw, th）和类别概率（class probs）的损失计算。

---

## 四、YOLOv3 中各类 loss 的作用对象

|Loss 类型|作用对象|
|-----------|----------|
|定位损失（Localization Loss）|正样本（负责预测的真实框）|
|置信度损失（Confidence Loss）|正样本 + 负样本|
|分类损失（Class Probability Loss）|正样本|

---

## 五、具体示例说明

假设我们有以下数据：

- 图像尺寸：`416 × 416`
- Ground Truth Box：`[0.5, 0.5, 0.2, 0.2]`（归一化坐标）
- Anchor Boxes（对应 52×52 特征图）：
  ```python
  anchors = [[10, 13], [16, 30], [33, 23]]  # 小目标 anchors
  ```

### Step-by-step 划分逻辑：

1. **确定 GT 中心落在哪个 grid cell？**

   - `cell_x = floor(0.5 * 52) = 26`
   - `cell_y = floor(0.5 * 52) = 26`

2. **计算每个 anchor 与 GT 的 IoU**

   ```python
   ious = [iou(anchor, gt_box) for anchor in anchors]
   best_anchor_idx = np.argmax(ious)
   ```

3. **标记该 anchor 对应的 bounding box 为正样本**

   - 假设 `best_anchor_idx = 0`，则第 0 个 bounding box 被标记为正样本；
   - 该 bounding box 的位置信息 `(tx, ty, tw, th)` 将被监督学习；
   - confidence 设为 1；
   - class prob 设置为目标类别；

4. **其他 anchor 若与 GT 的 IoU > threshold（如 0.5），也可被标记为正样本（可选）**

---

## 六、YOLOv3 中的负样本定义

### 负样本的判断标准：

- 该 bounding box **不负责预测任何真实框**；
- 且与所有真实框的 IoU < threshold（通常为 0.5）；
- 只参与置信度损失（confidence loss），不参与位置和分类损失；

---

## 七、YOLOv3 中的 ignore 样本（非正非负）

### 定义：

- 与某个真实框的 IoU > threshold（如 0.5），但不是最大 IoU 的那个 bounding box；
- 被标记为“ignore”，即：
  - 不参与位置损失；
  - 不参与分类损失；
  - 参与置信度损失吗？取决于实现方式（Darknet 中默认不参与）；

---

## 八、YOLOv3 中的标签分配伪代码（简化版）

```python
for each ground_truth in image:
    # Step 1: 确定 GT 中心所在的 grid cell
    cell_x = int(gt.x_center * feature_map_width)
    cell_y = int(gt.y_center * feature_map_height)

    # Step 2: 计算每个 anchor 与 GT 的 IoU
    ious_with_anchors = [bbox_iou(anchor, gt) for anchor in anchors]

    # Step 3: 找到 IoU 最大的 anchor
    best_anchor_index = np.argmax(ious_with_anchors)

    # Step 4: 标记该 anchor 为正样本
    y_true[cell_y, cell_x, best_anchor_index, ...] = {
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "confidence": 1,
        "class_prob": one_hot_label
    }

    # Step 5: （可选）IoU > threshold 的其他 anchor 也被标记为正样本
    for a in range(num_anchors):
        if ious_with_anchors[a] > ignore_thresh and a != best_anchor_index:
            y_true[cell_y, cell_x, a, ...] = {
                "confidence": 1,  # 仅置信度，不更新位置和类别
                ...
            }
```

---

## 九、YOLOv3 中的阈值设置（来自 darknet 源码）

在官方实现中，有两个关键参数影响正负样本划分：

|参数名|含义|默认值|
|--------|------|--------|
|`ignore_thresh`|若 IoU > 该阈值，则忽略（不参与损失）|0.5|
|`truth_thresh`|若 IoU > 该阈值，则强制作为正样本|1.0（默认只使用最大 IoU 的 anchor）|

> 注意：YOLOv3 默认只使用 IoU 最大的 anchor 作为正样本，其他高 IoU 的 anchor 并不参与训练。

---

## 十、YOLOv3 中的正负样本分布特点

|维度|内容|
|------|------|
|正样本数量|每个真实框 → 至少一个正样本|
|负样本数量|无物体的 bounding box|
|ignore 样本|IoU > ignore_thresh 但非最大匹配|
|位置损失|仅正样本参与|
|置信度损失|正样本 + 负样本|
|分类损失|仅正样本参与|

---

## 十一、YOLOv3 中的损失函数简要回顾

YOLOv3 的损失函数分为三个部分：

### 1. 定位损失（Localization Loss）

$$
\mathcal{L}_{loc} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B}
\mathbb{1}_{ij}^{obj}
\left[
(x_i - \hat{x}_i)^2 +
(y_i - \hat{y}_i)^2 +
(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 +
(\sqrt{h_i} - \sqrt{\hat{h}_i})^2
\right]
$$

> 仅对正样本计算此部分损失。

---

### 2. 置信度损失（Confidence Loss）

$$
\mathcal{L}_{conf} =
\sum_{i=0}^{S^2} \sum_{j=0}^{B}
\left[
\mathbb{1}_{ij}^{obj} \cdot (\text{confidence}_i^j - \hat{\text{confidence}}_i^j)^2
+ \lambda_{noobj} \cdot (1 - \mathbb{1}_{ij}^{obj}) \cdot (\text{confidence}_i^j)^2
\right]
$$

> 所有 bounding box 都参与此项损失，但负样本权重较低。

---

### 3. 分类损失（Class Probability Loss）

$$
\mathcal{L}_{cls} =
\sum_{i=0}^{S^2} \sum_{j=0}^{B}
\mathbb{1}_{ij}^{obj} \cdot \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

> 仅正样本参与分类损失。

---

## 十二、YOLOv3 的样本划分总结

|样本类型|IoU 条件|是否参与位置损失|是否参与置信度损失|是否参与分类损失|
|----------|------------|------------------|---------------------|--------------------|
|正样本|IoU 最大 或 IoU > threshold（可选）|是|是|是|
|ignore 样本|IoU > threshold 但非最大|否|否（Darknet 默认不参与）|否|
|负样本|IoU < threshold|否|是（降低 confidence）|否|

---

## 十三、结语

YOLOv3 的正负样本划分机制相比 YOLOv2 更加精细，主要体现在：

- 引入多尺度预测，提升小目标召回；
- 使用 anchor boxes 提升边界框匹配合理性；
- 通过 IoU 匹配选择最合适的 bounding box；
- 保留传统做法（仅最大 IoU 为正样本），避免过拟合；
- 支持 ignore 阈值，防止低 IoU 框干扰训练；



---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
