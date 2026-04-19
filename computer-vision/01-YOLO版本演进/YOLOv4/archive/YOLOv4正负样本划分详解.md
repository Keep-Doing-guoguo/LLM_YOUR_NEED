
# YOLOv4 正负样本划分详解（基于现实存在的内容）

## 一、前言

在目标检测中，**正负样本划分是训练过程中的关键环节**，它决定了哪些预测框参与损失计算，从而影响模型的学习效果。

YOLOv4 在 YOLOv3 的基础上进行了改进，包括：

- 使用 CSPDarknet53 主干网络；
- 引入 PANet 特征融合结构；
- 支持 Mosaic 数据增强；
- 使用 CIoU Loss 和 DIoU-NMS；

但在**正样本划分逻辑上**，YOLOv4 **保持了与 YOLOv3 类似的设计方式**，并未引入如 YOLOv5 中的“动态标签分配”等新机制。

本文将基于以下来源进行解析：

- [YOLOv4: Optimal Speed and Accuracy of Object Detection (论文原文)](https://arxiv.org/abs/2004.10934)
- [AlexeyAB/darknet 开源实现](https://github.com/AlexeyAB/darknet)

---

## 二、YOLOv4 的输出结构回顾

YOLOv4 输出三个层级的特征图：

|层级|特征图大小|anchor boxes|
|------|-------------|--------------|
|大目标|20×20|[116,90], [156,198], [373,326]|
|中目标|40×40|[30,61], [62,45], [59,119]|
|小目标|80×80|[10,13], [16,30], [33,23]|

每个 bounding box 包含：

```
(tx, ty, tw, th, confidence, class_0, ..., class_C)
```

> 其中 C 是类别数量（如 COCO 为 80）

---

## 三、YOLOv4 的正样本划分机制详解（来自 Darknet 实现）

### 核心思想：

YOLOv4 的正样本由以下两个条件共同决定：

1. **该 bounding box 所属的 grid cell 包含物体中心点**
2. **该 bounding box 的 anchor 与 ground truth 的 IoU 最大**

> 注：这是 YOLO 系列的传统做法，在 YOLOv4 中未发生本质变化。

---

### 示例说明（假设图像大小为 608×608）：

#### Step 1: 获取 ground truth 框（归一化坐标）

```python
gt_boxes = [
    {"class_id": 0, "bbox": [0.25, 0.38, 0.17, 0.25]},  # person
    {"class_id": 1, "bbox": [0.50, 0.17, 0.17, 0.17]}   # car
]
```

其中 bbox 格式为 `(x_center, y_center, width, height)`，均归一化到 [0, 1]

---

#### Step 2: 获取所有 anchor boxes（K-Means 聚类结果）

```python
anchors = [(10,13), (16,30), (33,23),
          (30,61), (62,45), (59,119),
          (116,90), (156,198), (373,326)]
```

---

#### Step 3: 判断哪个 anchor 应被标记为正样本

对每个 GT 框，执行如下操作：

1. 计算其落在哪个 grid cell；
2. 计算其与所有 anchor 的 IoU；
3. 找出 IoU 最大的那个 anchor；
4. 将该 anchor 对应的 bounding box 标记为正样本；
5. （可选）其他 IoU > threshold 的 anchor 可以被标记为 ignore（不参与 loss）；

```python
for gt in gt_boxes:
    x_center = int(gt["bbox"][0] * feature_map_width)  # 归一化 → grid_x
    y_center = int(gt["bbox"][1] * feature_map_height)

    ious = [compute_iou(anchor, gt["bbox"]) for anchor in anchors]
    best_anchor_idx = np.argmax(ious)

    # 找出该 anchor 所属层级（layer）
    layer_idx = get_layer_from_anchor(best_anchor_idx)  # 如 0 表示小目标层

    # 标记为正样本
    label_tensor[layer_idx][y_center, x_center, best_anchor_idx, ...] = {
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "confidence": 1,
        "class_prob": one_hot(class_id)
    }
```

---

## 四、YOLOv4 中各类样本定义总结

|样本类型|条件|是否参与定位损失|是否参与置信度损失|是否参与分类损失|
|----------|------|------------------|---------------------|--------------------|
|正样本|IoU 最大 或 IoU > threshold（可选）|是|是|是|
|负样本|无任何 GT 与其重叠|否|是|否|
|Ignore 样本|IoU > ignore_thresh 但非最大匹配|否|否（默认）|否|

---

## 五、YOLOv4 中的阈值设置（来自 darknet 源码）

YOLOv4 的 `.cfg` 文件中有几个关键参数用于控制 anchor 匹配和正样本选择：

```ini
[yolo]
iou_loss=ciou
iou_thresh=0.213
ignore_thresh=0.7
truth_thresh=1
```

|参数名|含义|默认值|
|--------|------|--------|
|`iou_thresh`|anchor 与 GT 的最小 IoU 阈值，用于筛选候选 anchor|0.213|
|`ignore_thresh`|若 IoU > 该阈值，则标记为 ignore（不参与损失）|0.7|
|`truth_thresh`|若 anchor 与 GT 的 IoU > 该阈值，则强制作为正样本|1.0（仅使用最大 IoU 的 anchor）|

> 注意：YOLOv4 默认只使用 IoU 最大的 anchor 作为正样本。

---

## 六、YOLOv4 的损失函数作用对象

YOLOv4 的损失函数分为三部分：

|损失类型|作用对象|
|----------|-----------|
|定位损失（CIoU）|正样本|
|置信度损失（BCE）|正样本 + 负样本|
|分类损失（BCE）|正样本|

---

## 七、YOLOv4 的正样本匹配流程总结（伪代码）

```python
for each ground_truth_box in image:
    # Step 1: 确定 GT 中心所在的 grid cell
    cell_x = int(gt.x_center * feature_map_width)
    cell_y = int(gt.y_center * feature_map_height)

    # Step 2: 计算每个 anchor 与 GT 的 IoU
    ious_with_anchors = [bbox_iou(anchor, gt) for anchor in anchors]

    # Step 3: 找到 IoU 最大的 anchor
    best_anchor_index = np.argmax(ious_with_anchors)

    # Step 4: 判断是否满足阈值要求
    if ious_with_anchors[best_anchor_index] >= truth_thresh:
        mark as positive sample
        assign to corresponding yolo layer
        compute localization loss
        compute class loss

    else:
        for a in range(num_anchors):
            if ious_with_anchors[a] > ignore_thresh and a != best_anchor_index:
                mark as ignore
            else:
                mark as negative sample
```

---

## 八、YOLOv4 中的标签构建（简化版）

YOLOv4 的输出是一个三维张量：

```text
[batch_size, H, W, (B × (5 + C))]
```

其中：
- `H × W`：特征图大小（如 80×80）
- `B = 3`：每个位置预测的 bounding box 数量
- `5 + C`：每个 bounding box 的参数（tx, ty, tw, th, confidence, class_probs）

### 示例标签构建（以 person 为例）：

```python
label_80x80 = np.zeros((80, 80, 3, 5 + num_classes))

# person 出现在 (30, 20) 的 grid cell，anchor_idx = 0
label_80x80[20, 30, 0, :4] = [0.25, 0.38, 0.17, 0.25]  # tx, ty, tw, th
label_80x80[20, 30, 0, 4] = 1.0  # confidence
label_80x80[20, 30, 0, 5] = 1.0  # person 类别置信度
```

---

## 九、YOLOv4 中的 NMS 阶段样本处理（推理阶段）

虽然 NMS 不属于训练阶段，但它也涉及样本保留或删除，可以视为一种广义上的“正样本后处理”。

YOLOv4 支持 **DIoU-NMS**，提升密集场景下的召回能力。

```python
keep_indices = diou_nms(boxes, scores, iou_threshold=0.45)
final_boxes = boxes[keep_indices]
```

---

## 十、YOLOv4 中的正负样本划分特点总结

|维度|内容|
|------|------|
|正样本选择|anchor 与 GT IoU 最大者为正样本|
|忽略样本|IoU > ignore_thresh 但非最大匹配|
|锚框匹配方式|K-Means 聚类 anchor，IoU 匹配|
|多尺度预测支持|三层输出，分别对应不同尺寸 anchor|
|Anchor Boxes|每个层级使用 3 个 anchor|
|损失函数|CIoU Loss + BCE Loss|
|改进空间|未引入 ATSS、SimOTA 等自适应策略|

---

## 十一、YOLOv4 的局限性（关于正样本划分）

|局限性|说明|
|--------|------|
|不支持动态标签分配|如 YOLOv5 中的 SimOTA 等|
|不支持 ATSS|自动匹配策略未集成|
|anchor 设置固定|新任务需重新聚类适配|
|缺乏多正样本机制|只使用最大 IoU 的 anchor|

---

## 十二、结语

YOLOv4 的正负样本划分机制沿用了 YOLOv3 的设计思路：

- 使用 anchor 与 GT 的 IoU 进行匹配；
- 每个 GT 只匹配一个 anchor；
- 保留传统做法，不引入复杂策略；
- 支持 ignore thresh 控制干扰样本；



 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
