
# YOLOv2 中 NMS 的详解

## 一、什么是 NMS？

### 定义：

**NMS（非极大值抑制）是一种目标检测中的后处理技术**，用于去除重复预测的边界框，保留置信度最高且不重叠的边界框。

### 目标：

- 提高检测结果的准确性；
- 避免同一物体被多次检测；
- 减少误检和冗余框；

---

## 二、YOLOv1 中的 NMS 实现

### 来源依据：
来自 [You Only Look Once: Unified, Real-Time Object Detection (CVPR 2016)](https://arxiv.org/abs/1506.02640)

### 输出结构回顾：

YOLOv1 输出为一个 `7×7×30` 的张量：

- 每个 grid cell 输出 2 个 bounding box；
- 每个 bounding box 包含 `(x, y, w, h, confidence)`；
- 后接 20 个类别的概率（PASCAL VOC）；

### NMS 执行流程（根据论文描述 + Darknet 实现）：

1. **按类别执行 NMS**
   - 对每个类别分别执行 NMS；
   - 即：只在同一类别的预测框之间比较 IoU；

2. **筛选该类别的所有预测框**

```python
class_boxes = [box for box in all_boxes if box.class_id == class_id]
```

3. **按 confidence 排序**

```python
class_boxes.sort(key=lambda x: x.confidence_score, reverse=True)
```

4. **依次选择最大 confidence 的框，并删除与其高度重叠的其他框**

```python
keep_boxes = []
while len(class_boxes) > 0:
    highest_conf_box = class_boxes.pop(0)
    keep_boxes.append(highest_conf_box)
    class_boxes = [
        box for box in class_boxes
        if iou(box, highest_conf_box) < iou_threshold
    ]
```

> 常用阈值：`iou_threshold = 0.5`

---

### YOLOv1 NMS 的特点总结：

|特点|说明|
|------|------|
|按类别执行|不同类别的框不会相互干扰|
|使用 confidence 排序|只使用 bounding box 的 confidence 分数|
|每个网格预测两个框|框的数量有限，召回率较低|
|简单高效|在 CPU 上也能运行良好|
|效果一般|对密集目标或小目标效果较差|

---

## 三、YOLOv2 中的 NMS 实现

### 来源依据：
来自 [YOLO9000: Better, Faster, Stronger (CVPR 2017)](https://arxiv.org/abs/1612.08242)

### 输出结构回顾：

YOLOv2 输出为一个 `13×13×(B × 5 + C)` 张量：

- 每个 grid cell 预测 B=5 个 anchor boxes；
- 每个 bounding box 包含 `(tx, ty, tw, th, confidence)`；
- 类别概率为 C 个（COCO 为 80）；
- 每个 anchor box 还包含对应的宽高偏移（基于聚类 anchors）；

---

### YOLOv2 NMS 的执行流程（基于 Darknet 实现）：

#### Step 1: 计算每个 bounding box 的综合得分（score）

不同于 YOLOv1，YOLOv2 在排序时使用的是：

$$
\text{score} = \text{confidence} \times \max(\text{class probabilities})
$$

即：
- 表示“这个框有多大概率是一个物体” × “它属于某个类别的概率”

```python
for box in all_boxes:
    box.score = box.confidence * max(box.class_probs)
```

#### Step 2: 按 score 排序

```python
all_boxes.sort(key=lambda x: x.score, reverse=True)
```

#### Step 3: 执行 NMS

```python
keep_boxes = []

while len(all_boxes) > 0:
    highest_score_box = all_boxes.pop(0)
    keep_boxes.append(highest_score_box)

    all_boxes = [
        box for box in all_boxes
        if iou(box, highest_score_box) < iou_threshold
    ]
```

---

### YOLOv2 NMS 的特点总结：

|特点|说明|
|------|------|
|综合 score 排序|使用 confidence × class probability，提升排序合理性|
|更多 anchor boxes|每个 grid cell 预测 5 个框，提高召回率|
|支持 Soft-NMS|Darknet 后续版本支持软抑制（非官方）|
|多尺度训练适配|对不同大小目标的鲁棒性更强|
|Anchor Boxes 支持|anchor 是通过 K-Means 聚类获得的预设框|

---

## 四、YOLOv1 与 YOLOv2 NMS 的核心差异对比（基于论文与 Darknet 实现）

|对比维度|YOLOv1|YOLOv2|
|----------|---------|---------|
|是否使用 Anchor Boxes|不使用|使用 K-Means 聚类 anchor|
|排序依据|confidence|confidence × max(class_probs)|
|是否按类别执行|是|是（默认）|
|每个 grid cell 预测框数量|2|5|
|是否支持 Soft-NMS|不支持|支持（Darknet 后续实现）|
|是否支持 DIoU-NMS|不支持|支持（可通过修改实现）|
|小目标检测能力|较差|明显提升|
|多尺度训练支持|不支持|支持输入尺寸随机变化|
|NMS 应用位置|检测头输出后|检测头输出后（同 YOLOv1）|

---

## 五、YOLOv1 / YOLOv2 中 NMS 的 PyTorch 示例代码（简化版）

### YOLOv1 NMS 示例（伪代码）：

```python
def nms_yolov1(boxes, confidences, iou_threshold=0.5):
    # 按置信度排序
    indices = np.argsort(confidences)[::-1]
    keep = []

    while len(indices) > 0:
        best_idx = indices[0]
        keep.append(best_idx)

        # 计算 IoU
        ious = [iou(boxes[best_idx], boxes[i]) for i in indices[1:]]
        indices = indices[1:][np.array(ious) < iou_threshold]

    return keep
```

---

### YOLOv2 NMS 示例（伪代码）：

```python
def nms_yolov2(boxes, scores, iou_threshold=0.5):
    # 按综合分数排序（confidence × class prob）
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        best_idx = indices[0]
        keep.append(best_idx)

        # 计算 IoU
        ious = [iou(boxes[best_idx], boxes[i]) for i in indices[1:]]
        indices = indices[1:][np.array(ious) < iou_threshold]

    return keep
```

---

## 六、YOLOv1 / YOLOv2 中 NMS 的实际表现对比（来源：YOLO 官方文档 & Darknet 实现）

|模型|mAP@COCO|FPS|NMS 类型|NMS 输入方式|
|------|------------|-----|-----------|----------------|
|YOLOv1|~63.4|45 fps|IoU-NMS|confidence 排序|
|YOLOv2|~76.8|67 fps|IoU-NMS（可扩展为 Soft/DIoU）|confidence × class_prob 排序|

---

## 七、YOLOv2 中 NMS 的改进意义

虽然 YOLOv2 并未在论文中明确提出新的 NMS 方法，但在实践中：

- **引入了 Anchor Boxes**：使得预测框更合理，提高召回率；
- **综合 score 排序**：提升了排序的合理性；
- **支持更多框预测**：从每格 2 个框增加到 5 个框，提高了覆盖范围；
- **配合多尺度训练**：增强了模型对不同大小目标的适应性；

这些变化间接使 NMS 的效果显著优于 YOLOv1。

---

## 八、结语

|模块|内容|
|------|------|
|**YOLOv1 的 NMS**|按类别执行，仅使用 confidence 排序|
|**YOLOv2 的 NMS**|使用 confidence × class prob 排序，anchor 框提升召回|
|**区别总结**|YOLOv2 的 NMS 更加合理，结合 anchor boxes 提升整体性能|
|**现实意义**|NMS 是目标检测不可或缺的一环，YOLOv2 的改进是后续版本的基础|

---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于计算机视觉、目标检测、深度学习、YOLO系列等内容！**
