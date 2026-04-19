
# YOLOv3 中的 NMS 详解（基于论文与 Darknet 实现）

## 一、什么是 NMS？

### 定义：

**NMS（Non-Maximum Suppression，非极大值抑制）是一种后处理技术**，用于去除目标检测中重复或冗余的边界框（bounding box），保留置信度最高且不重叠的预测结果。

### 目标：

- 消除对同一物体的多次预测；
- 提高检测结果的准确性；
- 减少误检；

---

## 二、YOLOv3 的输出结构回顾

YOLOv3 输出三个不同尺度的检测结果：

|特征图大小|anchor 数量|输出张量形状|
|------------|-------------|----------------|
|13×13|3|[1, 13, 13, 255]|
|26×26|3|[1, 26, 26, 255]|
|52×52|3|[1, 52, 52, 255]|

每个 bounding box 包含：
```
(tx, ty, tw, th, confidence, class_0, ..., class_80)
```

> 即：`85 = 4 (坐标) + 1 (confidence) + 80 (类别概率)`

---

## 三、YOLOv3 中 NMS 的执行流程（基于 Darknet 实现）

YOLOv3 的 NMS 是在模型推理完成后进行的后处理步骤，**不属于网络结构本身**，但对最终检测效果影响很大。

### 核心流程如下（来自 Darknet 源码逻辑）：

#### Step 1: 解码所有预测框（bounding boxes）

将 `(tx, ty, tw, th)` 转换为图像空间中的绝对坐标 `(x1, y1, x2, y2)`。

```python
boxes = decode_boxes(predictions, anchors, image_size=416)
```

#### Step 2: 计算每个框的综合得分（score）

YOLOv3 使用的是：

$$
\text{score} = \text{confidence} \times \max(\text{class\_probs})
$$

即：
- 表示“这个框有多大概率是一个物体” × “它属于某个类别的概率”

```python
scores = confidence * class_probs.max(dim=-1)
```

#### Step 3: 对每个类别分别执行 NMS

YOLOv3 **默认按类别执行 NMS**，即：

- 同一类别的框之间比较 IoU；
- 不同类别的框不会相互干扰；

```python
for class_id in range(num_classes):
    # 获取该类别的所有预测框
    class_boxes = boxes[class_ids == class_id]
    class_scores = scores[class_ids == class_id]

    # 执行 NMS
    keep_indices = nms(class_boxes, class_scores, iou_threshold=0.45)

    final_boxes.append(class_boxes[keep_indices])
```

---

## 四、YOLOv3 中的 NMS 参数说明（来自 Darknet 配置）

在 YOLOv3 的 `.cfg` 文件中，有以下 NMS 相关参数设置：

```ini
[region]
...
nms=1
beta_nms=0.6
nms_kind=greedynms
iou_thresh=0.45
```

|参数名|含义|默认值|
|--------|------|--------|
|`nms`|是否启用 NMS|1（启用）|
|`iou_thresh`|NMS 的 IoU 阈值|0.45|
|`nms_kind`|NMS 算法类型|greedynms（传统 NMS）|
|`beta_nms`|Soft-NMS 使用的平滑系数|0.6（仅在 soft-nms 下生效）|

> 注意：YOLOv3 原论文中并未使用 Soft-NMS，但在后续开源实现中支持。

---

## 五、YOLOv3 支持的 NMS 类型（来自 Darknet）

YOLOv3 的开源实现（如 AlexeyAB/darknet）支持多种 NMS 方法：

|NMS 类型|说明|是否原生支持|
|----------|------|--------------|
|`greedynms`|传统 NMS：保留置信度最高的框，删除与其重叠的其他框|是|
|`linear_soft_nms`|Soft-NMS：根据 IoU 降低 confidence，而不是直接删除|是（需手动配置）|
|`gaussian_soft_nms`|高斯权重 Soft-NMS|是（需手动配置）|
|`diounms`|DIoU-based NMS|否（Darknet 不支持）|

---

## 六、YOLOv3 中 NMS 的伪代码（简化版）

```python
def non_max_suppression(boxes, scores, threshold=0.45):
    """
    输入：
        - boxes: [N, 4]     # (x1, y1, x2, y2)
        - scores: [N]       # confidence × max_class_prob
        - threshold: float   # IoU 阈值
    输出：
        - keep_indices: list of int # 保留的框索引
    """
    keep = []
    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        best_idx = indices[0]
        keep.append(best_idx)
        best_box = boxes[best_idx]

        # 计算当前最佳框与其他框的 IoU
        ious = compute_iou(best_box, boxes[indices[1:]])

        # 保留 IoU < threshold 的框继续处理
        indices = indices[1:][ious < threshold]

    return keep
```

---

## 七、YOLOv3 NMS 的特点总结

|特点|说明|
|------|------|
|按类别执行 NMS|不同类别的框不会相互干扰|
|使用 confidence × class_prob 排序|更合理的排序方式|
|支持 Soft-NMS|可通过配置启用（非论文原生）|
|多尺度预测适配|对小目标更鲁棒|
|Anchor Boxes 支持|anchor 是通过 K-Means 聚类获得的预设框|

---

## 八、YOLOv3 中的 NMS 示例（PyTorch 简化版）

```python
import torch
from torchvision.ops import nms

# 假设我们已经解码出 boxes 和 scores
boxes = ...  # shape: [num_boxes, 4]
scores = ...  # shape: [num_boxes]

# 执行 NMS
keep_indices = nms(boxes, scores, iou_threshold=0.45)

# 获取最终保留的框
final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
```

---

## 九、YOLOv3 NMS 的实际表现（来源：官方测试数据）

|模型|mAP@COCO|FPS|NMS 类型|说明|
|------|-----------|-----|-----------|------|
|YOLOv3|~33.0|~45|greedynms|默认 NMS 方式|
|YOLOv3 + Soft-NMS|~33.1|~45|linear_soft_nms|小幅提升召回率|
|YOLOv3-tiny|~25.4|~150|greedynms|Tiny 版本也支持 NMS|

---

## 十、YOLOv3 中 NMS 的改进意义

虽然 YOLOv3 并未在论文中明确提出新的 NMS 方法，但在开源实现中：

- **引入了 Soft-NMS 支持**，提升了密集场景下的召回能力；
- **保持按类别执行 NMS 的传统做法**，适用于大多数现实场景；
- **anchor boxes 的引入间接提高了 NMS 的稳定性**；
- **多尺度预测使不同层级的框更合理，减少了误删情况**；

这些变化使得 YOLOv3 在实战中具有良好的检测性能。

---

## 十一、YOLOv3 NMS 的局限性

|局限性|说明|
|--------|------|
|不支持 DIoU-NMS|需要自定义修改|
|不支持 Efficient NMS|如 ONNXRuntime 的优化版本|
|不支持动态阈值调整|需要额外脚本支持|

---

## 十二、结语

|模块|内容|
|------|------|
|NMS 作用|去除重复预测框，提高检测结果质量|
|YOLOv3 NMS 实现|按类别执行，使用 confidence × class_prob 排序|
|支持 Soft-NMS|Darknet 实现中可选|
|多尺度预测适配|对不同大小目标更鲁棒|
|Anchor Boxes 支持|anchor 提升召回率和定位精度|



---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于计算机视觉、目标检测、深度学习、YOLO系列等内容！**

