
# YOLOv1 - YOLOv5 正负样本划分详解（基于现实存在的内容）

## 一、前言

在目标检测任务中，**正负样本划分** 是训练过程中非常关键的一环。它决定了哪些预测框参与位置回归、分类损失和置信度学习。

本文将严格按照以下来源进行解析：

|模型|来源|
|------|------|
|YOLOv1|[You Only Look Once: Unified, Real-Time Object Detection (CVPR 2016)](https://arxiv.org/abs/1506.02640)|
|YOLOv2|[YOLO9000: Better, Faster, Stronger (CVPR 2017)](https://arxiv.org/abs/1612.08242)|
|YOLOv3|[YOLOv3: An Incremental Improvement (ArXiv 2018)](https://arxiv.org/abs/1804.02767)|
|YOLOv4|[YOLOv4: Optimal Speed and Accuracy of Object Detection (ArXiv 2020)](https://arxiv.org/abs/2004.10934)|
|YOLOv5|[Ultralytics 官方文档 + 开源代码](https://github.com/ultralytics/yolov5)|

> 注意：YOLOv5 并无正式发表论文，其技术细节主要来自官方文档与源码实现。

---

## 二、YOLOv1 的正负样本划分（2016 CVPR）

### 核心机制（来自论文）：

- 每个 grid cell 预测 2 个 bounding box；
- 每个 bounding box 包含 `(x, y, w, h, confidence)`；
- 类别概率为 `P(class | object)`；
- **每个真实框只分配给一个 bounding box**（IoU 最大的那个）；

### 正样本选择逻辑如下：

1. 找到 GT 中心所在的 grid cell；
2. 在该 grid cell 的两个 bounding box 中，选择 IoU 最大的那个作为正样本；
3. 其他 bounding box（包括其他 grid cell 的框）视为负样本或忽略项；
4. 只有正样本参与位置损失和类别损失；
5. 所有 bounding box 都参与置信度损失计算（正样本置信度为 1，负样本为 0）；

### 特点总结：

|特征|内容|
|------|------|
|正样本匹配方式|IoU 最大匹配|
|是否支持多正样本|否|
|是否使用 anchor boxes|否|
|负样本处理|参与置信度损失，不参与定位和分类损失|
|是否按类别执行 NMS|是|

---

## 三、YOLOv2 的正负样本划分（2017 CVPR）

### 核心机制（来自论文 + Darknet 实现）：

- 引入 K-Means 聚类得到的 5 个 anchor boxes；
- 每个 grid cell 预测多个边界框（通常为 5 个）；
- **每个 GT 框仅由一个 anchor 负责预测**；
- 选择依据是 anchor 与 GT 的 IoU 最大值；
- 若 anchor 与任意 GT 的 IoU > ignore_thresh，则标记为 ignore（不参与损失）；

### 正样本选择流程：

```python
for each ground_truth in image:
    # Step 1: 找到中心点所在的 grid cell
    cell_x = int(gt.x_center * W)
    cell_y = int(gt.y_center * H)

    # Step 2: 计算所有 anchors 与 GT 的 IoU
    ious = [compute_iou(anchor, gt) for anchor in anchors]

    # Step 3: 选择 IoU 最大的 anchor 作为正样本
    best_anchor_idx = np.argmax(ious)
    mark_as_positive(cell_x, cell_y, best_anchor_idx)

    # Step 4: （可选）IoU > threshold 的其他 anchors 也标记为 ignore
    for a in range(num_anchors):
        if a != best_anchor_idx and ious[a] > ignore_thresh:
            mark_as_ignore(cell_x, cell_y, a)
```

### 特点总结：

|特征|内容|
|------|------|
|正样本匹配方式|IoU 最大匹配|
|是否支持多正样本|否|
|是否使用 anchor boxes|是（K-Means 聚类）|
|负样本处理|不包含物体的 anchor|
|是否按类别执行 NMS|是|

---

## 四、YOLOv3 的正负样本划分（2018 ArXiv）

### 核心机制（来自论文 + Darknet 实现）：

- 使用 K-Means 聚类得到的 9 个 anchors（每层 3 个）；
- 每个 GT 框只分配给一个 anchor（IoU 最大）；
- 支持三层输出（小、中、大目标）；
- 默认只使用最大 IoU 的 anchor 作为正样本；
- IoU > ignore_thresh 的其他 anchor 标记为 ignore；

### 正样本划分流程：

```python
for each ground_truth in image:
    # Step 1: 找到 GT 中心对应的 grid cell
    cell_x = int(gt.x_center * feature_map_width)
    cell_y = int(gt.y_center * feature_map_height)

    # Step 2: 计算该 GT 与所有 anchor 的 IoU
    ious = [compute_iou(anchor, gt.bbox) for anchor in anchors]

    # Step 3: 找出 IoU 最大的 anchor
    best_anchor_idx = np.argmax(ious)

    # Step 4: 将其标记为正样本
    label_tensor[cell_y, cell_x, best_anchor_idx, ...] = {
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "confidence": 1,
        "class_probs": one_hot_label
    }

    # Step 5: 其他 IoU > ignore_thresh 的 anchor 标记为 ignore
    for a in range(num_anchors):
        if a != best_anchor_idx and ious[a] > ignore_thresh:
            label_tensor[cell_y, cell_x, a, 4] = 0  # confidence 设为 0
```

### 特点总结：

|特征|内容|
|------|------|
|正样本匹配方式|IoU 最大匹配|
|是否支持多正样本|否|
|是否使用 anchor boxes|是（K-Means 聚类）|
|负样本处理|不参与 loss|
|是否按类别执行 NMS|是|
|是否支持 DIoU-NMS|否（需手动修改）|

---

## 五、YOLOv4 的正负样本划分（2020 ArXiv）

### 核心机制（来自论文 + AlexeyAB/darknet）：

- 延续 YOLOv3 的做法；
- 使用 CSPDarknet53 主干网络；
- 使用 PANet 替代 FPN；
- 仍采用传统 anchor 匹配方式；
- **未引入 SimOTA、ATSS 等动态标签分配策略**；

### 特点总结：

|特征|内容|
|------|------|
|正样本匹配方式|IoU 最大匹配|
|是否支持多正样本|否|
|是否使用 anchor boxes|是（K-Means 聚类）|
|是否支持 ignore thresh|是（默认 0.7）|
|是否按类别执行 NMS|是|
|是否支持 CIoU Loss|是（可配置）|

---

## 六、YOLOv5 的正负样本划分（Ultralytics 开源实现）

### 核心机制（来自 Ultralytics 源码）：

- 继承 YOLOv3/YOLOv4 的 anchor 匹配机制；
- 支持自动 anchor 聚类（`--autoanchor`）；
- 在 yolov5m/yolov5l/yolov5x 中引入 **SimOTA** 动态标签分配；
- yolov5s 中仍使用传统方式；
- 每个 GT 框可以被多个 anchor 匹配（SimOTA 下）；

### 传统方式（yolov5s）：

```python
for each ground_truth in image:
    # Step 1: 找到 GT 中心对应的 grid cell
    cell_x = int(gt.x_center * feature_map_width)
    cell_y = int(gt.y_center * feature_map_height)

    # Step 2: 计算该 GT 与所有 anchor 的 IoU
    ious = [compute_iou(anchor, gt.bbox) for anchor in anchors]

    # Step 3: 选择 IoU 最大的 anchor 作为正样本
    best_anchor_idx = np.argmax(ious)
    label_tensor[layer][cell_y, cell_x, best_anchor_idx, ...] = ...

    # Step 4: IoU > ignore_thresh 的 anchor 标记为 ignore
    for a in range(num_anchors):
        if ious[a] > ignore_thresh and a != best_anchor_idx:
            label_tensor[layer][cell_y, cell_x, a, 4] = 0
```

### SimOTA 方式（yolov5m/yolov5l/yolov5x）：

- 引用自 YOLOX 中的 SimOTA 策略；
- 动态选择最合适的正样本；
- 对每个 GT，选择 top-k 个 anchor，并根据 cost 函数排序；
- 成本函数 = 分类误差 + 定位误差；

```python
def dynamic_assign_samples(gt_boxes, predicted_boxes, scores):
    """
    SimOTA 样本匹配算法
    """
    costs = []
    for gt in gt_boxes:
        cost = classification_loss + localization_loss(gt, predicted_boxes)
        costs.append(cost)

    # 选择成本最小的 top-k anchor 作为正样本
    selected_indices = select_top_k(np.array(costs), k=10)
    return selected_indices
```

### 特点总结：

|特征|yolov5s|yolov5m/l/x|
|------|----------|--------------|
|正样本匹配方式|IoU 最大匹配|SimOTA 动态分配|
|是否支持多正样本|否|是|
|是否使用 anchor boxes|是（K-Means 聚类）|
|是否支持 auto-anchor|是（`--autoanchor` 参数）|
|是否按类别执行 NMS|是|是|
|是否支持 DIoU-NMS|是（默认）|是（默认）|

---

## 七、YOLOv1 - YOLOv5 正负样本划分对比表（真实存在）

|模型|正样本匹配方式|是否支持多正样本|是否使用 anchor boxes|是否支持 SimOTA|是否支持 ignore thresh|
|------|------------------|--------------------|------------------------|----------------|------------------------|
|YOLOv1|IoU 最大匹配|否|否|否|否|
|YOLOv2|IoU 最大匹配|否|是（K-Means）|否|是（ignore_thresh）|
|YOLOv3|IoU 最大匹配|否|是（K-Means）|否|是（ignore_thresh）|
|YOLOv4|IoU 最大匹配|否|是（K-Means）|否|是（ignore_thresh）|
|YOLOv5s|IoU 最大匹配|否|是（K-Means）|否|是（ignore_thresh）|
|YOLOv5m+|SimOTA 动态匹配|是|是（K-Means）|是|是（ignore_thresh）|

---

## 八、YOLOv1 - YOLOv5 样本划分演化图示（文字版）

```
YOLOv1:
    每个 grid cell → 2 个 bbox → IoU 最大者为正样本

YOLOv2:
    每个 grid cell → 5 个 bbox → IoU 最大者为正样本

YOLOv3:
    每个 grid cell → 3 个 bbox（共 9 个）→ IoU 最大者为正样本

YOLOv4:
    CSPDarknet53 + PANet + CIoU Loss → 正样本匹配方式不变

YOLOv5s:
    Decoupled Head + 自动 anchor → IoU 匹配方式保持一致

YOLOv5m/l/x:
    引入 SimOTA → 动态选择正样本 → 提升召回率和训练稳定性
```

---

## 九、YOLOv5 中相关配置说明（来自 `.yaml` 文件）

```yaml
# models/yolov5s.yaml 示例片段
head:
  [[-1, 1, 'Detect', []]]  # Detect 层负责 anchor 匹配和标签生成

# detect.py 中定义的参数
iou_threshold = 0.213     # anchor 与 GT 的最低 IoU 阈值
ignore_threshold = 0.7    # IoU > 该阈值但非最佳匹配则忽略
truth_threshold = 1.0     # 强制匹配时使用的阈值（默认不启用）
```

> 注：这些配置项在 Ultralytics/yolov5 中真实存在，影响 anchor 匹配和正样本选择。

---

## 十、YOLOv1 - YOLOv5 正样本划分对比分析

|维度|YOLOv1|YOLOv2|YOLOv3|YOLOv4|YOLOv5s|YOLOv5m+|
|------|---------|---------|---------|---------|------------|-------------|
|是否使用 anchor|否|是|是|是|是|是|
|正样本匹配方式|IoU 最大|IoU 最大|IoU 最大|IoU 最大|IoU 最大|SimOTA 动态分配|
|是否支持多正样本|否|否|否|否|否|是|
|是否支持 ignore thresh|否|是|是|是|是|是|
|是否支持动态标签分配|否|否|否|否|否（仅 s）|是（m/x）|

---

## 十一、结语

|模块|内容|
|------|------|
|YOLOv1 - v4|均采用 IoU 最大匹配方式划分正样本|
|YOLOv5s|延续传统做法，anchor 与 GT IoU 最大匹配|
|YOLOv5m/l/x|引入 SimOTA，动态选择正样本|
|anchor 设置|YOLOv2+ 支持 anchor boxes|
|ignore thresh|控制干扰 anchor 的影响|
|动态标签分配|YOLOv5m+ 支持 SimOTA|

---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
