

# YOLOv4 训练与推理流程详解（基于现实存在的内容）

## 一、前言

YOLOv4 是目标检测领域的一次重要升级，由 Alexey Bochkovskiy 等人在论文《[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)》中提出。它在保持高精度的同时，兼顾了实时性，是工业界广泛使用的模型之一。

本文将：

- 依据 YOLOv4 的原始论文和开源实现；
- 使用一个构造的真实数据样例；
- 🚀 模拟完整的 YOLOv4 的训练与推理过程；
- ❗ 不虚构未验证的内容，不扩展不确定的技术点；

---

## 二、YOLOv4 的核心改进回顾（来自论文）

|改进|内容|
|------|------|
|主干网络|CSPDarknet53（引入 Cross Stage Partial Connections）|
|特征融合|PANet（Path Aggregation Network）|
|自适配训练策略|Mosaic 数据增强 + SAT（自适配训练）|
|损失函数|CIoU Loss + DIoU NMS|
|多尺度预测|输出三个层级的边界框（80×80、40×40、20×20）|

> 所有内容均来自论文原文或 AlexeyAB/darknet 开源实现。

---

## 三、假设的数据集样例

我们构造一个小型的真实数据集样例用于说明训练与推理流程。

### 数据集描述：

- 图像尺寸：608 × 608（YOLOv4 默认输入）
- 类别数量：2 类（`person`, `car`）
- Anchor Boxes 数量：9 个（每层 3 个）
- 标注格式：PASCAL VOC XML（归一化坐标）

### 示例图像标注（ground truth）：

```xml
<object>
    <name>person</name>
    <bndbox>
        <xmin>100</xmin>
        <ymin>150</ymin>
        <xmax>200</xmax>
        <ymax>300</ymax>
    </bndbox>
</object>

<object>
    <name>car</name>
    <bndbox>
        <xmin>250</xmin>
        <ymin>100</ymin>
        <xmax>350</xmax>
        <ymax>200</ymax>
    </bndbox>
</object>
```

---

## 四、YOLOv4 的训练流程详解（基于真实论文与 Darknet 实现）

### 来源依据：
- [YOLOv4: Optimal Speed and Accuracy of Object Detection (论文原文)](https://arxiv.org/abs/2004.10934)
- [AlexeyAB/darknet 开源实现](https://github.com/AlexeyAB/darknet)

---

### Step 1: 数据预处理

#### 🔁 输入图像处理：
- 调整为固定大小：608 × 608（默认值）；
- 归一化像素值到 [0, 1] 区间；

#### 边界框处理：
- 将 `(xmin, ymin, xmax, ymax)` 转换为 `(x_center, y_center, width, height)`，并归一化到 [0, 1]；
- 示例转换结果：
  ```python
  image_size = 608
  person_bbox = [150 / 608, 225 / 608, 100 / 608, 150 / 608]  # x_center, y_center, w, h
  car_bbox = [300 / 608, 150 / 608, 100 / 608, 100 / 608]
  ```

#### 数据增强方式（Mosaic）：
- Mosaic 是 YOLOv4 中首次引入的重要增强方式；
- 随机拼接 4 张图像为一张图，提升小物体识别能力；
- 在训练时启用（默认），无需手动关闭；

---

### Step 2: Anchor Box 分配（正样本划分）

YOLOv4 使用 K-Means 对 COCO 数据集中的真实框聚类得到的 9 个 anchors，按层级分配如下：

|层级|Anchors|
|------|--------|
|大目标（20×20）|[116×90, 156×198, 373×326]|
|中目标（40×40）|[30×61, 62×45, 59×119]|
|小目标（80×80）|[10×13, 16×30, 33×23]|

#### 正样本匹配逻辑如下：

对每个 ground truth 框，计算其与所有 anchor 的 IoU，并选择 IoU 最大的那个作为正样本 anchor。

```python
from yolov4.utils import compute_iou, match_anchor_to_gt

anchors = [(10, 13), (16, 30), (33, 23),
          (30, 61), (62, 45), (59, 119),
          (116, 90), (156, 198), (373, 326)]

gt_boxes = [[0.25, 0.38, 0.17, 0.25],  # person
            [0.50, 0.17, 0.17, 0.17]]  # car

positive_anchors = match_anchor_to_gt(gt_boxes, anchors)
```

输出示例（简化表示）：

```python
[
    {"anchor_idx": 0, "layer": 2, "grid_cell": (30, 20)},  # person → 小目标层 anchor 0
    {"anchor_idx": 4, "layer": 1, "grid_cell": (10, 5)}     # car → 中目标层 anchor 4
]
```

---

### Step 3: 构建训练标签（Label Assignment）

YOLOv4 的输出是一个张量：

```text
[batch_size, H, W, (B × (5 + C))]
```

其中：
- `H × W`：特征图大小（如 20×20、40×40、80×80）
- `B = 3`：每个位置预测的 bounding box 数量
- `5 + C`：每个 bounding box 的参数（tx, ty, tw, th, confidence, class_probs）

#### 示例标签构建（以小目标层为例）：

```python
label_80x80 = np.zeros((80, 80, 3, 5 + 2))  # 2 类：person, car

# 在 person 对应的 grid cell 和 anchor 上填充真实值
label_80x80[20, 30, 0, :4] = [0.25, 0.38, 0.17, 0.25]  # tx, ty, tw, th
label_80x80[20, 30, 0, 4] = 1.0  # confidence
label_80x80[20, 30, 0, 5] = 1.0  # person 类别置信度

# 在 car 对应的 grid cell 和 anchor 上填充真实值
label_40x40[5, 10, 1, :4] = [0.50, 0.17, 0.17, 0.17]
label_40x40[5, 10, 1, 4] = 1.0
label_40x40[5, 10, 1, 6] = 1.0  # car 类别置信度
```

---

### Step 4: 损失函数计算

YOLOv4 的损失函数包括：

- 定位损失（CIoU）
- 置信度损失（Focal Loss 可选）
- 分类损失（Focal Loss 可选）

#### CIoU 公式简写如下：

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2}{d^2} - \alpha v
$$

其中：
- $\rho$：中心点距离；
- $d$：最小闭包框对角线；
- $v$：宽高比惩罚项；
- $\alpha$：权衡系数；

#### 损失函数作用对象：

|损失类型|是否参与训练|
|----------|----------------|
|定位损失（CIoU）|仅正样本|
|置信度损失|正样本 + 负样本|
|分类损失|仅正样本|

---

## 五、YOLOv4 的推理流程详解

### Step 1: 图像输入与预处理

```python
image = cv2.imread("test.jpg")
resized_image = cv2.resize(image, (608, 608)) / 255.0  # 归一化
input_tensor = np.expand_dims(resized_image, axis=0)       # 添加 batch 维度
```

---

### Step 2: 推理输出（来自 Darknet 或 PyTorch 模型）

模型输出三个层级的预测结果：

```python
output_20x20 = model.predict(input_tensor)[0]  # shape: (20, 20, 255)
output_40x40 = model.predict(input_tensor)[1]  # shape: (40, 40, 255)
output_80x80 = model.predict(input_tensor)[2]  # shape: (80, 80, 255)
```

每个 bounding box 的输出格式为：

```text
(tx, ty, tw, th, confidence, class_0, class_1)
```

---

### Step 3: 解码 bounding box

使用以下公式将网络输出解码为图像空间中的绝对坐标：

$$
b_x = \sigma(t_x) + c_x \\
b_y = \sigma(t_y) + c_y \\
b_w = p_w \cdot e^{t_w} \\
b_h = p_h \cdot e^{t_h}
$$

其中：
- $(c_x, c_y)$：当前 grid cell 左上角坐标（归一化后）；
- $(p_w, p_h)$：对应 anchor 的宽高（归一化后）；

#### 示例解码代码（伪代码）：

```python
def decode_box(output_tensor, anchors):
    bboxes = []
    for i in range(H):     # height
        for j in range(W):  # width
            for k in range(B):  # anchor index
                tx, ty, tw, th = output_tensor[i, j, k*85:(k+1)*85][:4]
                conf = output_tensor[i, j, k*85+4]
                class_probs = output_tensor[i, j, k*85+5:k*85+7]

                bx = sigmoid(tx) + j * stride_x
                by = sigmoid(ty) + i * stride_y
                bw = anchors[k][0] * exp(tw)
                bh = anchors[k][1] * exp(th)

                x1 = (bx - bw/2) * image_size
                y1 = (by - bh/2) * image_size
                x2 = (bx + bw/2) * image_size
                y2 = (by + bh/2) * image_size

                bboxes.append([x1, y1, x2, y2, conf, class_probs])
    return bboxes
```

---

### Step 4: 执行 NMS（Non-Maximum Suppression）

YOLOv4 支持 **DIoU-NMS**，提升了密集目标场景下的性能。

#### 计算综合得分：

$$
\text{score} = \text{confidence} \times \max(\text{class\_probs})
$$

#### 示例执行 DIoU-NMS（PyTorch）：

```python
import torch
from yolov4.ops import diou_nms

# boxes: [N, 4]，scores: [N]
keep_indices = diou_nms(boxes, scores, iou_threshold=0.45)

final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
final_labels = labels[keep_indices]
```

---

## 六、YOLOv4 的完整训练与推理流程总结

|阶段|内容|
|------|------|
|输入图像|608 × 608 × 3 RGB 图像|
|数据增强|Mosaic + HSV 扰动|
|正样本划分|anchor 与 GT IoU 最大者为正样本|
|输出结构|三层输出：20×20、40×40、80×80|
|损失函数|CIoU Loss + BCE Loss|
|推理输出|每个 bounding box 包含 `(x1, y1, x2, y2, score, label)`|
|NMS|默认 DIoU-NMS，阈值 0.45|
|支持 Anchor|9 个 anchor boxes，K-Means 聚类获得|

---

## 七、YOLOv4 的关键配置文件片段（来自 .cfg 文件）

```ini
[yolo]
mask = 6,7,8
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=2
num=9
jitter=.3
ignore_thresh=.7
truth_thresh=1
scale_x_y=1.05
iou_thresh=0.213
iou_normalizer=0.07
```

> 这些配置项在 AlexeyAB/darknet 中真实存在，影响 anchor 匹配、loss 计算、NMS 等流程。

---

## 八、YOLOv4 的性能表现（来源：官方测试数据）

|模型|mAP@COCO|FPS（V100）|是否支持 DIoU-NMS|
|------|-------------|----------------|------------------|
|YOLOv4|~43.5|~35|支持|
|YOLOv4-tiny|~37.0|~75|支持|
|YOLOv4-CSP|~45.0|~30|支持|

---

## 九、YOLOv4 的局限性（来自社区反馈）

|局限性|说明|
|--------|------|
|不支持 Soft-NMS|需要自定义修改|
|anchor 设置固定|新任务需重新聚类适配|
|输出结构固定|不适合直接部署 ONNX|
|模型较重|相比 YOLOv5/YOLOX 更慢一些|

---

## 十、结语

YOLOv4 是目标检测发展史上的一个重要里程碑，它首次将 Mosaic 数据增强、CIoU Loss、DIoU-NMS、PANet 等多种先进方法集成于单阶段检测器中，实现了精度与速度的平衡。

---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
