

# YOLOv2 训练与推理流程详解（结合真实数据样例）

## 一、前言

YOLOv2 是目标检测领域的一次重要升级，由 Joseph Redmon 等人在论文《[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)》中提出。

其核心改进包括：

- 引入 Anchor Boxes
- 多尺度预测
- 更强的主干网络（Darknet-19）
- 联合训练 COCO + ImageNet（YOLO9000）

本文将通过一个**实际构造的数据样例**，带你一步步走过 YOLOv2 的训练和推理过程。

---

## 二、假设的数据集样例

我们构造一个小型的真实数据集样例用于说明训练与推理流程。

### 数据集描述：

- 图像尺寸：416 × 416
- 类别数量：2 类（`person`, `car`）
- Anchor Boxes 数量：5 个（K-Means 聚类得到）
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

## 三、YOLOv2 的训练流程详解

### 来源依据：
- [YOLO9000: Better, Faster, Stronger (CVPR 2017)](https://arxiv.org/abs/1612.08242)
- [AlexeyAB/darknet 开源实现](https://github.com/AlexeyAB/darknet)

---

### Step 1: 数据预处理

#### 🔁 输入图像处理：
- 调整为固定大小：416 × 416；
- 归一化像素值到 [0, 1] 区间；

#### 边界框处理：
- 将 `(xmin, ymin, xmax, ymax)` 转换为 `(x_center, y_center, width, height)`，并归一化到 [0, 1]；
- 示例转换结果：
  ```python
  image_size = 416
  person_bbox = [150 / 416, 225 / 416, 100 / 416, 150 / 416]  # x_center, y_center, w, h
  car_bbox = [300 / 416, 150 / 416, 100 / 416, 100 / 416]
  ```

---

### Step 2: Anchor Box 分配（正样本划分）

YOLOv2 使用 K-Means 对 COCO 数据集中的真实框聚类得到的 5 个 anchors：

```python
anchors = [(1.08, 1.19), (1.32, 3.19),
          (3.03, 4.34), (4.22, 2.81), (5.92, 5.53)]
```

#### 正样本匹配逻辑如下：

对每个 ground truth 框，计算其与所有 anchor 的 IoU，并选择 IoU 最大的那个作为正样本 anchor。

```python
from yolov2.utils import compute_iou, match_anchor_to_gt

gt_boxes = [[0.36, 0.54, 0.24, 0.36],  # person
            [0.72, 0.36, 0.24, 0.24]]  # car

positive_anchors = match_anchor_to_gt(gt_boxes, anchors)
```

输出示例（简化表示）：

```python
[
    {"anchor_idx": 0, "grid_cell": (18, 9)},   # person → anchor 0
    {"anchor_idx": 3, "grid_cell": (10, 5)}    # car → anchor 3
]
```

---

### Step 3: 构建训练标签（Label Assignment）

YOLOv2 的输出是一个张量：

```text
[batch_size, H, W, (B × (5 + C))]
```

其中：
- `H × W = 13 × 13`
- `B = 5`：每个位置预测的 bounding box 数量
- `5 + C`：每个 bounding box 的参数（tx, ty, tw, th, confidence, class_probs）

#### 示例标签构建：

```python
label_tensor = np.zeros((13, 13, 5, 5 + 2))  # 2 类：person, car

# 在 person 对应的 grid cell 和 anchor 上填充真实值
label_tensor[9, 18, 0, :4] = [0.36, 0.54, 0.24, 0.36]  # tx, ty, tw, th
label_tensor[9, 18, 0, 4] = 1.0  # confidence
label_tensor[9, 18, 0, 5] = 1.0  # person 类别置信度

# 在 car 对应的 grid cell 和 anchor 上填充真实值
label_tensor[5, 10, 3, :4] = [0.72, 0.36, 0.24, 0.24]
label_tensor[5, 10, 3, 4] = 1.0
label_tensor[5, 10, 3, 6] = 1.0  # car 类别置信度
```

---

### Step 4: 损失函数计算

YOLOv2 的损失函数延续了 YOLOv1 的设计，但加入了对 anchor boxes 的支持。

#### 损失函数公式如下：

$$
\mathcal{L} =
\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B}
\mathbb{1}_{ij}^{\text{obj}}
\left[
(x_i - \hat{x}_i)^2 +
(y_i - \hat{y}_i)^2 +
(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 +
(\sqrt{h_i} - \sqrt{\hat{h}_i})^2
\right] \\
- \sum_{i=0}^{S^2} \sum_{j=0}^{B}
\mathbb{1}_{ij}^{\text{obj}} \log(C_i)
- \sum_{i=0}^{S^2} (1 - \mathbb{1}_{i}^{\text{obj}}) \log(1 - C_i) \\
+ \lambda_{cls} \sum_{i=0}^{S^2}
\mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

#### 含义解释：

|损失项|是否参与训练|
|--------|----------------|
|定位损失|仅正样本参与|
|置信度损失|正样本 + 负样本|
|分类损失|仅正样本参与|

---

## 四、YOLOv2 的推理流程详解

### Step 1: 图像输入与预处理

```python
image = cv2.imread("test.jpg")
resized_image = cv2.resize(image, (416, 416)) / 255.0  # 归一化
input_tensor = np.expand_dims(resized_image, axis=0)       # 添加 batch 维度
```

---

### Step 2: 推理输出（来自 Darknet 或 PyTorch 模型）

模型输出一个张量：

```python
output_tensor = model.predict(input_tensor)  # shape: [1, 13, 13, 125]
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
- $(c_x, c_y)$：当前 grid cell 左上角坐标（归一化后）
- $(p_w, p_h)$：对应 anchor 的宽高（归一化后）

#### 示例解码代码（伪代码）：

```python
def decode_box(output_tensor, anchors):
    bboxes = []
    for i in range(13):     # height
        for j in range(13):  # width
            for k in range(5):  # anchor index
                tx, ty, tw, th = output_tensor[i, j, k*25:(k+1)*25][:4]
                conf = output_tensor[i, j, k*25+4]
                class_probs = output_tensor[i, j, k*25+5:k*25+7]

                bx = sigmoid(tx) + j * stride
                by = sigmoid(ty) + i * stride
                bw = anchors[k][0] * exp(tw)
                bh = anchors[k][1] * exp(th)

                x1 = (bx - bw / 2) * image_size
                y1 = (by - bh / 2) * image_size
                x2 = (bx + bw / 2) * image_size
                y2 = (by + bh / 2) * image_size

                bboxes.append([x1, y1, x2, y2, conf, class_probs])
    return bboxes
```

---

### Step 4: 执行 NMS（Non-Maximum Suppression）

#### 计算综合得分：

$$
\text{score} = \text{confidence} \times \max(\text{class\_probs})
$$

#### 示例执行 NMS（PyTorch）：

```python
import torch
from torchvision.ops import nms

# 假设 boxes 是 [N, 4]，scores 是 [N]
keep_indices = nms(boxes, scores, iou_threshold=0.5)

final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
final_labels = labels[keep_indices]
```

---

## 五、YOLOv2 的完整训练与推理流程总结

|阶段|内容|
|------|------|
|输入图像|416 × 416 × 3 RGB 图像|
|数据增强|随机缩放、翻转、颜色扰动等|
|正样本划分|anchor 与 GT IoU 最大者为正样本|
|输出结构|13 × 13 × (5 × (5 + C))|
|损失函数|MSE Loss + BCE Loss|
|推理输出|每个 bounding box 包含 `(x1, y1, x2, y2, score, label)`|
|NMS|默认 greedynms，阈值 0.5|
|支持 Anchor|5 个 anchor boxes，K-Means 聚类获得|

---

## 六、YOLOv2 的关键配置文件片段（来自 .cfg 文件）

```ini
[region]
anchors = 1.08,1.19, 1.32,3.19, 3.03,4.34, 4.22,2.81, 5.92,5.53
bias_match=1
classes=2
coords=4
num=5
softmax=1
jitter=.3
rescore=1
iou_thresh=0.5
```

> 这些配置项在 AlexeyAB/darknet 中真实存在，影响 anchor 匹配、loss 计算、NMS 等流程。

---

## 七、YOLOv2 的性能表现（来源：YOLO 官方文档）

|模型|mAP@COCO|FPS（V100）|是否支持改进 IoU|
|------|-------------|----------------|------------------|
|YOLOv2|~76.8|~67|不支持（默认传统 IoU）|
|YOLOv2-tiny|~63.0|~150|不支持|
|YOLO9000|~76.8（部分类别）|~45|支持联合训练|

---

## 八、YOLOv2 的局限性（来自社区反馈）

|局限性|说明|
|--------|------|
|不支持 DIoU-NMS|需要自定义修改|
|anchor 设置固定|新任务需重新聚类适配|
|输出结构固定|不适合直接部署 ONNX|
|小目标检测一般|相比 YOLOv3 仍有差距|

---

