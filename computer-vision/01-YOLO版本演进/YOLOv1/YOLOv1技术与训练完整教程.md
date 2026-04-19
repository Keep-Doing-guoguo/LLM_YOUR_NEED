# YOLOv1 技术与训练完整教程

## 一、前言

YOLOv1，全称 **You Only Look Once**，由 Joseph Redmon 等人在 2016 年提出。它将目标检测从传统的多阶段流程，转化为一个端到端的单阶段回归问题。

传统两阶段检测方法通常需要：

```text
候选区域生成 -> 特征提取 -> 分类 -> 边界框回归 -> 后处理
```

YOLOv1 的思路更直接：

```text
输入整张图像 -> 单个网络前向传播 -> 直接输出所有检测框和类别概率
```

它的意义在于：

- 第一次系统性地把检测任务做成实时端到端预测。
- 将目标检测统一为一个回归问题。
- 为后续 YOLOv2、YOLOv3、YOLOv5、YOLOv8 等系列奠定基础。

---

## 二、YOLOv1 的核心思想

YOLOv1 将输入图像划分为 `S x S` 个网格。论文中默认：

```text
S = 7
B = 2
C = 20
```

含义：

- `S`：网格数量，PASCAL VOC 中使用 `7 x 7`
- `B`：每个网格预测的 bounding box 数量，YOLOv1 中为 2
- `C`：类别数量，PASCAL VOC 为 20

核心规则：

> 如果某个目标的中心点落在某个 grid cell 中，那么这个 grid cell 负责预测该目标。

每个 grid cell 预测：

```text
B 个 bounding boxes + C 个类别条件概率
```

每个 bounding box 包含：

```text
x, y, w, h, confidence
```

所以最终输出维度为：

```text
S x S x (B * 5 + C)
```

在 VOC 设置下：

```text
7 x 7 x (2 * 5 + 20) = 7 x 7 x 30
```

---

## 三、YOLOv1 的输入输出

### 3.1 输入

YOLOv1 的输入图像尺寸为：

```text
448 x 448 x 3
```

训练和推理前通常需要：

- resize 图像到 `448 x 448`
- 归一化像素值
- 将标注框转换到归一化坐标

### 3.2 输出

输出张量：

```text
7 x 7 x 30
```

其中每个 grid cell 输出：

```text
box1: x, y, w, h, confidence
box2: x, y, w, h, confidence
class_probs: 20 个类别条件概率
```

### 3.3 参数含义

| 参数 | 含义 |
|------|------|
| `x, y` | box 中心点相对于当前 grid cell 左上角的偏移，范围 0 到 1 |
| `w, h` | box 宽高相对于整张图像的比例，范围 0 到 1 |
| `confidence` | `Pr(object) * IoU(pred, truth)` |
| `class_probs` | `Pr(class_i | object)` |

最终某个类别的检测分数：

```text
score_i = confidence * Pr(class_i | object)
```

---

## 四、网络结构

YOLOv1 的主干网络受 GoogLeNet 启发，但没有使用 Inception 模块，而是使用 `1x1` 和 `3x3` 卷积交替堆叠。

整体结构可以概括为：

```text
Input: 448 x 448 x 3
  -> Conv + LeakyReLU
  -> MaxPool
  -> 多层 1x1 / 3x3 Conv
  -> Flatten
  -> Fully Connected
  -> Output: 7 x 7 x 30
```

YOLOv1 检测网络包含：

- 24 个卷积层
- 2 个全连接层

最后输出：

```text
1470 = 7 * 7 * 30
```

再 reshape 成：

```text
[7, 7, 30]
```

---

## 五、DarkNet 预训练的作用

YOLOv1 训练检测模型前，通常会先训练一个 DarkNet 分类模型。

原因是：

- 检测数据集规模相对较小。
- 从头训练 Backbone 容易不稳定。
- 分类预训练可以让卷积层先学到通用视觉特征。

预训练流程：

```text
分类数据集
  -> 训练 DarkNet 分类网络
  -> 保存卷积层权重
  -> 加载到 YOLOv1 检测模型
  -> 检测头重新初始化并训练
```

DarkNet 分类网络通常包括：

- `Conv2d + BatchNorm2d + LeakyReLU`
- `MaxPool2d`
- `1x1` 与 `3x3` 卷积交替
- 分类头：`AvgPool2d + Linear`

示例：

```python
class DarkNet(nn.Module):
    ...
    def _make_fc_layers(self):
        return nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, num_classes)
        )
```

---

## 六、DarkNet 分类训练流程

### 6.1 定义模型与优化器

```python
model = DarkNet(conv_only=False, bn=True, init_weight=True)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

### 6.2 数据集格式

如果使用 `ImageFolder`，目录结构建议为：

```text
archive/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...
```

### 6.3 加载数据

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(root="archive/train", transform=transform),
    batch_size=64,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(root="archive/val", transform=transform),
    batch_size=64,
    shuffle=False
)
```

### 6.4 分类训练主循环

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/darknet")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    acc = correct / total
    avg_loss = running_loss / len(train_loader)

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)

writer.close()
```

### 6.5 TensorBoard 可视化

```bash
tensorboard --logdir=runs
```

默认访问地址：

```text
http://localhost:6006
```

常见记录指标：

- `train/loss`
- `train/top1`
- `train/top5`
- `test/loss`
- `test/top1`
- `test/top5`
- `lr`

---

## 七、加载 DarkNet 权重训练 YOLOv1

分类预训练完成后，只加载卷积层权重到 YOLOv1。

常见做法：

```python
pretrained_dict = torch.load(darknet_weight_path)
model_dict = yolo.backbone.state_dict()

matched_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}

model_dict.update(matched_dict)
yolo.backbone.load_state_dict(model_dict)
```

注意：

- Backbone 卷积层加载预训练权重。
- 检测头通常重新初始化。
- 分类任务和检测任务输出不同，最后分类层不能直接迁移。

---

## 八、VOC 数据集准备

YOLOv1 常使用 PASCAL VOC 数据集。

典型目录：

```text
VOC2012/
  JPEGImages/
  Annotations/
  ImageSets/
```

其中：

- `JPEGImages`：原始图像
- `Annotations`：XML 标注
- `ImageSets/Main`：训练集、验证集划分

示例配置：

```python
label_path = "data/voc2012.txt"
image_dir = "/path/to/VOCdevkit/VOC2012/JPEGImages"
annotation_dir = "/path/to/VOCdevkit/VOC2012/Annotations"
```

---

## 九、YOLOv1 标签构建

训练标签需要构造成：

```text
7 x 7 x 30
```

对于每个目标：

1. 读取 VOC XML 中的 `(xmin, ymin, xmax, ymax)`。
2. 转换为归一化中心点格式 `(cx, cy, w, h)`。
3. 根据 `(cx, cy)` 判断目标中心落在哪个 grid cell。
4. 该 grid cell 负责该目标。
5. 写入 box 坐标、confidence 和类别 one-hot。

坐标转换：

```python
cx = (xmin + xmax) / 2 / image_width
cy = (ymin + ymax) / 2 / image_height
w = (xmax - xmin) / image_width
h = (ymax - ymin) / image_height
```

确定 grid cell：

```python
grid_x = int(cx * S)
grid_y = int(cy * S)
```

cell 内偏移：

```python
x_cell = cx * S - grid_x
y_cell = cy * S - grid_y
```

---

## 十、边界框解码

YOLOv1 输出的 `x, y` 是相对于当前 grid cell 的偏移。

假设：

- 网格大小为 `S x S`
- 当前 cell 位于第 `i` 行、第 `j` 列
- 模型输出为 `x, y, w, h`

则整图归一化坐标为：

```text
x_abs = (j + x) / S
y_abs = (i + y) / S
w_abs = w
h_abs = h
```

示例：

```text
S = 7
i = 2
j = 3
x = 0.5
y = 0.5
w = 0.2
h = 0.3
```

解码：

```text
x_abs = (3 + 0.5) / 7 = 0.5
y_abs = (2 + 0.5) / 7 = 0.357
w_abs = 0.2
h_abs = 0.3
```

---

## 十一、正负样本分配

YOLOv1 没有 anchor。它的正负样本分配规则比较简单：

1. 每个目标由其中心点所在的 grid cell 负责。
2. 该 grid cell 中有 `B=2` 个预测框。
3. 计算这两个预测框与真实框的 IoU。
4. IoU 更大的那个预测框负责该目标。
5. 负责目标的预测框计算坐标损失和 object confidence 损失。
6. 不负责目标的预测框计算 no-object confidence 损失。

这也是 YOLOv1 的一个限制：

> 一个 grid cell 对多个中心落在同一 cell 的目标处理能力有限。

---

## 十二、损失函数

YOLOv1 的损失使用 **sum-squared error**，可以分为 4 部分：

1. 坐标损失
2. 有目标 confidence 损失
3. 无目标 confidence 损失
4. 分类损失

### 12.1 坐标损失

```text
L_coord = lambda_coord * sum 1_obj [
    (x - x_hat)^2
  + (y - y_hat)^2
  + (sqrt(w) - sqrt(w_hat))^2
  + (sqrt(h) - sqrt(h_hat))^2
]
```

这里对 `w, h` 开平方，是为了降低大目标尺寸误差对损失的主导影响，让小目标更敏感。

### 12.2 Confidence 损失

有目标框：

```text
L_obj = sum 1_obj (C - C_hat)^2
```

无目标框：

```text
L_noobj = lambda_noobj * sum 1_noobj (C - C_hat)^2
```

其中：

```text
C_hat = IoU(pred_box, gt_box)
```

常用权重：

```text
lambda_coord = 5
lambda_noobj = 0.5
```

### 12.3 分类损失

分类损失只在有目标的 grid cell 上计算：

```text
L_cls = sum 1_obj sum_c (p(c) - p_hat(c))^2
```

### 12.4 总损失

```text
L_total = L_coord + L_obj + L_noobj + L_cls
```

注意：

- YOLOv1 原论文中损失主要是平方误差，不是现代 YOLO 常见的 BCE、CIoU、DFL。
- 这也是后续 YOLO 版本不断改进损失函数的重要原因。

---

## 十三、YOLOv1 训练主流程

训练流程可以概括为：

```text
读取图像和 XML 标注
  -> resize 到 448 x 448
  -> 构建 7 x 7 x 30 标签
  -> 模型前向传播
  -> 计算 YOLOv1 loss
  -> 反向传播
  -> 保存 checkpoint
```

伪代码：

```python
for images, targets in train_loader:
    images = images.to(device)
    targets = targets.to(device)

    preds = model(images)
    loss = criterion(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 十四、推理流程

YOLOv1 推理流程：

```text
输入图像
  -> resize 到 448 x 448
  -> 模型输出 7 x 7 x 30
  -> 解码 bounding boxes
  -> 计算 class-specific score
  -> 过滤低分框
  -> NMS
  -> 绘制检测结果
```

类别分数：

```text
score_class = confidence * class_probability
```

NMS 一般按类别分别执行：

```python
for class_id in range(num_classes):
    boxes_cls = boxes[classes == class_id]
    scores_cls = scores[classes == class_id]
    keep = nms(boxes_cls, scores_cls, iou_threshold=0.5)
```

---

## 十五、训练脚本功能整理

如果项目中有 `train_darknet.py`，它通常负责：

| 功能 | 说明 |
|------|------|
| 模型结构 | 使用 DarkNet 分类模型 |
| 数据集 | 使用 `ImageFolder` 加载分类数据 |
| TensorBoard | 记录 loss、accuracy、lr |
| Resume | 从 checkpoint 恢复训练 |
| Top-K Accuracy | 计算 top-1 和 top-5 分类准确率 |
| Checkpoint | 保存当前模型和最佳模型 |
| 参数配置 | 使用 argparse 或 namespace 配置训练参数 |

如果项目中有 `train_yolo.py`，它通常负责：

| 功能 | 说明 |
|------|------|
| 加载 Backbone | 加载 DarkNet 预训练权重 |
| 加载 VOC | 读取图像和 XML 标注 |
| 构建标签 | 生成 `7 x 7 x 30` 标签 |
| 检测训练 | 训练 YOLOv1 检测模型 |
| 保存权重 | 保存 `last.pth` 和 `best.pth` |

---

## 十六、YOLOv1 的优点

| 优点 | 说明 |
|------|------|
| 速度快 | 单阶段检测，一次前向完成预测 |
| 结构统一 | 分类、定位、置信度在一个网络中完成 |
| 全局建模 | 网络看整张图，背景误检相对更少 |
| 易于理解 | 网格划分和回归思想直观 |
| 开创性强 | 奠定后续 YOLO 系列基础 |

---

## 十七、YOLOv1 的局限

| 局限 | 说明 |
|------|------|
| 小目标效果差 | 低分辨率网格难以表示小目标 |
| 密集目标困难 | 一个 cell 只能负责有限数量目标 |
| 定位不够精细 | 使用 MSE 回归框坐标 |
| 没有 anchor | 缺少先验框机制 |
| 输出约束强 | `7 x 7` 网格限制了预测灵活性 |

---

## 十八、YOLOv1 与后续版本的关系

| 版本 | 相比 YOLOv1 的主要变化 |
|------|------------------------|
| YOLOv2 | 引入 anchor boxes、BatchNorm、多尺度训练 |
| YOLOv3 | 使用 Darknet-53、多尺度预测、logistic 分类 |
| YOLOv4 | 引入大量 Bag of Freebies / Bag of Specials |
| YOLOv5+ | 工程化训练、自动 anchor、数据增强、部署生态 |
| YOLOv8+ | anchor-free、解耦头、TAL、DFL 等现代机制 |

---

## 十九、学习 YOLOv1 时最重要的几个问题

### 19.1 为什么 YOLOv1 快？

因为它不生成候选区域，也不对每个 proposal 单独分类，而是整图一次前向直接输出所有预测。

### 19.2 为什么 YOLOv1 对小目标不好？

因为输入被划分成 `7 x 7` 网格，一个 cell 表达能力有限，小目标和密集目标容易冲突。

### 19.3 为什么需要 DarkNet 预训练？

因为检测数据有限，先训练分类 Backbone 可以获得更好的通用视觉特征。

### 19.4 YOLOv1 有没有 anchor？

没有。anchor 是 YOLOv2 开始引入的重要机制。

### 19.5 YOLOv1 的 confidence 是什么？

```text
confidence = Pr(object) * IoU(pred, truth)
```

推理时类别最终得分为：

```text
score = confidence * class_probability
```

---

## 二十、总结

YOLOv1 的核心可以概括为：

1. 将目标检测统一成单阶段回归任务。
2. 将图像划分为 `7 x 7` 网格。
3. 每个 grid cell 预测 2 个框和 20 个类别条件概率。
4. 通过 DarkNet 提取整图特征。
5. 使用平方误差损失同时优化位置、置信度和分类。
6. 推理阶段通过类别分数和 NMS 得到最终检测结果。

YOLOv1 虽然已经不是现代工程中的主流模型，但它是理解后续 YOLO 系列的起点。学清楚 YOLOv1，后面再看 anchor、multi-scale prediction、decoupled head、TAL、DFL 等机制会更容易。

