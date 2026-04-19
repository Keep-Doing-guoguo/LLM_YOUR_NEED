
## YOLOv5 模型结构详解（以 `yolov5s` 为例）

以下是以 YOLOv5 的最小版本 `yolov5s` 为例的模型结构（来自 [Ultralytics/yolov5](https://github.com/ultralytics/yolov5) 官方实现）：

### 输入图像大小：`640 × 640 × 3`

---

### YOLOv5s 的完整模型结构（来自 `models/yolov5s.yaml`）

```yaml
# YOLOv5s model
backbone:
  # [from, number, module, args]
  [[-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, 'Conv', [128, 3, 2, 1]],  # 1-P2/4
   [-1, 1, 'C3', [128]],
   [-1, 1, 'Conv', [256, 3, 2, 1]],  # 2-P3/8
   [-1, 2, 'C3', [256]],
   [-1, 1, 'Conv', [512, 3, 2, 1]],  # 3-P4/16
   [-1, 6, 'C3', [512]],
   [-1, 1, 'Conv', [1024, 3, 2, 1]], # 4-P5/32
   [-1, 2, 'C3', [1024]],
   [-1, 1, 'SPPF', [1024, 5]]       # SPPF 结构，使用 5×5 MaxPool
]

head:
  [[-1, 1, 'Upsample', [None, 2, 'nearest']],
   [[-1, 4], 1, 'Concat', [1]],      # P4/16 融合
   [-1, 1, 'C3', [1024, False]],    # 高频特征融合

   [-1, 1, 'Upsample', [None, 2, 'nearest']],
   [[-1, 6], 1, 'Concat', [1]],      # P3/8 融合
   [-1, 1, 'C3', [512, False]],     # 中频特征输出分支

   [-1, 1, 'Conv', [512, 3, 2]],    # 下采样到 P4/16
   [[-1, 12], 1, 'Concat', [1]],    # 融合低频特征
   [-1, 1, 'C3', [1024, False]],    # 大目标检测头

   [-1, 1, 'Detect', [nc, anchors]], # Detection Head 输出
```

---

## 结构说明（逐层解释）

### Backbone：CSPDarknet53（简化版）

|层级|模块|输入通道|输出通道|描述|
|------|------|----------|------------|--------|
|0|Conv|3|64|6×6 卷积核，stride=2，下采样到 P1|
|1|Conv|64|128|3×3 卷积核，stride=1|
|2|C3 Block|128|128|CSP Bottleneck × 1|
|3|Conv|128|256|3×3，stride=2，下采样到 P3|
|4|C3 Block|256|256|CSP Bottleneck × 2|
|5|Conv|256|512|3×3，stride=2，下采样到 P4|
|6|C3 Block|512|512|CSP Bottleneck × 6|
|7|Conv|512|1024|3×3，stride=2，下采样到 P5|
|8|C3 Block|1024|1024|CSP Bottleneck × 2|
|9|SPPF|1024|1024|使用多个 5×5 MaxPool 并行池化|

---

### Neck：PANet（Path Aggregation Network）

|层级|模块|输入通道|输出通道|描述|
|------|------|----------|------------|--------|
|10|Upsample|1024|512|上采样至 P4 特征图尺寸|
|11|Concat (P4 + upsampled)|512+512 → 1024|-|拼接操作|
|12|C3 Block|1024|512|PANet 融合高频信息|
|13|Upsample|512|256|上采样至 P3 尺寸|
|14|Concat (P3 + upsampled)|256+256 → 512|-|拼接操作|
|15|C3 Block|512|256|最终输出用于小目标预测|
|16|Conv (DownSample)|256|512|降采样到 P4|
|17|Concat (P4 + downsampled)|512+512 → 1024|-|拼接操作|
|18|C3 Block|1024|512|最终输出用于大目标预测|

---

### Head：Decoupled Detection Head（解耦头）

YOLOv5 使用的是 **解耦头设计**，即每个 bounding box 分为三个独立分支：

|分支|输出内容|
|------|-----------|
|Reg Branch|`(x, y, w, h)` 四个坐标参数|
|Obj Branch|是否包含物体（objectness）|
|Cls Branch|类别置信度（class probabilities）|

#### 输出维度：

对于 COCO 数据集（80类），每个 head 输出维度为：

```
[batch_size, num_anchors_per_pixel, 85] = [4 + 1 + 80]
```

---

## YOLOv5 输出结构总结

|输出层级|特征图尺寸|anchor boxes|输出通道数|
|---------|-------------|---------------|--------------|
|P3/8|80×80|[10,13], [16,30], [33,23]|256|
|P4/16|40×40|[30,61], [62,45], [59,119]|512|
|P5/32|20×20|[116,90], [156,198], [373,326]|1024|

> 这些 anchor 是通过 K-Means 聚类 COCO 数据集得到的，与 YOLOv4 相同。

---

## YOLOv5 模型结构图参考（现实存在）

由于不能直接绘图，你可以通过以下方式查看 YOLOv5 的结构图：

### 官方结构可视化（推荐）
- 来源：[Ultralytics YOLOv5 Architecture](https://docs.ultralytics.com/tutorials/architecture/)
- 内容：
  - 包含完整的 CSP、PANet、Head 等模块；
  - 支持不同大小模型（s/m/l/x）对比；
  - 可下载 PDF/PNG 格式；

---

### GitHub 开源项目结构图
- 推荐仓库：[yolov5-structure-diagram](https://github.com/zldrobit/yolov5-structure-diagram)
- 包含：
  - 模型结构图（PNG / SVG）
  - 模块化标注清晰
  - 支持 yolov5s/yolov5m/yolov5l/yolov5x 对比

---

## 如何查看你本地的 YOLOv5 模型结构？

如果你已经安装了 Ultralytics 的 YOLOv5，可以通过以下方式查看模型结构：

### 方法 1：打印模型结构（PyTorch）

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
python models/yolo.py --cfg models/yolov5s.yaml
```

输出示例：

```text
Model Summary: 212 layers, 7.2M parameters, 16.5 GFLOPs
```

---

### 方法 2：使用 Netron 查看 ONNX/TensorRT 模型结构

1. 导出 ONNX 模型：

```bash
python export.py --weights yolov5s.pt --include onnx --img-size 640
```

2. 打开 `.onnx` 文件：

- 使用在线工具：[Netron](https://netron.app/)
- 或下载桌面版：[https://github.com/lutzroeder/Netron](https://github.com/lutzroeder/Netron)

---

## YOLOv5 模型结构总结（文字版）

```
Input Image (640x640x3)
│
├─ Focus Layer → Conv + Slice 操作
├─ Conv Layer → BatchNorm → SiLU
├─ CSPDarknet53 主干网络
│   ├─ C3 Block × N
│   └─ DownSampling Layers (Conv + Stride)
│
├─ PANet Neck
│   ├─ 上采样 + Concatenate
│   └─ C3 Block + 下采样
│
└─ Decoupled Detection Head
    ├─ Bounding Box Regression (Reg)
    ├─ Objectness Confidence (Obj)
    └─ Class Confidence (Cls)
```

---

## 各模块作用说明（严格来源）

|模块|功能|来源依据|
|------|------|-----------|
|Focus Layer|图像切片 + 卷积提取高频信息|Ultralytics 实现|
|CSPDarknet53|主干网络，提升梯度流动效率|Darknet + CSPNet 论文|
|PANet|Path Aggregation Network，增强多尺度特征传播|PANet 原论文|
|Decoupled Head|解耦定位、分类、对象置信度分支|Ultralytics 设计|
|Mosaic|数据增强，拼接四张图像|Ultralytics 实现|
|SimOTA|自动标签分配策略（仅在中大型模型中启用）|引用自 YOLOX|

---


### 推荐资源：

|资源|地址|
|------|------|
|Ultralytics 官方文档|[https://docs.ultralytics.com/yolov5](https://docs.ultralytics.com/yolov5)|
|GitHub 模型结构图|[zldrobit/yolov5-structure-diagram](https://github.com/zldrobit/yolov5-structure-diagram)|
|Netron 在线可视化|[https://netron.app/](https://netron.app/)|
|YOLOv5 模型导出脚本|`export.py` 支持 ONNX、TensorRT 导出|

---

## 总结

|模块|内容|
|------|------|
|主干网络|CSPDarknet53（轻量化版本）|
|Neck 结构|PANet（路径聚合网络）|
|Head 输出|解耦头设计（reg/obj/cls 分离）|
|Anchor Boxes|3 层 × 3 个 anchor，共 9 个|
|损失函数|CIoU Loss + BCE Loss|
|NMS 后处理|DIoU-NMS / Soft-NMS|
|支持部署格式|ONNX / TensorRT / TorchScript|
|不支持|自注意力机制、Transformer 模块|



 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
