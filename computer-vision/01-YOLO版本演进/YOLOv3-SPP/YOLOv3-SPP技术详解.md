# YOLOv3-SPP 技术详解

YOLOv3-SPP 是在 YOLOv3 基础上加入 SPP 模块的改进版本。它保留了 YOLOv3 的 Darknet-53、anchor-based 检测头、三尺度预测和 NMS 后处理，同时在深层特征处加入多尺度最大池化，用来扩大感受野、增强高层语义信息。

从学习角度看，YOLOv3-SPP 不适合拆成很多孤立文件。SPP 模块、anchor 聚类、训练配置、ONNX 导出和部署后处理都围绕同一条链路展开：

```text
输入图像
  -> Darknet-53 提取特征
  -> SPP 增强深层特征
  -> FPN-like 多尺度融合
  -> 三个 YOLO 检测头输出
  -> anchor 匹配与损失计算
  -> 推理解码
  -> 置信度过滤与 NMS
  -> ONNX/TensorRT/ONNX Runtime 部署
```

本文把原目录中的模型结构、SPP 模块、AutoAnchor 聚类和 ONNX 部署内容合并为一篇，重复的背景说明已经删减，容易混淆的实现差异也做了区分。

## 一、YOLOv3-SPP 是什么

YOLOv3-SPP 可以理解为：

```text
YOLOv3 + Spatial Pyramid Pooling
```

它的主体仍然是 YOLOv3：

| 部分 | 说明 |
|---|---|
| Backbone | 通常仍是 Darknet-53 |
| Neck | 类 FPN 的多尺度特征融合 |
| Head | YOLOv3 anchor-based 检测头 |
| 输出尺度 | stride 8、16、32 三个尺度 |
| 后处理 | 置信度筛选 + NMS |

SPP 的作用是在深层特征图上做多个不同核大小的最大池化，然后把原始特征与池化后的特征拼接。这样模型不用继续下采样，也能获得更大的感受野。

## 二、与 YOLOv3 的关系

YOLOv3-SPP 不是 YOLOv4，也不是 YOLOv5。它更像是 YOLOv3 的一个增强配置。

| 对比项 | YOLOv3 | YOLOv3-SPP |
|---|---|---|
| 主体框架 | Darknet-53 + FPN-like + YOLO Head | 基本相同 |
| SPP 模块 | 无 | 有 |
| 检测尺度 | 三尺度 | 三尺度 |
| anchor | 9 个 anchor | 通常仍为 9 个 anchor |
| NMS | 标准 NMS 或实现扩展 | 标准 NMS 或实现扩展 |
| 主要提升 | 基础检测能力 | 深层语义与大目标检测能力 |

需要注意：不同开源实现会把 YOLOv3-SPP 和其他改进混合在一起，例如 CIoU loss、DIoU-NMS、Mosaic、CSPDarknet 等。这些可以作为工程增强项，但不能都算作 YOLOv3-SPP 的原始核心定义。

## 三、整体结构

YOLOv3-SPP 的结构可以概括为：

```text
Input
  -> Darknet-53
  -> SPP on deepest feature
  -> feature fusion
  -> detect at stride 32
  -> upsample + concat
  -> detect at stride 16
  -> upsample + concat
  -> detect at stride 8
```

输出尺度取决于输入尺寸。不能固定写成 80x80、40x40、20x20。

| 输入尺寸 | stride 8 | stride 16 | stride 32 |
|---:|---:|---:|---:|
| 416x416 | 52x52 | 26x26 | 13x13 |
| 608x608 | 76x76 | 38x38 | 19x19 |
| 640x640 | 80x80 | 40x40 | 20x20 |

所以更准确的写法是：

```text
P3: input / 8
P4: input / 16
P5: input / 32
```

其中 P3 负责小目标，P4 负责中等目标，P5 负责大目标。

## 四、SPP 模块原理

SPP 是 Spatial Pyramid Pooling 的缩写。标准 SPPNet 中的 SPP 主要用于适配不同尺寸输入，而 YOLOv3-SPP 中的 SPP 更偏向特征增强：通过多个不同大小的 max pooling 扩大深层特征的感受野。

YOLOv3-SPP 常用池化核为：

```text
5x5, 9x9, 13x13
```

简化结构如下：

```text
输入特征 x
  -> MaxPool 5x5
  -> MaxPool 9x9
  -> MaxPool 13x13
  -> concat(x, pool5, pool9, pool13)
  -> 输出增强特征
```

假设输入特征为 `[B, C, H, W]`，使用 stride=1 和合适 padding 后，每个池化分支输出仍是 `[B, C, H, W]`。拼接后输出为：

```text
[B, 4C, H, W]
```

## 五、SPP 的 PyTorch 简化实现

下面是一个符合 YOLOv3-SPP 思想的简化实现：

```python
import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, pool_sizes=(5, 9, 13)):
        super().__init__()
        self.maxpools = nn.ModuleList(
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in pool_sizes
        )

    def forward(self, x):
        features = [x]
        for pool in self.maxpools:
            features.append(pool(x))
        return torch.cat(features, dim=1)
```

需要注意：Darknet 的 `.cfg` 文件在不同版本中可能不是写成一个抽象的 `[spp]` 块，而是通过多个 `[maxpool]` 和 `[route]` 层组合出来。学习时理解结构比死记配置块名称更重要。

## 六、SPP 放在什么位置

SPP 通常放在 Darknet-53 最深层特征附近，也就是 stride 32 的高语义特征图上。

原因是：

| 位置 | 适合做 SPP 吗 | 原因 |
|---|---|---|
| 浅层特征 | 不太适合 | 分辨率高，计算量大，语义较弱 |
| 中层特征 | 可以但不常见 | 计算量和收益需要权衡 |
| 深层特征 | 适合 | 语义强，尺寸小，扩大感受野收益明显 |

SPP 不改变特征图的空间尺寸，只改变通道信息，因此它适合插在 backbone 和后续特征融合之间。

## 七、检测头输出

YOLOv3-SPP 继承 YOLOv3 的 anchor-based 检测方式。对于 COCO 的 80 类检测任务，每个 anchor 输出：

```text
tx, ty, tw, th, objectness, class_1, ..., class_80
```

维度为：

```text
4 + 1 + 80 = 85
```

每个尺度通常有 3 个 anchor，因此每个检测头的输出通道数为：

```text
3 x 85 = 255
```

以 416x416 输入为例：

| 层级 | stride | 输出尺寸 | 输出通道 | 预测框数量 |
|---|---:|---:|---:|---:|
| P3 | 8 | 52x52 | 255 | 52 x 52 x 3 |
| P4 | 16 | 26x26 | 255 | 26 x 26 x 3 |
| P5 | 32 | 13x13 | 255 | 13 x 13 x 3 |

YOLOv3-SPP 的检测头不是现代 YOLO 中常说的解耦头。它仍然是 YOLOv3 风格的耦合检测头，分类、objectness 和 bbox 参数由同一个检测分支输出。

## 八、Anchor 设置

YOLOv3-SPP 通常沿用 YOLOv3 在 COCO 上的 9 个 anchor：

```ini
anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326
```

常见分配方式是：

| 检测层 | 目标类型 | anchor |
|---|---|---|
| P3, stride 8 | 小目标 | 10x13, 16x30, 33x23 |
| P4, stride 16 | 中等目标 | 30x61, 62x45, 59x119 |
| P5, stride 32 | 大目标 | 116x90, 156x198, 373x326 |

如果训练自定义数据集，默认 anchor 不一定合适。目标尺寸分布和 COCO 差异越大，重新聚类 anchor 的收益越明显。

## 九、AutoAnchor 聚类

AutoAnchor 的核心目的很简单：根据当前数据集的标注框宽高，重新生成更适合该数据集的 anchor。

完整流程如下：

```text
读取标注文件
  -> 提取所有 bbox 的宽高
  -> 按训练输入尺寸还原到像素尺度
  -> K-Means 或遗传算法优化 anchor
  -> 得到 9 个 anchor
  -> 写回 cfg/yaml 配置
  -> 重新训练并观察召回率和 mAP
```

一个简化的 K-Means 示例：

```python
import numpy as np
from sklearn.cluster import KMeans


def kmeans_anchors(boxes_wh, n_clusters=9):
    boxes_wh = np.asarray(boxes_wh, dtype=np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans.fit(boxes_wh)
    anchors = kmeans.cluster_centers_
    areas = anchors[:, 0] * anchors[:, 1]
    return anchors[np.argsort(areas)]
```

标注读取时要注意：YOLO 标签中的 `w, h` 通常是归一化坐标，需要乘以输入尺寸：

```python
def load_yolo_label_wh(label_files, img_size):
    boxes = []
    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, _, _, w, h = map(float, parts)
                boxes.append([w * img_size, h * img_size])
    return boxes
```

聚类后，把结果写入配置中的 `anchors` 字段即可。

## 十、什么时候需要重新聚类 anchor

不是所有项目都需要重新聚类 anchor。

| 场景 | 是否建议 |
|---|---|
| 使用 COCO 类似数据 | 可以先不改 |
| 小目标特别多 | 建议聚类 |
| 目标长宽比很特殊 | 建议聚类 |
| 工业缺陷、遥感、票据等垂直场景 | 建议聚类 |
| 只是跑通 demo | 不必优先处理 |

判断 anchor 是否合适，可以看训练早期的 recall、正样本匹配情况、bbox loss 收敛速度，以及验证集 mAP。

## 十一、训练流程

YOLOv3-SPP 的训练流程与 YOLOv3 基本一致：

```text
读取图像与标注
  -> resize / multi-scale training / 数据增强
  -> Darknet-53 前向
  -> SPP 增强深层特征
  -> 三尺度检测头输出
  -> ground truth 匹配最佳 anchor
  -> 构造正样本、负样本、ignore 样本
  -> 计算定位、objectness、分类损失
  -> 反向传播
```

在 AlexeyAB/darknet 等工程实现中，可能会支持额外配置，例如：

```ini
iou_loss=ciou
nms_kind=diounms
ignore_thresh=.7
truth_thresh=1
```

这些配置代表的是具体实现中的增强能力。学习时建议这样区分：

| 内容 | 是否属于 YOLOv3-SPP 核心 |
|---|---|
| SPP 模块 | 是 |
| 三尺度检测 | 来自 YOLOv3，保留 |
| anchor-based 匹配 | 来自 YOLOv3，保留 |
| CIoU loss | 工程实现可选增强 |
| DIoU-NMS | 工程实现可选增强 |
| Mosaic / CopyPaste | 后续训练增强，不是核心定义 |
| CSPDarknet | 其他变体，不是标准 YOLOv3-SPP 必备项 |

## 十二、推理流程

推理阶段流程如下：

```text
输入图像
  -> resize/letterbox
  -> 归一化
  -> 前向推理
  -> 解码三个尺度的 anchor 输出
  -> score = objectness * class probability
  -> 置信度阈值过滤
  -> NMS
  -> 坐标映射回原图
```

YOLOv3-SPP 的 ONNX 或 TensorRT 模型通常只负责前向计算。bbox 解码和 NMS 是否放进模型图，取决于导出脚本和部署框架。很多部署方案会把后处理放在模型外部实现。

## 十三、ONNX 导出

原生 Darknet `.weights` 并不天然等价于 ONNX 模型。常见路线有两种：

| 路线 | 说明 |
|---|---|
| Darknet 权重转 PyTorch 再导出 | 常见于 Ultralytics/yolov3 等实现 |
| 使用第三方转换工具 | 需要检查算子和后处理是否一致 |

PyTorch 导出 ONNX 的基本形式如下：

```python
import torch


model.eval()
dummy = torch.randn(1, 3, 416, 416)

torch.onnx.export(
    model,
    dummy,
    "yolov3-spp.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["outputs"],
    dynamic_axes={
        "images": {0: "batch"},
        "outputs": {0: "batch"},
    },
)
```

如果使用 Ultralytics 风格工程，可能会看到类似命令：

```bash
python export.py --weights yolov3-spp.pt --include onnx --img-size 416
```

这类命令依赖具体仓库的模型定义，不能直接套到所有 Darknet 权重上。导出前要确认模型结构、类别数、anchor、输入尺寸和权重完全匹配。

## 十四、ONNX 输出与后处理

ONNX 输出有两种常见形式。

第一种是保留三个原始检测头：

```text
[B, 255, H3, W3]
[B, 255, H4, W4]
[B, 255, H5, W5]
```

第二种是在导出脚本里已经 reshape/concat：

```text
[B, num_boxes, 85]
```

部署时必须先确认输出格式，再写后处理代码。不能只根据文件名判断。

ONNX Runtime 基本验证流程：

```python
import numpy as np
import onnx
import onnxruntime as ort


model = onnx.load("yolov3-spp.onnx")
onnx.checker.check_model(model)

session = ort.InferenceSession("yolov3-spp.onnx")
images = np.random.rand(1, 3, 416, 416).astype(np.float32)
outputs = session.run(None, {"images": images})

for output in outputs:
    print(output.shape)
```

后处理要保证这些内容与训练一致：

| 项目 | 必须一致 |
|---|---|
| 输入尺寸 | 是 |
| letterbox 方式 | 是 |
| RGB/BGR 顺序 | 是 |
| 归一化方式 | 是 |
| anchor 数值 | 是 |
| 类别数 | 是 |
| stride | 是 |
| conf threshold | 需要按任务调 |
| NMS threshold | 需要按任务调 |

## 十五、TensorRT / OpenVINO 部署

ONNX 通常是中间格式，实际部署可能继续转为 TensorRT、OpenVINO、NCNN、MNN 等格式。

TensorRT 示例流程：

```text
PyTorch/Darknet
  -> ONNX
  -> TensorRT engine
  -> 推理
  -> 解码与 NMS
```

常见转换命令形式：

```bash
trtexec --onnx=yolov3-spp.onnx --saveEngine=yolov3-spp.engine --fp16
```

部署时重点不是“能不能转换成功”，而是转换后输出是否和原模型一致。建议至少做以下验证：

1. 同一张图片，PyTorch/Darknet 与 ONNX 输出是否接近。
2. ONNX 与 TensorRT 输出是否接近。
3. 后处理前后的坐标尺度是否正确。
4. NMS 结果是否和原工程一致。
5. FP16/INT8 量化后 mAP 是否可接受。

## 十六、常见问题

### 1. 416 输入为什么有人写 80/40/20 输出？

80/40/20 对应的是 640 输入，不是 416 输入。YOLOv3-SPP 的输出尺寸由 stride 决定：

```text
输出尺寸 = 输入尺寸 / stride
```

416 输入对应 52/26/13。

### 2. YOLOv3-SPP 是否使用 CSPDarknet？

标准 YOLOv3-SPP 通常仍以 Darknet-53 为主体。CSPDarknet 属于后续变体或其他工程组合，不能作为 YOLOv3-SPP 的必备结构。

### 3. YOLOv3-SPP 是否是解耦头？

不是。YOLOv3-SPP 仍然是 YOLOv3 风格的耦合检测头。现代 YOLO 中常说的 decoupled head 主要出现在更后面的版本或其他实现中。

### 4. SPP 是否主要提升小目标？

SPP 放在深层特征上，主要增强大感受野和语义表达，对大目标和复杂上下文更直接。小目标检测更多依赖 stride 8 的高分辨率检测层、数据增强、anchor 设置和训练策略。

### 5. ONNX 导出后为什么检测结果不一致？

常见原因包括：

| 原因 | 说明 |
|---|---|
| 预处理不一致 | resize、letterbox、RGB/BGR、归一化不同 |
| anchor 不一致 | 部署后处理使用了错误 anchor |
| 输出格式误读 | `[B,255,H,W]` 和 `[B,N,85]` 混淆 |
| sigmoid/exp 重复执行 | 模型图内外重复解码 |
| NMS 实现不同 | per-class NMS 和 class-agnostic NMS 不一致 |

## 十七、优点与局限

YOLOv3-SPP 的优点：

| 优点 | 说明 |
|---|---|
| 改动小 | 基于 YOLOv3 增加 SPP，理解和部署成本较低 |
| 感受野更大 | 多尺度 max pooling 增强深层特征 |
| 对大目标友好 | 深层语义特征更充分 |
| 工程成熟 | Darknet、PyTorch 复现和部署资料较多 |

YOLOv3-SPP 的局限：

| 局限 | 说明 |
|---|---|
| 仍依赖 anchor | 自定义数据集可能需要重新聚类 |
| 仍依赖 NMS | 密集目标场景可能出现误删 |
| 检测头较旧 | 没有现代解耦头设计 |
| 标签分配较简单 | 没有 SimOTA、TAL 等动态分配 |
| 部署后处理复杂 | ONNX 输出和 NMS 常需要手工对齐 |

## 十八、学习重点总结

学习 YOLOv3-SPP 时，重点掌握下面几个问题：

1. YOLOv3-SPP 相比 YOLOv3 主要增加了什么？
2. SPP 为什么通常放在深层特征处？
3. 5x5、9x9、13x13 max pooling 为什么不会改变空间尺寸？
4. 输入尺寸和输出尺度之间是什么关系？
5. 416 输入为什么对应 52/26/13？
6. 什么时候需要重新聚类 anchor？
7. ONNX 导出后，哪些后处理细节必须和训练工程一致？
8. 哪些内容是 YOLOv3-SPP 核心，哪些只是后续工程增强？

## 十九、结论

YOLOv3-SPP 的核心是：在 YOLOv3 的深层特征上加入 SPP，用很小的结构改动扩大感受野，提升高层语义表达。

它适合作为 YOLOv3 到 YOLOv4、YOLOv5 之间的过渡版本来学习。理解它之后，再看后续版本中的 SPPF、PAN/FPN、CSP、CIoU、DIoU-NMS、AutoAnchor、解耦头和动态标签分配，会更容易看清这些技术各自解决的问题。
