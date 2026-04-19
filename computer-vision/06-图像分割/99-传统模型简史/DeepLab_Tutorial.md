# DeepLab 系列详解：从空洞卷积到 DeepLabv3+

> DeepLab 是语义分割领域非常经典的一条技术路线。它的核心贡献不是单一模型，而是一系列围绕 **空洞卷积、ASPP、多尺度上下文、边界细化、Encoder-Decoder** 的改进。

---

## 1. DeepLab 是什么

DeepLab 是 Google 提出的语义分割系列模型，主要用于解决图像中的像素级分类问题。

语义分割任务的目标是：

```text
输入：一张图像
输出：每个像素所属的类别
```

例如：

```text
road / sky / person / car / building / background
```

DeepLab 系列的发展大致如下：

| 版本 | 代表论文 | 核心贡献 |
|------|----------|----------|
| DeepLabv1 | Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs | 空洞卷积 + CRF |
| DeepLabv2 | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | ASPP + 多尺度上下文 |
| DeepLabv3 | Rethinking Atrous Convolution for Semantic Image Segmentation | 改进 ASPP，去掉 CRF 依赖 |
| DeepLabv3+ | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation | Encoder-Decoder + 深度可分离空洞卷积 |

---

## 2. DeepLab 想解决的问题

FCN 证明了分类 CNN 可以改造成语义分割网络，但 FCN 有几个明显问题：

| 问题 | 说明 |
|------|------|
| 输出太粗 | Backbone 多次下采样导致特征图分辨率很低 |
| 小目标丢失 | 低分辨率特征图难以保留小目标 |
| 边界不准 | 上采样无法恢复精细边界 |
| 上下文不足 | 单一尺度特征很难理解不同大小目标 |

DeepLab 系列主要围绕这几个问题改进：

1. 用**空洞卷积**扩大感受野，同时减少下采样。
2. 用 **ASPP** 捕获多尺度上下文。
3. 用 **CRF** 或 Decoder 改善边界。
4. 用 **Encoder-Decoder** 恢复空间细节。
5. 用 **深度可分离卷积**降低计算量。

---

## 3. DeepLab 系列核心技术概览

DeepLab 的核心技术可以概括为：

```text
DeepLab = Backbone + Atrous Convolution + ASPP + Upsampling / Decoder
```

不同版本的侧重点不同：

```text
DeepLabv1: Atrous Convolution + CRF
DeepLabv2: ASPP + CRF
DeepLabv3: Stronger ASPP + image-level features
DeepLabv3+: Encoder-Decoder + Atrous Separable Convolution
```

---

## 4. 空洞卷积 Atrous Convolution

### 4.1 为什么需要空洞卷积

普通 CNN 中，想扩大感受野通常有两种方式：

1. 增加卷积层深度。
2. 使用 pooling 或 stride conv 下采样。

但在语义分割中，下采样太多会导致输出很粗：

```text
输入图像：512 x 512
下采样 32 倍：16 x 16
```

这种低分辨率特征图对分类有用，但对像素级定位不友好。

空洞卷积的作用是：

> 在不降低特征图分辨率的情况下扩大感受野。

---

### 4.2 空洞卷积的直观理解

普通 `3x3` 卷积：

```text
x x x
x x x
x x x
```

空洞率 `rate=2` 的 `3x3` 空洞卷积：

```text
x . x . x
. . . . .
x . x . x
. . . . .
x . x . x
```

虽然卷积核参数仍然是 `3x3`，但实际覆盖的区域变成了 `5x5`。

空洞率 `rate=3`：

```text
x . . x . . x
. . . . . . .
. . . . . . .
x . . x . . x
. . . . . . .
. . . . . . .
x . . x . . x
```

实际覆盖范围更大。

---

### 4.3 感受野计算

对于 `kernel_size = k`、空洞率 `r` 的卷积，有效卷积核大小为：

```text
k_eff = k + (k - 1) * (r - 1)
```

例如：

| kernel | rate | 有效卷积核 |
|--------|------|------------|
| 3x3 | 1 | 3x3 |
| 3x3 | 2 | 5x5 |
| 3x3 | 6 | 13x13 |
| 3x3 | 12 | 25x25 |

---

### 4.4 PyTorch 中的空洞卷积

```python
import torch
import torch.nn as nn


conv = nn.Conv2d(
    in_channels=256,
    out_channels=256,
    kernel_size=3,
    padding=2,
    dilation=2
)

x = torch.randn(1, 256, 64, 64)
y = conv(x)
print(y.shape)  # torch.Size([1, 256, 64, 64])
```

注意：

```text
padding = dilation
```

当 `kernel_size=3` 时，通常可以保持输入输出尺寸一致。

---

## 5. Output Stride

DeepLab 中经常出现一个概念：

```text
Output Stride
```

它表示：

```text
输入图像尺寸 / 输出特征图尺寸
```

例如：

```text
输入图像：512 x 512
输出特征图：32 x 32
Output Stride = 16
```

常见设置：

| Output Stride | 特征图大小 | 特点 |
|---------------|------------|------|
| 32 | 最小 | 速度快，但细节差 |
| 16 | 常用 | 精度和速度平衡 |
| 8 | 更精细 | 精度更高，但计算量更大 |

DeepLab 通过修改 Backbone 后几层的 stride 和 dilation 来控制 output stride。

例如把 ResNet 后面某些 stride=2 改成 stride=1，再用 dilation 补偿感受野。

---

## 6. DeepLabv1

DeepLabv1 的主要思想：

```text
CNN 特征提取
  -> 空洞卷积保持较高分辨率
  -> 双线性上采样
  -> 全连接 CRF 优化边界
```

### 6.1 DeepLabv1 的贡献

| 贡献 | 说明 |
|------|------|
| 空洞卷积 | 在不额外下采样的情况下扩大感受野 |
| Large Field of View | 通过空洞卷积获得更大上下文 |
| CRF 后处理 | 用全连接条件随机场细化边界 |

### 6.2 为什么需要 CRF

CNN 输出的分割图通常比较平滑，边界容易模糊。

CRF 的作用是根据：

- 像素颜色相似性
- 像素空间距离
- 类别一致性

对分割结果做边界细化。

简化理解：

```text
如果两个像素位置接近、颜色相似，那么它们更可能属于同一类别。
如果两个像素颜色差异明显，那么边界可能在它们之间。
```

### 6.3 DeepLabv1 的局限

- CRF 是后处理，不是端到端主干的一部分。
- 训练和推理流程相对复杂。
- 多尺度上下文建模还不够系统。

---

## 7. DeepLabv2

DeepLabv2 的核心改进是：

```text
ASPP: Atrous Spatial Pyramid Pooling
```

### 7.1 ASPP 是什么

ASPP 使用多个不同 dilation rate 的空洞卷积分支，并行提取不同尺度上下文。

结构示意：

```text
Input Feature
  |--------- 3x3 conv, rate=6  --------|
  |--------- 3x3 conv, rate=12 --------|
  |--------- 3x3 conv, rate=18 --------|
  |--------- 3x3 conv, rate=24 --------|
                         |
                         v
                   sum / concat
```

不同 rate 对应不同感受野：

```text
小 rate：关注局部细节
大 rate：关注更大范围上下文
```

### 7.2 ASPP 的意义

语义分割中目标大小差异很大：

- 人可能很小
- 车可能中等
- 道路和天空可能很大

单一尺度卷积很难同时处理所有目标。

ASPP 的作用是：

> 让同一层特征同时看到不同尺度的上下文。

---

## 8. ASPP 代码实现

下面是一个简化 ASPP 模块，用于理解 DeepLabv2/v3 的核心结构。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = ASPPConv(in_channels, out_channels, rates[0])
        self.branch3 = ASPPConv(in_channels, out_channels, rates[1])
        self.branch4 = ASPPConv(in_channels, out_channels, rates[2])

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.project(x)


if __name__ == "__main__":
    aspp = ASPP(in_channels=2048, out_channels=256)
    x = torch.randn(2, 2048, 32, 32)
    y = aspp(x)
    print(y.shape)  # torch.Size([2, 256, 32, 32])
```

这段代码体现了 ASPP 的关键思想：

```text
同一输入特征 -> 多个不同 dilation rate 的分支 -> 拼接融合
```

---

## 9. DeepLabv3

DeepLabv3 对 ASPP 进行了更系统的改进。

主要变化：

1. 改进 ASPP 结构。
2. 加入 image-level feature。
3. 更重视 output stride 设置。
4. 弱化甚至去掉 CRF 后处理依赖。

---

## 10. Image-level Feature

DeepLabv3 中引入了图像级特征分支。

它的作用是获取全局上下文。

结构：

```text
Input Feature
  -> Global Average Pooling
  -> 1x1 Conv
  -> Upsample to feature size
  -> concat with ASPP branches
```

为什么需要这个分支？

ASPP 的不同 dilation rate 可以看到不同范围，但它们仍然是局部卷积。

图像级特征可以告诉模型：

```text
整张图大概是什么场景？
当前像素更可能属于哪些类别？
```

例如：

- 道路场景中更可能出现 car、person、road。
- 医学图像中更可能出现 organ、lesion、background。

---

## 11. DeepLabv3 ASPP 完整结构

DeepLabv3 的 ASPP 通常包括：

| 分支 | 作用 |
|------|------|
| 1x1 Conv | 保留局部原始信息 |
| 3x3 Atrous Conv rate 1 | 小感受野 |
| 3x3 Atrous Conv rate 2 | 中感受野 |
| 3x3 Atrous Conv rate 3 | 大感受野 |
| Image Pooling | 全局上下文 |

具体 rate 会根据 output stride 调整。

常见配置：

| Output Stride | ASPP Rates |
|---------------|------------|
| 16 | 6, 12, 18 |
| 8 | 12, 24, 36 |

---

## 12. DeepLabv3 ASPP 代码实现

下面加入 image-level pooling 分支，更接近 DeepLabv3。

```python
class ImagePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DeepLabV3ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = ASPPConv(in_channels, out_channels, rates[0])
        self.branch3 = ASPPConv(in_channels, out_channels, rates[1])
        self.branch4 = ASPPConv(in_channels, out_channels, rates[2])
        self.branch5 = ImagePooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        features = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x)
        ]
        x = torch.cat(features, dim=1)
        return self.project(x)
```

---

## 13. DeepLabv3 的整体结构

DeepLabv3 可以概括为：

```text
Input Image
  -> Backbone, such as ResNet / Xception
  -> Atrous Convolution controls output stride
  -> ASPP
  -> 1x1 Conv classifier
  -> Upsample to original image size
  -> Pixel prediction
```

与 FCN 相比：

```text
FCN: Backbone + simple upsampling
DeepLabv3: Backbone + atrous convolution + ASPP + global context
```

---

## 14. DeepLabv3+

DeepLabv3+ 是 DeepLab 系列中非常重要的版本。

它的核心改进是：

```text
DeepLabv3+ = DeepLabv3 Encoder + Lightweight Decoder
```

为什么需要 Decoder？

DeepLabv3 虽然有 ASPP，但最终仍然主要从高层特征直接上采样到原图，边界细节不够精细。

DeepLabv3+ 引入低层特征辅助恢复边界：

```text
高层特征：语义强，但分辨率低
低层特征：语义弱，但边界细节多
```

这和 U-Net 的 skip connection 思路类似，但 DeepLabv3+ 的 decoder 更轻量。

---

## 15. DeepLabv3+ 结构流程

DeepLabv3+ 的整体结构：

```text
Input Image
  |
  v
Backbone
  |----------------------|
  |                      |
  v                      v
Low-level Feature      High-level Feature
  |                      |
  |                      v
  |                    ASPP
  |                      |
  |                    Upsample
  |                      |
  v                      v
1x1 Conv            Concatenate
                         |
                         v
                    3x3 Conv x 2
                         |
                         v
                    Upsample
                         |
                         v
                    Segmentation Map
```

---

## 16. DeepLabv3+ Decoder 代码实现

下面是一个简化的 DeepLabv3+ Decoder。

```python
class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_channels, high_channels, num_classes):
        super().__init__()

        self.low_project = nn.Sequential(
            nn.Conv2d(low_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(high_channels + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, high_feature, low_feature):
        low_feature = self.low_project(low_feature)

        high_feature = F.interpolate(
            high_feature,
            size=low_feature.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([high_feature, low_feature], dim=1)
        x = self.fuse(x)
        return self.classifier(x)
```

这个 Decoder 的关键是：

```text
先压缩低层特征通道 -> 上采样高层语义特征 -> 拼接 -> 卷积融合 -> 分类
```

---

## 17. Atrous Separable Convolution

DeepLabv3+ 还使用了 **Atrous Separable Convolution**。

它可以理解为：

```text
Atrous Convolution + Depthwise Separable Convolution
```

普通卷积：

```text
同时做空间卷积和通道融合
```

深度可分离卷积：

```text
Depthwise Conv：每个通道单独做空间卷积
Pointwise Conv：用 1x1 卷积做通道融合
```

优点：

- 参数更少
- 计算量更低
- 适合大分辨率语义分割

---

## 18. 深度可分离空洞卷积代码

```python
class AtrousSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)
```

---

## 19. DeepLabv3+ 简化模型结构代码

下面是一个只用于理解结构的简化版本，不是完整论文复现。

```python
class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, low_channels, high_channels, num_classes):
        super().__init__()
        self.backbone = backbone
        self.aspp = DeepLabV3ASPP(high_channels, out_channels=256)
        self.decoder = DeepLabV3PlusDecoder(
            low_channels=low_channels,
            high_channels=256,
            num_classes=num_classes
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        low_feature, high_feature = self.backbone(x)
        high_feature = self.aspp(high_feature)
        logits = self.decoder(high_feature, low_feature)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )
        return logits
```

---

## 20. DeepLab 的训练方式

DeepLab 训练语义分割时，输入输出通常是：

```text
image: [B, 3, H, W]
mask:  [B, H, W]
logits: [B, num_classes, H, W]
```

常用损失函数：

```python
criterion = nn.CrossEntropyLoss(ignore_index=255)
loss = criterion(logits, mask)
```

其中：

- `mask` 每个像素是类别 ID。
- `ignore_index=255` 用于忽略未标注区域。
- 多类别语义分割不需要对输出先做 softmax，`CrossEntropyLoss` 内部会处理。

---

## 21. DeepLab 的推理流程

推理流程：

```text
输入图像
  -> resize / normalize
  -> Backbone
  -> ASPP
  -> Decoder / Upsampling
  -> logits
  -> argmax
  -> mask
```

代码示意：

```python
with torch.no_grad():
    logits = model(image)
    pred = logits.argmax(dim=1)
```

输出：

```text
pred: [B, H, W]
```

---

## 22. DeepLab 的数据格式

和 FCN 类似，DeepLab 需要图像和像素级 mask。

```text
images/
  0001.jpg
  0002.jpg

masks/
  0001.png
  0002.png
```

mask 中每个像素是类别 ID：

```text
0: background
1: person
2: car
...
255: ignore
```

注意：

- mask resize 必须使用 nearest neighbor。
- 图像 resize 可以使用 bilinear。
- 颜色可视化 mask 不一定等于训练 mask。
- 训练 mask 应该是类别 ID 图，而不是 RGB 彩色图。

---

## 23. DeepLab 的常见 Backbone

DeepLab 可以搭配不同 Backbone：

| Backbone | 特点 |
|----------|------|
| VGG | 早期版本使用较多 |
| ResNet | DeepLabv3 中常见 |
| Xception | DeepLabv3+ 中常见 |
| MobileNet | 移动端轻量化版本 |

Backbone 的作用：

```text
提取高层语义特征
```

DeepLab 的关键在于：

```text
如何在高层语义和空间分辨率之间取得平衡
```

---

## 24. DeepLab 与 FCN 的区别

| 对比项 | FCN | DeepLab |
|--------|-----|---------|
| 核心思想 | 全卷积 + 上采样 | 空洞卷积 + ASPP |
| 感受野 | 依赖深层 CNN | 空洞卷积显式扩大感受野 |
| 多尺度上下文 | 较弱 | ASPP 强化多尺度 |
| 边界优化 | skip + upsample | CRF 或 Decoder |
| 输出细节 | 较粗 | 更精细 |

简单理解：

```text
FCN 解决了 CNN 如何做像素预测的问题。
DeepLab 解决了如何在不损失太多分辨率的情况下看见更大上下文的问题。
```

---

## 25. DeepLab 与 U-Net 的区别

| 对比项 | U-Net | DeepLab |
|--------|-------|---------|
| 典型场景 | 医学图像、小数据 | 通用语义分割 |
| 结构重点 | 对称 Encoder-Decoder | 空洞卷积 + ASPP |
| Skip 方式 | 多层 encoder-decoder 拼接 | DeepLabv3+ 使用低层特征融合 |
| 多尺度上下文 | 原版较弱 | ASPP 很强 |
| 边界恢复 | Decoder 很强 | v3+ 引入轻量 Decoder |

---

## 26. DeepLab 的优点

| 优点 | 说明 |
|------|------|
| 感受野大 | 空洞卷积扩大感受野 |
| 保留分辨率 | 减少过度下采样 |
| 多尺度能力强 | ASPP 适应不同大小目标 |
| 结构可扩展 | 可搭配 ResNet、Xception、MobileNet |
| 精度较高 | 长期是语义分割强基线 |

---

## 27. DeepLab 的局限性

| 局限性 | 说明 |
|--------|------|
| 计算量较大 | 高分辨率特征图和 ASPP 都较耗算力 |
| 边界仍非完美 | 复杂边界和细长结构仍可能出错 |
| 对 Backbone 依赖强 | Backbone 决定很大一部分性能 |
| 空洞卷积可能有栅格效应 | dilation 设置不当会导致采样不连续 |
| 不具备开放词汇能力 | 仍是固定类别语义分割模型 |

---

## 28. 空洞卷积的栅格效应

空洞卷积虽然扩大了感受野，但如果 dilation rate 设置不合理，可能出现采样点过于稀疏的问题。

这被称为：

```text
gridding effect
```

直观理解：

```text
卷积核只看到了棋盘状的离散点，中间区域没有被充分利用。
```

ASPP 通过多个 dilation rate 并行，可以在一定程度上缓解这个问题。

---

## 29. DeepLab 系列演进总结

```text
DeepLabv1:
  Atrous Convolution + CRF

DeepLabv2:
  ASPP + CRF

DeepLabv3:
  Improved ASPP + Image-level Feature

DeepLabv3+:
  Encoder-Decoder + Atrous Separable Convolution
```

---

## 30. 学习 DeepLab 时最重要的几个问题

### 30.1 DeepLab 为什么不用普通卷积一直下采样？

因为语义分割需要像素级定位，过多下采样会导致边界和小目标丢失。

### 30.2 空洞卷积是不是增加参数量？

不是。空洞卷积增加的是采样间隔，不增加卷积核参数数量。

### 30.3 ASPP 为什么有效？

因为不同 dilation rate 对应不同感受野，可以同时捕获小目标和大目标上下文。

### 30.4 DeepLabv3+ 为什么要加 Decoder？

因为仅靠高层特征上采样，边界恢复不够精细。Decoder 引入低层细节特征后，边界质量更好。

### 30.5 DeepLab 现在还值得学吗？

值得。虽然现在有 SegFormer、Mask2Former、SAM 等新模型，但 DeepLab 的空洞卷积、ASPP、output stride 等概念仍然是理解语义分割的重要基础。

---

## 31. 总结

DeepLab 系列的核心可以概括为：

1. 用空洞卷积扩大感受野，同时保持较高特征图分辨率。
2. 用 ASPP 捕获多尺度上下文。
3. 早期版本用 CRF 细化边界。
4. DeepLabv3 引入 image-level feature，强化全局上下文。
5. DeepLabv3+ 加入 Encoder-Decoder 和 Atrous Separable Convolution。

DeepLab 的历史意义在于，它系统地解决了语义分割中的两个核心矛盾：

```text
语义理解需要大感受野
边界定位需要高分辨率
```

DeepLab 正是通过空洞卷积、ASPP 和 Decoder，在这两个目标之间取得平衡。

