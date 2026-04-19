# FCN 详解：Fully Convolutional Networks for Semantic Segmentation

> 论文：Fully Convolutional Networks for Semantic Segmentation  
> 作者：Jonathan Long, Evan Shelhamer, Trevor Darrell  
> 会议：CVPR 2015  
> arXiv：https://arxiv.org/abs/1411.4038

---

## 1. FCN 是什么

FCN，全称 **Fully Convolutional Network**，中文通常称为**全卷积网络**。

它是深度学习语义分割中的经典起点之一。FCN 的核心思想是：

> 把原本用于图像分类的 CNN 改造成可以输出像素级预测结果的网络。

传统分类网络只输出一个类别：

```text
输入图像 -> CNN -> 分类向量
```

FCN 输出的是一张分割图：

```text
输入图像 -> FCN -> 每个像素的类别
```

也就是说，FCN 解决的是：

```text
每个像素属于哪个类别？
```

例如：

```text
图像中的每个像素 -> person / car / road / sky / background
```

---

## 2. FCN 解决了什么问题

在 FCN 之前，很多语义分割方法并不是端到端的深度网络，常见做法包括：

1. 使用滑动窗口逐块分类。
2. 先提取手工特征，再用分类器预测像素类别。
3. CNN 只作为局部 patch 分类器使用。

这些方法的问题是：

| 问题 | 说明 |
|------|------|
| 计算慢 | 每个像素或每个 patch 都要重复前向推理 |
| 不能端到端训练 | 特征提取、分类、后处理往往是分开的 |
| 空间信息利用不足 | 分类网络越深，特征图越小，定位能力下降 |
| 输出分辨率低 | CNN 下采样后很难恢复精细边界 |

FCN 的贡献是把分类 CNN 改造成密集预测模型，使网络可以一次性输出整张图的像素级预测。

---

## 3. 语义分割任务回顾

语义分割的输入和输出如下：

```text
输入：一张 H x W x 3 的 RGB 图像
输出：一张 H x W 的类别图
```

如果有 `C` 个类别，模型通常先输出：

```text
[B, C, H, W]
```

其中：

- `B`：batch size
- `C`：类别数量
- `H, W`：输出图像的高和宽

然后对每个像素取类别概率最大的通道：

```python
pred_mask = logits.argmax(dim=1)
```

最终得到：

```text
[B, H, W]
```

---

## 4. 从分类网络到 FCN

FCN 最关键的改造是：

> 将分类网络中的全连接层替换为卷积层。

以 VGG 为例，传统分类网络大致是：

```text
Input Image
  -> Conv Blocks
  -> Flatten
  -> Fully Connected
  -> Class Scores
```

FCN 改造成：

```text
Input Image
  -> Conv Blocks
  -> 1x1 Conv
  -> Class Score Map
  -> Upsampling
  -> Pixel-level Prediction
```

---

## 5. 为什么要去掉全连接层

全连接层有两个问题：

1. 固定输入尺寸。
2. 丢失空间结构。

例如分类网络最后通常会把特征图展平：

```text
[C, H, W] -> [C * H * W]
```

一旦展平，网络就不再保留像素和空间位置之间的对应关系。

FCN 用卷积层替代全连接层后，网络可以保留二维空间结构：

```text
[B, C, H, W] -> [B, num_classes, H, W]
```

这样每个空间位置都可以得到一个类别预测。

---

## 6. 全连接层如何等价为卷积层

假设分类网络最后有一个全连接层：

```text
输入特征图：7 x 7 x 512
输出类别数：4096
```

这个全连接层可以等价为：

```text
卷积核大小：7 x 7
输入通道：512
输出通道：4096
```

也就是：

```text
FC(7*7*512 -> 4096)
等价于
Conv2d(512 -> 4096, kernel_size=7)
```

再往后的全连接层可以替换为 `1x1 Conv`：

```text
FC(4096 -> 4096)
等价于
Conv2d(4096 -> 4096, kernel_size=1)
```

最后分类层：

```text
FC(4096 -> num_classes)
等价于
Conv2d(4096 -> num_classes, kernel_size=1)
```

这样分类网络就变成了全卷积网络。

---

## 7. FCN 的整体流程

FCN 的完整流程可以概括为：

```text
输入图像
  -> Backbone 提取特征
  -> 1x1 Conv 得到类别得分图
  -> 上采样恢复分辨率
  -> Softmax
  -> 每个像素分类
```

更具体地：

```text
Image [B, 3, H, W]
  |
  v
CNN Backbone
  |
  v
Feature Map [B, C, H/32, W/32]
  |
  v
1x1 Conv
  |
  v
Score Map [B, num_classes, H/32, W/32]
  |
  v
Upsampling
  |
  v
Segmentation Map [B, num_classes, H, W]
```

---

## 8. 为什么需要上采样

CNN Backbone 会不断下采样，例如：

```text
输入图像：512 x 512
经过多次 pooling / stride conv 后：
输出特征图：16 x 16
```

这时特征图分辨率太低，无法直接作为最终分割结果。

所以 FCN 需要把低分辨率 score map 放大回原图大小：

```text
16 x 16 -> 512 x 512
```

这个过程就是上采样。

---

## 9. FCN 中的上采样方式

FCN 论文中使用的是 **反卷积**，也就是现在常说的：

```text
Transposed Convolution
```

在 PyTorch 中对应：

```python
nn.ConvTranspose2d
```

简单示例：

```python
up = nn.ConvTranspose2d(
    in_channels=num_classes,
    out_channels=num_classes,
    kernel_size=64,
    stride=32,
    padding=16,
    bias=False
)
```

它可以把一个低分辨率 score map 放大 32 倍。

---

## 10. 反卷积和普通插值的区别

常见上采样方式有三类：

| 方法 | 是否可学习 | 说明 |
|------|------------|------|
| Nearest | 否 | 最近邻插值，速度快但粗糙 |
| Bilinear | 否 | 双线性插值，更平滑 |
| Transposed Conv | 是 | 参数可学习，FCN 使用这种方式 |

FCN 的转置卷积层可以初始化为双线性插值，然后在训练中继续学习。

---

## 11. FCN-32s

最基础的 FCN 版本称为 **FCN-32s**。

它使用最后一层特征图直接预测分割结果：

```text
Input
  -> CNN Backbone
  -> stride 32 feature map
  -> 1x1 Conv
  -> upsample x32
  -> output mask
```

例如：

```text
输入：512 x 512
最后特征图：16 x 16
上采样 32 倍：512 x 512
```

优点：

- 结构简单
- 语义信息强

缺点：

- 空间细节损失严重
- 边界粗糙
- 小目标容易丢失

---

## 12. FCN-16s

FCN-16s 在 FCN-32s 基础上加入了一次跳跃连接。

它将深层特征和较浅层特征融合：

```text
pool5 score -> upsample x2
pool4 score -> add
fused score -> upsample x16
```

结构示意：

```text
pool5: stride 32, strong semantics
  |
  | upsample x2
  v
pool4: stride 16, better spatial details
  |
  v
fuse
  |
  v
upsample x16
```

优点：

- 比 FCN-32s 边界更清晰
- 能恢复更多空间细节

---

## 13. FCN-8s

FCN-8s 进一步加入更浅层的 pool3 特征：

```text
pool5 score -> upsample x2
pool4 score -> add
fused score -> upsample x2
pool3 score -> add
final score -> upsample x8
```

结构示意：

```text
pool5: stride 32
  |
  v
pool4: stride 16
  |
  v
pool3: stride 8
  |
  v
final prediction
```

FCN-8s 比 FCN-16s 更精细，是 FCN 系列中效果较好的版本。

---

## 14. FCN-32s、FCN-16s、FCN-8s 对比

| 模型 | 使用特征层 | 上采样倍率 | 特点 |
|------|------------|------------|------|
| FCN-32s | pool5 | 32x | 语义强，边界粗 |
| FCN-16s | pool5 + pool4 | 16x | 加入中层细节 |
| FCN-8s | pool5 + pool4 + pool3 | 8x | 边界更精细 |

总结：

```text
FCN-32s：粗分割
FCN-16s：中等精度
FCN-8s：更细粒度分割
```

---

## 15. Skip Connection 的意义

FCN 的 skip connection 和 U-Net 的 skip connection 思想类似，但实现方式不同。

FCN 的 skip connection 通常是：

```text
score map 相加
```

U-Net 的 skip connection 通常是：

```text
feature map 拼接
```

对比：

| 模型 | 融合方式 | 说明 |
|------|----------|------|
| FCN | add | 把浅层 score map 和深层 score map 相加 |
| U-Net | concat | 把 encoder 特征和 decoder 特征沿通道拼接 |

FCN 的 skip connection 作用：

- 深层特征提供语义信息
- 浅层特征提供位置信息
- 融合后分割边界更准确

---

## 16. 1x1 卷积在 FCN 中的作用

FCN 中的 `1x1 Conv` 常用于把特征图转换为类别得分图。

假设 Backbone 输出：

```text
[B, 512, H/32, W/32]
```

类别数为 21，则：

```python
score = nn.Conv2d(512, 21, kernel_size=1)
```

输出：

```text
[B, 21, H/32, W/32]
```

含义：

```text
每个空间位置都有 21 个类别得分
```

---

## 17. FCN 的训练目标

FCN 训练时使用逐像素交叉熵损失。

模型输出：

```text
logits: [B, C, H, W]
```

标签：

```text
target: [B, H, W]
```

PyTorch 中可以直接使用：

```python
criterion = nn.CrossEntropyLoss(ignore_index=255)
loss = criterion(logits, target)
```

其中：

- `C` 是类别数量
- `target` 中每个像素是类别 ID
- `ignore_index=255` 常用于忽略未标注区域

---

## 18. FCN 的推理流程

推理流程：

```text
输入图像
  -> 图像预处理
  -> Backbone 提取特征
  -> 1x1 Conv 得到 score map
  -> 上采样到原图大小
  -> 对通道维度取 argmax
  -> 得到语义分割 mask
```

代码示意：

```python
with torch.no_grad():
    logits = model(image)          # [B, C, H, W]
    pred = logits.argmax(dim=1)    # [B, H, W]
```

---

## 19. FCN 的 PyTorch 简化实现

下面是一个用于教学的简化版本。它不是论文原版 VGG-FCN，但保留了 FCN 的核心思想：

- 全卷积结构
- `1x1 Conv` 输出类别得分
- 转置卷积上采样

```python
import torch
import torch.nn as nn


class SimpleFCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upsample = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=16,
            stride=8,
            padding=4
        )

    def forward(self, x):
        h = self.encoder(x)
        h = self.classifier(h)
        out = self.upsample(h)
        return out


if __name__ == "__main__":
    model = SimpleFCN(num_classes=21)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(y.shape)  # torch.Size([2, 21, 256, 256])
```

---

## 20. FCN-8s 简化实现思路

如果想实现 FCN-8s，需要保留多个尺度的中间特征。

伪代码：

```python
def forward(x):
    pool3 = block1_to_pool3(x)   # stride 8
    pool4 = block4(pool3)        # stride 16
    pool5 = block5(pool4)        # stride 32

    score5 = score_pool5(pool5)
    score4 = score_pool4(pool4)
    score3 = score_pool3(pool3)

    up5 = upsample_x2(score5)
    fuse4 = up5 + score4

    up4 = upsample_x2(fuse4)
    fuse3 = up4 + score3

    out = upsample_x8(fuse3)
    return out
```

关键点：

- `pool5` 语义最强，但空间最粗。
- `pool4` 提供中等尺度细节。
- `pool3` 提供更高分辨率细节。
- 最终输出比 FCN-32s 更精细。

---

## 21. FCN 的数据格式

语义分割数据通常包括：

```text
image: RGB image
mask: single-channel label image
```

例如：

```text
image shape: [H, W, 3]
mask shape:  [H, W]
```

mask 中每个像素是类别 ID：

```text
0: background
1: person
2: car
3: road
...
```

训练时转换为：

```text
image tensor: [3, H, W], float32
mask tensor:  [H, W], int64
```

注意：

- mask 不能像图像一样做双线性插值。
- resize mask 时应使用最近邻插值。
- mask 的像素值必须是类别 ID，而不是 RGB 颜色。

---

## 22. 数据增强注意事项

分割任务的数据增强必须同时作用于图像和 mask。

例如随机翻转：

```python
image = horizontal_flip(image)
mask = horizontal_flip(mask)
```

图像可以使用：

- 颜色扰动
- 模糊
- 亮度变化
- 归一化

mask 不应该使用：

- 颜色扰动
- 模糊
- 双线性插值

原因是 mask 是类别 ID 图，不是普通图像。

---

## 23. 常用评价指标

FCN 语义分割常用指标包括：

| 指标 | 含义 |
|------|------|
| Pixel Accuracy | 所有像素中预测正确的比例 |
| Mean Accuracy | 每个类别准确率的平均值 |
| IoU | 单个类别的交并比 |
| mIoU | 所有类别 IoU 的平均值 |
| Frequency Weighted IoU | 按类别出现频率加权的 IoU |

其中最常用的是：

```text
mIoU
```

IoU 定义：

```text
IoU = TP / (TP + FP + FN)
```

---

## 24. FCN 的优点

| 优点 | 说明 |
|------|------|
| 端到端训练 | 输入图像直接输出像素级预测 |
| 任意尺寸输入 | 全卷积结构不依赖固定输入尺寸 |
| 计算效率高 | 一次前向即可预测整张图 |
| 奠定分割基础 | 后续 U-Net、DeepLab 等都受其影响 |
| 支持预训练 Backbone | 可以从分类网络迁移参数 |

---

## 25. FCN 的局限性

| 局限 | 说明 |
|------|------|
| 边界较粗 | 上采样难以恢复精细边界 |
| 小目标困难 | 深层特征图分辨率太低 |
| 上下文建模不足 | 没有 ASPP、注意力、Transformer 等机制 |
| skip 融合较简单 | 主要是 score map 相加，表达能力有限 |
| 输出可能不够平滑 | 早期方法常结合 CRF 后处理 |

---

## 26. FCN 与 U-Net 的区别

| 对比项 | FCN | U-Net |
|--------|-----|-------|
| 提出时间 | 2015 | 2015 |
| 主要场景 | 通用语义分割 | 医学图像分割 |
| Backbone | 常基于 VGG 等分类网络 | Encoder-Decoder 对称结构 |
| Skip 方式 | score map add | feature map concat |
| Decoder | 较简单 | 更完整的逐级恢复路径 |
| 边界细节 | 相对粗糙 | 更精细 |

简单理解：

```text
FCN：把分类网络改成分割网络
U-Net：专门为少样本医学分割设计的 Encoder-Decoder 网络
```

---

## 27. FCN 与 DeepLab 的关系

DeepLab 可以看作是在 FCN 思路上的进一步发展。

FCN 的问题：

```text
下采样太多 -> 输出太粗
```

DeepLab 的改进：

```text
使用空洞卷积减少下采样带来的分辨率损失
使用 ASPP 获取多尺度上下文
早期版本结合 CRF 改善边界
```

对比：

| 模型 | 核心思想 |
|------|----------|
| FCN | 全卷积 + 上采样 |
| DeepLab | 空洞卷积 + 多尺度上下文 |
| U-Net | 编码器-解码器 + skip concat |

---

## 28. FCN 的历史意义

FCN 的历史意义很大：

1. 第一次系统证明分类 CNN 可以自然扩展到像素级预测。
2. 提出了全卷积化思想。
3. 推动了端到端语义分割。
4. 引入了可学习上采样。
5. 使用 skip 结构融合粗语义和细空间信息。

后续很多分割模型都可以看作是在 FCN 基础上改进：

```text
FCN
  -> U-Net
  -> SegNet
  -> DeepLab
  -> PSPNet
  -> HRNet
  -> SegFormer
  -> MaskFormer / Mask2Former
```

---

## 29. 学习 FCN 时最容易混淆的问题

### 29.1 FCN 是不是没有全连接层？

是。FCN 的核心是用卷积层替换全连接层，所以可以接受任意尺寸输入。

### 29.2 FCN 的输出为什么是低分辨率？

因为 Backbone 中有多次下采样，例如 pooling 或 stride convolution。

### 29.3 为什么要用 1x1 卷积？

用于把每个位置的特征向量映射成类别得分。

### 29.4 反卷积是不是卷积的逆运算？

不是严格数学意义上的逆运算。它是一种可学习的上采样方式。

### 29.5 FCN 和 U-Net 谁更强？

在很多需要精细边界的场景中，U-Net 更实用。但 FCN 是更基础、更通用的全卷积语义分割框架。

---

## 30. 总结

FCN 的核心可以概括为五句话：

1. 把分类 CNN 改造成全卷积网络。
2. 用 `1x1 Conv` 输出每个位置的类别得分。
3. 用转置卷积把低分辨率 score map 上采样回原图大小。
4. 用 skip connection 融合浅层细节和深层语义。
5. 使用逐像素交叉熵进行端到端训练。

FCN 虽然现在已经不是最先进的分割模型，但它是理解 U-Net、DeepLab、SegFormer、MaskFormer 等模型的重要基础。

