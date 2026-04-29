# Q-Former 结构详解

Q-Former 是 BLIP-2 中最关键的桥接模块之一。它的全称通常写作：

```text
Querying Transformer
```

它要解决的问题很直接：

```text
视觉编码器输出了很多图像 patch 特征
语言模型只希望接收少量、精炼、可读的视觉信息
```

所以 Q-Former 的作用不是替代视觉编码器，也不是替代大语言模型，而是充当中间桥梁：

```text
Frozen Vision Encoder
  -> Q-Former
  -> Frozen LLM
```

![BLIP-2 Q-Former](./assets/blip2-qformer.svg)

## 一、为什么需要 Q-Former

视觉编码器例如 ViT 会输出很多 patch token。

假设一张图经过 ViT 后得到：

```text
[N, D]
```

其中：

- `N` 是 patch token 数量；
- `D` 是视觉特征维度。

如果直接把这些视觉 token 全部喂给 LLM，会有几个问题：

- token 太多，计算成本高；
- 视觉维度和 LLM hidden size 不一定一致；
- 不是所有 patch 都同等重要；
- 很难直接把大量视觉 patch 接入冻结的语言模型。

Q-Former 的核心思想是：

```text
不用让 LLM 看全部 patch
而是先用少量可学习 query 从图像特征中“查询”出最重要的信息
```

## 二、Q-Former 的整体结构

可以把 Q-Former 理解成一个带有 query token 的 Transformer。

整体流程：

```text
Image
  -> Frozen Vision Encoder
  -> Patch Features

Learnable Query Tokens
  -> Q-Former Self-Attention
  -> Cross-Attention to Patch Features
  -> Query Outputs
  -> Linear Projection
  -> Frozen LLM
```

它和普通 Transformer 的最大区别是：

```text
输入里有一组可学习 query tokens
这些 query 不是来自文本，而是模型自己学出来的
```

## 三、Q-Former 里到底有什么

Q-Former 内部通常可以理解为三部分：

### 1. Learnable Query Tokens

这是一组可学习参数，例如：

```text
Q = 32
```

表示有 32 个 query token。

这些 query token 不是句子里的单词，而是一组“视觉信息探针”。

可以把它们理解成：

```text
query_1 想找图中的主体
query_2 想找颜色信息
query_3 想找局部区域
query_4 想找动作或关系
...
```

当然实际训练时不会显式指定每个 query 的含义，而是让模型自己学。

### 2. Self-Attention

query tokens 之间会先做 self-attention：

```text
queries
  -> self-attention
  -> 彼此交换信息
```

这样不同 query 之间可以协同工作，而不是完全独立地读取图像。

### 3. Cross-Attention

随后 query tokens 会通过 cross-attention 去读取视觉编码器输出的 patch 特征：

```text
Query = learnable queries
Key, Value = image patch features
```

这一步最关键：

```text
query 去“查询”图像特征
把大量 patch 信息压缩成少量 query 输出
```

## 四、Q-Former 的结构示意

一个简化版 Q-Former Block 可以写成：

```text
Query Tokens
  -> Self-Attention
  -> Cross-Attention to Image Features
  -> MLP
  -> Output Query Tokens
```

如果堆叠多层：

```text
Input Queries
  -> Block 1
  -> Block 2
  -> Block 3
  -> ...
  -> Final Query Representations
```

最终输出仍然是固定数量的 query token：

```text
[Q, Dq]
```

而不是原来那一大堆 patch token。

## 五、输入输出分别是什么

### 1. 输入

Q-Former 的输入主要有两部分：

#### 输入 A：图像 patch 特征

来自冻结视觉编码器，例如 ViT：

```text
image_features = [N, Dv]
```

#### 输入 B：learnable queries

```text
query_tokens = [Q, Dq]
```

其中：

- `Q` 是 query 数量，例如 32；
- `Dq` 是 Q-Former 内部 hidden size。

### 2. 输出

输出是被图像信息更新后的 query 表示：

```text
query_outputs = [Q, Dq]
```

然后通常再经过线性映射，变成语言模型可接收的维度：

```text
[Q, Dq] -> [Q, hidden_size_of_LLM]
```

再送入 LLM 作为 prefix 或视觉 token。

## 六、Q-Former 和普通 Projector 的区别

Q-Former 经常和 LLaVA 里的 MLP Projector 对比。

### 1. Projector 路线

LLaVA 常见做法：

```text
image features
  -> MLP projector
  -> visual tokens
  -> LLM
```

特点：

- 结构简单；
- 参数少；
- 训练方便；
- 工程实现直接。

### 2. Q-Former 路线

BLIP-2 做法：

```text
image features
  -> learnable queries
  -> self-attention + cross-attention
  -> compressed query outputs
  -> LLM
```

特点：

- 能主动从图像中提取关键视觉信息；
- 能把大量 patch 压缩成较少 query；
- 更适合在冻结视觉模型和冻结 LLM 之间做桥接；
- 结构更复杂，训练成本高于简单 projector。

### 3. 两者对比

| 方式 | 代表模型 | 核心思路 | 优点 | 缺点 |
|---|---|---|---|---|
| MLP Projector | LLaVA | 线性/MLP 映射视觉特征到 LLM | 简单、直接、易训 | 压缩和选择能力有限 |
| Q-Former | BLIP-2 | 用 query 通过 cross-attention 提取关键视觉信息 | 压缩强、桥接能力强 | 结构更复杂 |

## 七、Q-Former 为什么适合冻结式训练

BLIP-2 的重要设计是：

```text
Vision Encoder 冻结
LLM 冻结
主要训练 Q-Former
```

这样做的原因是：

- 训练成本低；
- 不需要从头训练整个大模型；
- 能复用已有强大的视觉模型和语言模型；
- 参数高效。

Q-Former 在这里像一个翻译层：

```text
把视觉编码器的输出翻译成语言模型能接受的表示
```

## 八、Q-Former 的信息压缩本质

Q-Former 并不是让每个 patch 都原样进入 LLM，而是做信息提炼。

例如一张图里有 300 个 patch token：

```text
300 image patches
  -> Q-Former
  -> 32 query outputs
```

这就相当于：

```text
从大量局部视觉特征中抽取一组摘要
```

它的本质更接近：

- 信息压缩；
- 跨模态选择性读取；
- 视觉摘要构建。

## 九、Q-Former 在 BLIP-2 中的训练目标

Q-Former 在 BLIP-2 里不是孤立训练的，而是配合图文任务一起训练。

常见训练目标包括：

- 图文对齐；
- 图文匹配；
- 图像描述生成；
- 下游视觉语言任务。

这些任务共同促使 query tokens 学会：

```text
哪些视觉信息值得保留
哪些视觉信息对文本生成最重要
```

## 十、Q-Former 的局限

Q-Former 很强，但也不是万能的。

### 1. Query 数量有限

如果 query 太少，可能压缩过度，损失细节。

### 2. 高分辨率细节可能丢失

对于 OCR、文档、细粒度定位任务，如果图像细节很多，少量 query 不一定能完整保留所有信息。

### 3. 结构比 projector 更复杂

训练、实现和调参都更复杂。

### 4. 更偏“摘要式视觉输入”

Q-Former 很适合把图像摘要成 LLM 可用表示，但对需要大量细节保留的场景，后续模型经常会采用动态分辨率、更多视觉 token 或更复杂视觉适配器。

## 十一、一句话理解 Q-Former

可以把 Q-Former 理解为：

```text
一组可学习 query token
通过 cross-attention 去读取图像 patch 特征
再把图像压缩成少量、精炼、可输入 LLM 的视觉表示
```

## 十二、Q-Former 在多模态发展中的位置

Q-Former 的意义在于它代表了一条很重要的多模态路线：

```text
不是从头联合训练超大视觉语言模型
而是复用已有视觉模型和已有语言模型
用一个中间桥接模块把二者连接起来
```

这条路线对后续很多多模态模型设计都有启发。

从学习顺序上，理解 Q-Former 后，再看：

- BLIP-2；
- LLaVA 的 projector；
- Qwen-VL 的 visual adapter；
- Flamingo 的 cross-attention；

会更容易看出这些模型的区别。

