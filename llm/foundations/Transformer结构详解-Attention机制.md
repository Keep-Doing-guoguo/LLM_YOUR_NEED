# Transformer 结构详解：Attention 机制

Attention 是 Transformer 的核心。它解决的问题是：当前 token 应该从上下文中的哪些 token 获取信息，以及获取多少信息。

一句话概括：

```text
Attention = 用 Query 去查 Key，再根据相似度加权汇总 Value
```

## 一、为什么需要 Attention

看这个句子：

```text
我 把 苹果 放进 冰箱 因为 它 很 新鲜
```

这里的“它”更可能指“苹果”，不是“冰箱”。模型要理解这个关系，就需要让“它”这个 token 去关注前面的“苹果”。

Attention 做的事情就是：

```text
当前 token
  -> 和其他 token 计算相关性
  -> 找到更重要的上下文
  -> 汇总上下文信息
  -> 更新当前 token 表示
```

## 二、Q、K、V 是什么

Self-Attention 会把每个 token 的向量投影成三个向量：

```text
x -> Wq -> Q
x -> Wk -> K
x -> Wv -> V
```

可以用一个检索系统来类比：

| 名称 | 类比 | 作用 |
|---|---|---|
| Query | 查询条件 | 当前 token 想找什么信息 |
| Key | 索引标签 | 每个 token 能被匹配的特征 |
| Value | 实际内容 | 被取出来汇总的信息 |

如果 Query 和某个 Key 很相似，说明当前 token 应该更多关注那个 token 对应的 Value。

## 三、Attention 的计算流程

输入：

```text
X: [batch, seq_len, hidden_size]
```

经过线性层得到：

```text
Q = XWq
K = XWk
V = XWv
```

然后计算注意力分数：

```text
score = QK^T / sqrt(d_k)
```

再做 softmax：

```text
weight = softmax(score)
```

最后加权汇总：

```text
output = weight * V
```

完整流程：

```text
Input X
  -> Linear Q, K, V
  -> QK^T
  -> scale
  -> mask, if needed
  -> softmax
  -> multiply V
  -> output
```

## 四、为什么要除以 sqrt(d_k)

当向量维度很大时，Q 和 K 点积的数值可能变得很大。

如果直接做 softmax，可能出现：

```text
[1, 2, 50] -> softmax -> 几乎全压到 50 上
```

这样梯度会不稳定。

所以使用缩放：

```text
QK^T / sqrt(d_k)
```

目的是控制分数范围，让 softmax 更稳定。

## 五、Self-Attention 和 Cross-Attention

### 1. Self-Attention

Q、K、V 都来自同一个序列：

```text
X -> Q
X -> K
X -> V
```

用于让同一段文本内部的 token 互相交互。

Decoder-only 大模型主要使用 Self-Attention。

### 2. Cross-Attention

Q 来自一个序列，K/V 来自另一个序列：

```text
Decoder hidden state -> Q
Encoder output       -> K, V
```

常见于 Encoder-Decoder 结构，例如机器翻译：

```text
源语言句子 -> Encoder
目标语言生成 -> Decoder 通过 Cross-Attention 读取 Encoder 信息
```

## 六、Causal Mask

GPT 这类 Decoder-only 模型在生成时不能偷看未来 token。

例如训练样本：

```text
我 喜欢 机器 学习
```

当模型预测“机器”时，只能看到：

```text
我 喜欢
```

不能看到后面的“学习”。

所以需要 Causal Mask：

```text
位置 1 只能看 1
位置 2 只能看 1,2
位置 3 只能看 1,2,3
位置 4 只能看 1,2,3,4
```

矩阵形式：

```text
可见:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

被 mask 的位置会被加上一个很大的负数，使 softmax 后接近 0。

## 七、Padding Mask

一个 batch 中句子长度可能不同：

```text
句子1: 我 喜欢 机器 学习
句子2: 你好 [PAD] [PAD]
```

`[PAD]` 只是补齐用的，不应该参与 attention。

Padding Mask 的作用是让模型忽略 padding token。

## 八、Multi-Head Attention

单个 attention 头只能从一种表示空间里学习关系。Multi-Head Attention 会把 hidden size 分成多个 head，每个 head 独立计算 attention。

```text
Input
  -> Head 1 Attention
  -> Head 2 Attention
  -> Head 3 Attention
  -> ...
  -> Concat
  -> Output Projection
```

例如：

```text
hidden_size = 4096
num_heads = 32
head_dim = 4096 / 32 = 128
```

每个 head 可能关注不同关系：

- 有的关注语法结构；
- 有的关注指代关系；
- 有的关注局部相邻 token；
- 有的关注长距离依赖。

## 九、MHA、MQA、GQA

现代大模型为了提升推理效率，对 Attention 做了很多变体。

### 1. MHA

Multi-Head Attention，每个 head 都有自己的 Q、K、V。

```text
Q heads: 多个
K heads: 多个
V heads: 多个
```

优点是表达能力强，缺点是 KV Cache 大。

### 2. MQA

Multi-Query Attention，多个 Q head 共享同一组 K、V。

```text
Q heads: 多个
K heads: 1 个
V heads: 1 个
```

优点是推理时 KV Cache 更小，速度更快。缺点是表达能力可能受影响。

### 3. GQA

Grouped-Query Attention，把多个 Q head 分成若干组，每组共享 K、V。

```text
Q heads: 多个
K/V heads: 少于 Q heads
```

GQA 是 MHA 和 MQA 的折中，很多现代大模型采用这种方式。

## 十、KV Cache

自回归生成时，模型一次生成一个 token：

```text
第 1 步生成 token1
第 2 步生成 token2
第 3 步生成 token3
```

如果每一步都重新计算历史 token 的 K、V，会非常浪费。

KV Cache 会把历史 token 的 K、V 保存下来：

```text
历史 K/V
  + 当前 token 的 K/V
  -> 当前 token 做 attention
```

这样每次只需要计算新 token 的 Q、K、V，并复用历史 K、V。

KV Cache 是大模型推理加速的关键，但它会占用显存。上下文越长，batch 越大，KV Cache 越大。

## 十一、Attention 的复杂度

普通 Attention 的主要计算量来自：

```text
QK^T: [seq_len, hidden] x [hidden, seq_len]
```

复杂度约为：

```text
O(seq_len^2)
```

这意味着上下文长度翻倍，attention 相关计算和显存开销可能接近变成 4 倍。

所以长上下文模型经常使用：

- FlashAttention；
- Sliding Window Attention；
- Sparse Attention；
- GQA / MQA；
- KV Cache 量化；
- RoPE 扩展方法。

## 十二、Attention 和 MLP 的分工

可以这样理解：

| 模块 | 主要作用 |
|---|---|
| Attention | token 之间交换信息 |
| MLP | 每个 token 内部做特征变换 |

Attention 更像“查资料和汇总上下文”，MLP 更像“对当前信息做加工和推理”。

一个 Transformer Block 中，两者配合工作：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

## 十三、总结

Attention 的核心流程是：

```text
Q = XWq
K = XWk
V = XWv
score = QK^T / sqrt(d_k)
weight = softmax(score + mask)
output = weight V
```

最需要掌握的概念：

| 概念 | 解释 |
|---|---|
| Q | 当前 token 的查询向量 |
| K | 每个 token 的匹配向量 |
| V | 每个 token 的内容向量 |
| Mask | 控制哪些位置可见 |
| Multi-Head | 多个注意力头并行学习不同关系 |
| KV Cache | 推理时缓存历史 K/V，避免重复计算 |
| GQA/MQA | 减少 K/V 数量，降低推理显存和带宽压力 |

