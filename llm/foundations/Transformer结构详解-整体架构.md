# Transformer 结构详解：整体架构

Transformer 是现代大模型的基础结构。它最早用于机器翻译，后来逐渐成为 BERT、GPT、T5、LLaMA、Qwen、ChatGLM 等模型的核心骨架。

理解 Transformer 时，不要一开始就陷入公式。可以先把它看成一个“把 token 序列不断变成更有上下文信息的 token 表示”的网络。

```text
输入文本
  -> 分词 Tokenizer
  -> Token ID
  -> Embedding
  -> 多层 Transformer Block
  -> 输出每个位置的 hidden state
  -> 语言模型头 LM Head
  -> 预测下一个 token
```

## 一、Transformer 要解决什么问题

传统 RNN 按顺序处理文本：

```text
我 -> 喜欢 -> 机器 -> 学习
```

这种方式天然适合序列，但有两个问题：

- 不能很好并行，训练慢；
- 长距离依赖容易衰减；
- 后面的词要等前面的词处理完。

Transformer 的思路是：每个 token 都可以直接和其他 token 建立关系。

例如句子：

```text
我 喜欢 用 Transformer 做 大模型
```

在 Self-Attention 中，“大模型”这个 token 可以直接关注 “Transformer”“做”“喜欢”等 token，而不需要像 RNN 那样一步一步传递信息。

## 二、Transformer 的基本组成

一个标准 Transformer 主要包含：

```text
Token Embedding
Position Encoding / Position Embedding
Transformer Block x N
Output Layer
```

其中最重要的是 Transformer Block。

一个 Block 通常长这样：

```text
Input
  -> Self-Attention
  -> Add & Norm
  -> Feed Forward Network
  -> Add & Norm
  -> Output
```

现在的大模型里经常使用 Pre-LN 结构，也就是先 Norm 再进入子模块：

```text
Input
  -> RMSNorm / LayerNorm
  -> Self-Attention
  -> Add
  -> RMSNorm / LayerNorm
  -> MLP
  -> Add
  -> Output
```

## 三、Embedding 层

文本不能直接输入神经网络，需要先分词并转成数字。

```text
"我喜欢机器学习"
  -> Tokenizer
  -> [1001, 258, 3345, 998]
```

每个 token id 会查表得到一个向量：

```text
Token ID
  -> Embedding Table
  -> Token Vector
```

假设 hidden size 是 4096，那么每个 token 会被表示成一个 4096 维向量。

```text
input_ids: [batch, seq_len]
embedding: [batch, seq_len, hidden_size]
```

## 四、位置编码

Self-Attention 本身不理解顺序。

下面两个句子如果不加位置信息，模型很难区分：

```text
猫 追 狗
狗 追 猫
```

所以需要位置编码告诉模型每个 token 在哪里。

常见位置编码方式：

| 类型 | 说明 | 常见模型 |
|---|---|---|
| Sinusoidal Position Encoding | 原始 Transformer 使用的正弦余弦位置编码 | 早期 Transformer |
| Learned Position Embedding | 位置向量可训练 | GPT-2、BERT 等 |
| RoPE | 旋转位置编码，适合长上下文扩展 | LLaMA、Qwen 等 |
| ALiBi | 通过 attention bias 表示距离 | 一些长文本模型 |

现在很多大模型使用 RoPE。

## 五、Encoder、Decoder、Decoder-only

Transformer 原始论文中有 Encoder 和 Decoder 两部分：

```text
Source Sentence
  -> Encoder
  -> Context Representation
  -> Decoder
  -> Target Sentence
```

不同模型会选择不同结构。

### 1. Encoder-only

典型代表：BERT。

```text
Input
  -> Transformer Encoder Blocks
  -> 每个 token 的上下文表示
```

特点：

- 可以双向看上下文；
- 适合理解任务；
- 常用于分类、匹配、抽取、embedding。

例如：

```text
今天 [MASK] 很好
```

BERT 可以同时看 `[MASK]` 左右两边的信息。

### 2. Decoder-only

典型代表：GPT、LLaMA、Qwen。

```text
Input Tokens
  -> Transformer Decoder Blocks
  -> LM Head
  -> Next Token Probability
```

特点：

- 只能看当前位置之前的 token；
- 适合文本生成；
- 当前主流大语言模型大多是 Decoder-only。

例如生成：

```text
北京 是 中国 的
```

模型下一步预测：

```text
首都
```

### 3. Encoder-Decoder

典型代表：T5、原始 Transformer。

```text
Input
  -> Encoder
  -> Decoder
  -> Output
```

特点：

- Encoder 负责理解输入；
- Decoder 负责生成输出；
- 适合翻译、摘要、改写等输入输出都较复杂的任务。

## 六、Self-Attention 在整体结构中的位置

Self-Attention 是 Transformer 的核心。

它的目标是：让每个 token 根据上下文重新更新自己的表示。

```text
原始 token 表示:
  我, 喜欢, 机器, 学习

经过 Self-Attention 后:
  我(结合全句信息)
  喜欢(结合主语和宾语)
  机器(结合学习)
  学习(结合机器)
```

Self-Attention 不改变 token 数量，只改变每个 token 的向量内容。

```text
[batch, seq_len, hidden_size]
  -> Self-Attention
  -> [batch, seq_len, hidden_size]
```

## 七、MLP / FFN 层

Attention 负责 token 之间的信息交互，MLP 负责对每个 token 自己的向量做非线性变换。

```text
每个 token 的 hidden state
  -> Linear up projection
  -> Activation
  -> Linear down projection
```

常见结构：

```text
x
  -> Linear(hidden_size, intermediate_size)
  -> GELU / SiLU / SwiGLU
  -> Linear(intermediate_size, hidden_size)
```

现代大模型常用 SwiGLU：

```text
MLP(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

MLP 的参数量通常非常大，有时比 Attention 层更多。

## 八、残差连接和归一化

Transformer 每个子模块外面都有残差连接：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

残差连接的作用：

- 保留原始信息；
- 缓解深层网络训练困难；
- 让梯度更容易反向传播。

归一化层的作用：

- 稳定训练；
- 控制激活值范围；
- 降低梯度爆炸或梯度消失风险。

常见归一化：

| 名称 | 说明 |
|---|---|
| LayerNorm | 原始 Transformer 常用 |
| RMSNorm | 现代大模型常用，计算更简单 |

## 九、LM Head

对于语言模型，最后需要预测词表中每个 token 的概率。

```text
hidden state: [batch, seq_len, hidden_size]
  -> LM Head Linear(hidden_size, vocab_size)
  -> logits: [batch, seq_len, vocab_size]
```

如果词表大小是 150000，那么每个位置都会输出 150000 个分数。

训练时通常用当前位置预测下一个 token：

```text
输入:  我 喜欢 机器
目标:  喜欢 机器 学习
```

## 十、整体前向流程

以 Decoder-only 大模型为例：

```text
文本
  -> Tokenizer
  -> input_ids
  -> Token Embedding
  -> 加入位置信息
  -> Transformer Block 1
  -> Transformer Block 2
  -> ...
  -> Transformer Block N
  -> Final Norm
  -> LM Head
  -> logits
  -> 采样下一个 token
```

## 十一、总结

Transformer 的核心可以概括为：

| 模块 | 作用 |
|---|---|
| Embedding | 把 token id 转成向量 |
| Position Encoding | 注入顺序信息 |
| Self-Attention | 建立 token 之间的关系 |
| MLP / FFN | 对每个 token 做非线性特征变换 |
| Residual | 保留信息，稳定深层训练 |
| Norm | 稳定数值分布 |
| LM Head | 把 hidden state 转成词表概率 |

学习 Transformer 的顺序建议是：

```text
整体结构
  -> Self-Attention
  -> Multi-Head Attention
  -> Mask
  -> MLP
  -> 残差和归一化
  -> 训练和推理
```

