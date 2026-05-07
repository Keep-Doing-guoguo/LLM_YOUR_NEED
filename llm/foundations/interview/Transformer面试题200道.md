# Transformer 面试题 200 道

本文整理 Transformer 相关面试题与参考答案，覆盖基础概念、Attention、位置编码、Encoder/Decoder、BERT/GPT、训练推理、长上下文、工程优化和常见追问。

## 一、基础概念

### 1. Transformer 是什么？

Transformer 是一种基于 Attention 机制的序列建模架构，最早用于机器翻译。它用 Self-Attention 建模 token 之间的依赖关系，避免了 RNN 按时间步串行计算的问题，后来成为 BERT、GPT、T5、LLaMA 等模型的基础。

### 2. Transformer 主要解决了 RNN 的什么问题？

主要解决三点：训练难以并行、长距离依赖容易衰减、序列越长计算越慢。Transformer 让每个 token 可以直接关注其他 token，因此并行性和长距离建模能力更强。

### 3. Transformer 的核心模块有哪些？

核心模块包括 Token Embedding、位置编码、Multi-Head Self-Attention、Feed Forward Network、残差连接、LayerNorm 或 RMSNorm，以及输出层。

### 4. Transformer Block 一般由什么组成？

经典 Block 包含 Self-Attention、Add & Norm、Feed Forward Network、Add & Norm。现代大模型常用 Pre-LN 结构，即 Norm 在 Attention 和 MLP 之前。

### 5. Self-Attention 的直观含义是什么？

Self-Attention 的含义是：序列中的每个 token 根据自己和其他 token 的相关性，从整个上下文中加权汇总信息，从而更新自己的表示。

### 6. Transformer 为什么能并行训练？

因为 Self-Attention 对一个序列中所有位置的 Q、K、V 可以一次性矩阵计算，不依赖前一个时间步的隐藏状态。训练时只需通过 mask 限制可见范围，不需要像 RNN 一样逐步递推。

### 7. Transformer 和 CNN 的区别是什么？

CNN 主要通过局部卷积核建模邻域关系，需要多层堆叠扩大感受野；Transformer 通过 Attention 直接建立任意位置之间的关系，天然有全局感受野，但计算复杂度通常随序列长度平方增长。

### 8. Transformer 和 RNN 的区别是什么？

RNN 按顺序更新隐藏状态，适合序列但并行性差；Transformer 一次处理整个序列，通过 Attention 建模关系，并行性强。RNN 的长依赖依赖状态传递，Transformer 的长依赖可以通过注意力直接连接。

### 9. Transformer 中 token 表示会怎样变化？

输入 token 先变成 embedding，再经过多层 Transformer Block。每一层都会根据上下文更新 token 表示，越高层通常包含越抽象、越任务相关的语义信息。

### 10. Transformer 原论文叫什么？

原论文是 2017 年提出的《Attention Is All You Need》。核心贡献是用完全基于 Attention 的架构替代 RNN/CNN 来做序列到序列建模。

## 二、输入表示与 Embedding

### 11. Transformer 的输入是什么？

模型输入通常是 token id 序列，形状为 `[batch_size, seq_len]`。经过 embedding 查表后变成 `[batch_size, seq_len, hidden_size]` 的连续向量。

### 12. Token Embedding 是什么？

Token Embedding 是一个可训练查表矩阵，每个 token id 对应一个向量。它把离散文本符号映射到神经网络可处理的连续空间。

### 13. 为什么需要位置编码？

Self-Attention 本身对输入顺序不敏感，如果没有位置信息，“猫追狗”和“狗追猫”可能很难区分。位置编码用于注入 token 的顺序和距离信息。

### 14. Token Embedding 和 Position Embedding 如何结合？

经典做法是两者逐元素相加，使每个位置的输入向量同时包含 token 语义和位置信息。也有模型使用 RoPE、ALiBi 等在 Attention 计算中注入位置关系。

### 15. Segment Embedding 是什么？

Segment Embedding 用于区分输入中的不同句子或片段，典型例子是 BERT 的句子 A 和句子 B。它帮助模型知道某个 token 属于哪个片段。

### 16. Embedding 矩阵的参数量怎么算？

参数量等于 `vocab_size * hidden_size`。例如词表 32000、hidden size 4096，则 embedding 参数量约为 1.31 亿。

### 17. 输入 embedding 为什么通常要乘以 sqrt(d_model)？

原始 Transformer 中这样做是为了调整 embedding 的数值尺度，使其与位置编码相加后更稳定。现代大模型不一定保留这个操作。

### 18. 什么是 tied embedding？

tied embedding 指输入 embedding 矩阵和输出 LM Head 权重共享。它可以减少参数量，并让输入 token 表示和输出分类空间保持一致。

### 19. 为什么大模型常用子词 tokenizer？

子词 tokenizer 能平衡词表大小和未知词问题。它可以表示常见词，也能把罕见词拆成子词，避免大量 OOV。

### 20. Tokenizer 会影响 Transformer 效果吗？

会。Tokenizer 决定文本被切分成什么 token，影响序列长度、语义颗粒度、多语言能力和推理成本。更短更合理的 tokenization 通常更利于训练和推理。

## 三、Q、K、V 与 Attention 计算

### 21. Q、K、V 分别是什么？

Q 是 Query，表示当前 token 想查询什么；K 是 Key，表示每个 token 可被匹配的特征；V 是 Value，表示真正被加权汇总的信息。

### 22. Attention 的公式是什么？

Scaled Dot-Product Attention 的公式是 `softmax(QK^T / sqrt(d_k))V`。先计算 Q 和 K 的相似度，再归一化成权重，最后对 V 做加权求和。

### 23. 为什么 QK^T 可以表示相关性？

点积可以衡量两个向量方向的一致程度。Q 与某个 K 点积越大，说明当前 token 与该位置越相关，softmax 后会分配更大的注意力权重。

### 24. 为什么 Attention 要除以 sqrt(d_k)？

维度越高，点积方差越大，softmax 容易进入饱和区，导致梯度不稳定。除以 `sqrt(d_k)` 可以控制分数尺度，使训练更稳定。

### 25. Attention 权重的形状是什么？

对于单头注意力，权重通常是 `[batch_size, seq_len, seq_len]`。多头时通常是 `[batch_size, num_heads, seq_len, seq_len]`。

### 26. Attention 输出的形状是什么？

单头输出通常是 `[batch_size, seq_len, head_dim]`。多头输出拼接后变成 `[batch_size, seq_len, hidden_size]`，再经过输出投影。

### 27. Softmax 在 Attention 中起什么作用？

Softmax 把相关性分数归一化成概率分布，使每个 query 对所有 key 的关注权重和为 1。这样模型可以进行加权平均。

### 28. Attention 中 mask 的作用是什么？

Mask 用于屏蔽不应该关注的位置。例如 padding mask 屏蔽无效 token，causal mask 屏蔽未来 token，防止自回归模型看到答案。

### 29. Attention 是不是一定表示可解释性？

不一定。Attention 权重可以提供一定参考，但它不等同于严格因果解释。模型输出还受到 Value、MLP、残差和多层变换影响。

### 30. Self-Attention 的时间复杂度是多少？

标准 Self-Attention 的时间复杂度是 `O(n^2 d)`，其中 `n` 是序列长度，`d` 是 hidden size。主要瓶颈来自 `QK^T` 的 `n*n` 注意力矩阵。

## 四、多头注意力

### 31. Multi-Head Attention 是什么？

Multi-Head Attention 是把 hidden 向量拆成多个 head，每个 head 独立做 Attention，最后拼接并线性投影。它允许模型在不同子空间学习不同关系。

### 32. 为什么需要多头注意力？

单头注意力表达能力有限，多头可以同时关注语法关系、指代关系、局部关系、长距离关系等不同模式，增强建模能力。

### 33. head_dim 通常怎么算？

`head_dim = hidden_size / num_heads`。例如 hidden size 是 4096，head 数是 32，则每个 head 的维度是 128。

### 34. 多头注意力会增加参数量吗？

在 hidden size 固定时，多头本身不一定显著增加 QKV 投影总参数量，因为总体投影维度仍是 hidden size。它主要改变计算的分组方式。

### 35. Attention head 越多越好吗？

不是。head 太少可能表达不足，太多会导致每个 head 维度过小、计算调度复杂，也可能出现冗余 head。需要结合模型规模和任务调参。

### 36. 多头注意力最后为什么要做输出投影？

输出投影用于融合不同 head 的信息，并把拼接后的表示映射回 hidden space。它让各个 head 的结果能重新交互。

### 37. 什么是 head redundancy？

head redundancy 指不同 attention head 学到的模式高度相似，存在冗余。一些研究发现部分 head 可剪枝而不明显影响性能。

### 38. 什么是 MHA？

MHA 是 Multi-Head Attention，标准多头注意力。每个 head 拥有独立的 Q、K、V 投影子空间。

### 39. 什么是 MQA？

MQA 是 Multi-Query Attention。它让多个 query head 共享同一组 K、V，从而显著减少 KV Cache，提升自回归推理速度和显存效率。

### 40. 什么是 GQA？

GQA 是 Grouped-Query Attention。它介于 MHA 和 MQA 之间，让一组 query heads 共享一组 K、V，在效果和推理效率之间折中。

## 五、Mask 机制

### 41. Padding Mask 是什么？

Padding Mask 用于屏蔽 batch 中补齐的 pad token，避免模型把无意义位置当作上下文信息。

### 42. Causal Mask 是什么？

Causal Mask 是自回归语言模型使用的下三角 mask，保证当前位置只能看到自己和之前的 token，不能看到未来 token。

### 43. Encoder 需要 causal mask 吗？

普通 Encoder 不需要 causal mask，因为它通常做双向编码，允许每个 token 看完整输入。它只需要 padding mask。

### 44. Decoder 为什么需要 causal mask？

Decoder 在生成任务中要按从左到右预测下一个 token。使用 causal mask 可以防止训练时泄漏未来答案，保持训练和推理一致。

### 45. Cross-Attention 需要什么 mask？

Cross-Attention 通常需要 encoder padding mask，用于屏蔽源序列中的 pad token。它不一定需要 causal mask，因为目标端的因果性由 decoder self-attention 保证。

### 46. Mask 一般加在 softmax 前还是后？

一般加在 softmax 前，把被屏蔽位置的分数设为很大的负数，使 softmax 后权重接近 0。

### 47. 为什么不能简单把 mask 后权重乘 0？

可以做但不如 softmax 前 mask 常见。softmax 前 mask 可以保证归一化时被屏蔽位置不参与概率分配，数值语义更清晰。

### 48. Attention mask 和 loss mask 区别是什么？

Attention mask 控制模型能看哪些 token；loss mask 控制哪些位置参与损失计算。例如 instruction tuning 中，prompt 可见但不一定计算 loss。

### 49. Prefix LM 的 mask 和 causal mask 有什么区别？

Prefix LM 允许前缀部分双向可见，生成部分仍保持因果可见。普通 causal mask 则所有位置只能看左侧。

### 50. 双向模型和自回归模型的 mask 区别是什么？

双向模型如 BERT 通常允许 token 看左右上下文；自回归模型如 GPT 只能看左侧上下文。两者 mask 决定了预训练目标和生成能力差异。

## 六、位置编码

### 51. 绝对位置编码是什么？

绝对位置编码为每个位置分配一个位置向量，并与 token embedding 相加。它告诉模型 token 位于第几个位置。

### 52. 正弦位置编码的特点是什么？

正弦位置编码是固定函数，不需要训练参数。它使用不同频率的 sin/cos 表示位置，理论上有一定外推能力。

### 53. 可学习位置编码的特点是什么？

可学习位置编码通过训练得到，灵活性强，但通常对训练长度范围依赖更明显，长度外推能力可能较弱。

### 54. 相对位置编码是什么？

相对位置编码关注 token 之间的相对距离，而不是绝对下标。它更符合很多语言关系依赖距离的特点。

### 55. RoPE 是什么？

RoPE 是 Rotary Position Embedding，通过旋转 Q、K 向量注入位置信息，使注意力分数天然包含相对位置信息。LLaMA、Qwen 等模型广泛使用它。

### 56. RoPE 为什么适合大模型？

RoPE 不需要单独位置 embedding 表，能把相对位置信息融入 QK 点积，效果稳定，且便于和 KV Cache 结合。

### 57. ALiBi 是什么？

ALiBi 是 Attention with Linear Biases，在 attention score 上加入与距离相关的线性偏置。距离越远惩罚越大，具备较好的长度外推能力。

### 58. 位置编码会影响长上下文能力吗？

会。位置编码决定模型如何理解位置和距离。绝对位置编码通常不利于外推，RoPE 需要缩放策略，ALiBi 对长距离外推较友好。

### 59. RoPE 外推为什么会出问题？

RoPE 的旋转频率在训练长度内学习较稳定，超出训练长度后位置角度分布可能偏离训练范围，导致模型对长距离位置关系理解变差。

### 60. RoPE Scaling 的目的是什么？

RoPE Scaling 通过调整位置频率或位置下标映射，让更长上下文的位置分布更接近训练时范围，从而扩展可用上下文长度。

## 七、Feed Forward Network 与激活函数

### 61. Transformer 中 FFN 的作用是什么？

FFN 对每个 token 独立进行非线性变换，提升表示能力。Attention 负责 token 间信息交互，FFN 负责对每个位置做特征加工。

### 62. FFN 的常见结构是什么？

经典结构是 `Linear(d_model, d_ff) -> Activation -> Linear(d_ff, d_model)`。其中 `d_ff` 通常是 `d_model` 的 4 倍左右。

### 63. 为什么 FFN 通常比 hidden size 更宽？

扩宽中间层可以增加非线性表达能力，让模型在更高维空间中处理特征，再压回 hidden size。

### 64. FFN 对不同 token 是否共享参数？

共享。FFN 对序列中每个位置使用同一组参数，相当于逐 token 的 MLP。

### 65. Transformer 中常见激活函数有哪些？

常见激活函数有 ReLU、GELU、SiLU、SwiGLU、GeGLU 等。现代大模型常用 SwiGLU 或 GeGLU。

### 66. GELU 相比 ReLU 有什么特点？

GELU 是平滑激活函数，会根据输入大小进行概率式门控，通常在 Transformer 中比 ReLU 表现更好。

### 67. SwiGLU 是什么？

SwiGLU 是一种门控 MLP 结构，通常形式类似 `SiLU(xW1) * xW2` 后再投影。它能增强表达能力，是 LLaMA 等模型常用结构。

### 68. 为什么现代大模型喜欢用门控 FFN？

门控 FFN 能动态控制信息通过，表达能力更强，在相近计算预算下常比普通 FFN 效果更好。

### 69. FFN 的参数量主要在哪里？

主要在两到三个线性层中。若使用普通 FFN，参数量约为 `2 * d_model * d_ff`；使用 SwiGLU 通常有三个投影矩阵。

### 70. Attention 和 FFN 哪个参数更多？

很多 Transformer 中 FFN 参数更多。Attention 通常约 `4 * d_model^2`，普通 FFN 若 `d_ff=4*d_model`，约 `8 * d_model^2`。

## 八、残差连接与归一化

### 71. 残差连接的作用是什么？

残差连接让子层学习输入的增量变化，缓解深层网络梯度消失，帮助信息和梯度跨层流动。

### 72. LayerNorm 的作用是什么？

LayerNorm 对每个样本的 hidden 维度做归一化，稳定激活分布，改善训练稳定性。

### 73. LayerNorm 和 BatchNorm 的区别是什么？

BatchNorm 按 batch 统计，依赖 batch size；LayerNorm 按单个样本的特征维度统计，更适合变长序列和小 batch 的 NLP 模型。

### 74. Pre-LN 和 Post-LN 有什么区别？

Post-LN 是子层后残差再归一化；Pre-LN 是先归一化再进子层，子层输出后直接残差相加。Pre-LN 在深层 Transformer 中更稳定。

### 75. 为什么现代大模型多用 Pre-LN？

Pre-LN 让残差路径更直接，梯度更容易从高层传到底层，深层训练更稳定。缺点是最终输出可能需要额外 norm。

### 76. RMSNorm 是什么？

RMSNorm 只使用均方根进行归一化，不减均值。它比 LayerNorm 更简单，计算更省，现代大模型中非常常见。

### 77. RMSNorm 和 LayerNorm 的核心区别是什么？

LayerNorm 做去均值和方差归一化；RMSNorm 只按均方根缩放，不做中心化。RMSNorm 计算更轻量。

### 78. 为什么归一化有助于训练稳定？

归一化可以控制激活尺度，减少层间分布漂移，使优化器更容易找到稳定更新方向。

### 79. 残差连接会不会导致数值越来越大？

可能会，所以通常结合归一化、合适初始化、学习率策略和残差缩放来稳定训练。

### 80. DeepNorm 是什么？

DeepNorm 是用于稳定深层 Transformer 训练的一类归一化和残差缩放方法，核心是调整残差分支比例，支持更深模型训练。

## 九、Encoder、Decoder 与 Seq2Seq

### 81. Transformer Encoder 的特点是什么？

Encoder 使用双向 Self-Attention，每个 token 可以看到完整输入序列，适合理解类任务，如分类、检索、序列标注。

### 82. Transformer Decoder 的特点是什么？

Decoder 使用 causal self-attention，每个 token 只能看到左侧上下文，适合自回归生成任务，如语言建模、对话、代码生成。

### 83. Encoder-Decoder 架构是什么？

Encoder 编码源序列，Decoder 根据已生成目标序列并通过 Cross-Attention 读取 Encoder 输出，逐步生成目标序列。

### 84. Cross-Attention 的 Q、K、V 来自哪里？

通常 Q 来自 Decoder 当前 hidden states，K 和 V 来自 Encoder 输出。这样目标端可以查询源端信息。

### 85. BERT 属于哪种架构？

BERT 是 Encoder-only 架构，使用双向上下文建模，主要用于理解任务。

### 86. GPT 属于哪种架构？

GPT 是 Decoder-only 架构，使用自回归语言建模，适合生成任务。

### 87. T5 属于哪种架构？

T5 是 Encoder-Decoder 架构，把各种 NLP 任务统一成 text-to-text 格式。

### 88. 为什么 GPT 不需要 Encoder？

GPT 的目标是根据左侧上下文预测下一个 token。Decoder-only 的 causal attention 已经足够完成自回归生成，不需要单独编码源序列。

### 89. 为什么机器翻译早期常用 Encoder-Decoder？

翻译需要先理解源语言句子，再生成目标语言句子。Encoder-Decoder 的 Cross-Attention 很适合从目标端动态读取源端表示。

### 90. Encoder-only 模型能直接生成文本吗？

不适合直接自回归生成。Encoder-only 模型通常用于理解任务，若要生成文本需要额外设计解码器或使用特殊生成方案。

## 十、BERT 相关

### 91. BERT 的预训练任务是什么？

BERT 原始预训练任务包括 MLM 和 NSP。MLM 是预测被 mask 的 token，NSP 是判断两句话是否连续。

### 92. MLM 是什么？

Masked Language Modeling 是随机遮住输入中的部分 token，让模型根据双向上下文预测被遮住的 token。

### 93. BERT 为什么适合理解任务？

BERT 使用双向 Attention，能同时利用左右上下文，适合分类、匹配、抽取、序列标注等理解任务。

### 94. BERT 为什么不适合直接做生成？

BERT 训练目标不是从左到右预测下一个 token，而是恢复 mask token。它没有天然的自回归生成机制。

### 95. NSP 的作用是什么？

NSP 用于训练模型判断两个句子是否是连续关系，服务于句间关系建模。但后续 RoBERTa 等模型发现 NSP 不一定必要。

### 96. BERT 的 [CLS] token 用来做什么？

`[CLS]` 是序列级表示位置，常用于分类任务。模型最后一层的 `[CLS]` hidden state 会接分类头。

### 97. BERT 的 [SEP] token 用来做什么？

`[SEP]` 用于分隔句子或片段，让模型知道输入边界。例如句子对任务中分隔句子 A 和句子 B。

### 98. RoBERTa 相比 BERT 改进了什么？

RoBERTa 去掉 NSP，使用更大数据、更长训练、更大 batch 和动态 mask，整体提升了 BERT 类模型效果。

### 99. ALBERT 的核心改进是什么？

ALBERT 使用参数共享和 embedding 分解来减少参数量，同时引入 SOP 替代 NSP。

### 100. DistilBERT 的核心思想是什么？

DistilBERT 使用知识蒸馏，把 BERT 的能力压缩到更小模型中，在较小参数量下保留较好性能。

## 十一、GPT 与自回归语言模型

### 101. GPT 的训练目标是什么？

GPT 使用 causal language modeling，根据前面的 token 预测下一个 token，最大化序列的自回归似然。

### 102. 自回归生成是什么意思？

自回归生成是每一步基于已经生成的 token 预测下一个 token，然后把新 token 加入上下文继续生成。

### 103. GPT 训练时为什么可以并行？

训练时整段目标序列已知，可以一次性计算所有位置的 logits，并用 causal mask 防止看到未来 token，因此不需要逐 token 训练。

### 104. GPT 推理时为什么不能完全并行？

推理时下一个 token 依赖前面已生成结果，必须逐步生成。虽然每步内部矩阵计算可并行，但 token 时间步之间是串行的。

### 105. LM Head 是什么？

LM Head 是把 hidden state 映射到词表维度的输出层，用于得到每个 token 的预测 logits。

### 106. logits 是什么？

logits 是 softmax 前的未归一化分数。语言模型输出每个词表 token 的 logits，再通过 softmax 得到概率。

### 107. 为什么训练时使用交叉熵损失？

语言建模本质是多分类问题，每个位置要预测正确的下一个 token。交叉熵适合衡量预测分布与真实 token 之间的差异。

### 108. GPT 中 prompt 的作用是什么？

Prompt 作为条件上下文影响后续 token 分布。模型不会真正“执行指令”，而是在给定上下文下生成概率最高或采样得到的延续。

### 109. 指令微调如何改变 GPT 行为？

指令微调用指令-回答数据训练模型，使模型学会遵循人类指令格式和回答风格，而不仅仅是续写普通文本。

### 110. ChatGPT 类模型为什么需要 RLHF 或偏好优化？

预训练模型学会语言分布，指令微调学会基本遵循指令，RLHF/DPO 等偏好优化进一步让输出更符合人类偏好、安全要求和对话体验。

## 十二、训练目标与优化

### 111. Teacher Forcing 是什么？

Teacher Forcing 指训练生成模型时，每一步输入真实历史 token，而不是模型自己生成的 token。这样训练更稳定、更高效。

### 112. Exposure Bias 是什么？

Exposure Bias 指训练时模型看到真实前缀，推理时看到自己生成的前缀，两者分布不一致，错误可能逐步累积。

### 113. Label Smoothing 是什么？

Label Smoothing 是把 one-hot 标签稍微平滑，不让正确类别概率目标为 1。它能缓解过度自信，提高泛化。

### 114. Transformer 常用优化器是什么？

常用 Adam 或 AdamW。AdamW 把权重衰减与梯度更新解耦，是训练 Transformer 和大模型的常见选择。

### 115. Warmup 的作用是什么？

训练初期参数和梯度不稳定，直接用大学习率容易发散。Warmup 先逐步升高学习率，再进入正常衰减阶段。

### 116. Transformer 原论文的学习率策略是什么？

原论文使用带 warmup 的 inverse square root decay，学习率先升高，达到 warmup 步数后按步数平方根倒数衰减。

### 117. Dropout 在 Transformer 中用在哪里？

Dropout 可用于 attention weight、attention output、FFN 中间层、残差分支和 embedding 后，用于缓解过拟合。

### 118. Weight Decay 的作用是什么？

Weight Decay 通过惩罚过大的权重来正则化模型，有助于泛化。大模型训练中通常不会对 bias 和 norm 参数做 weight decay。

### 119. Gradient Clipping 的作用是什么？

Gradient Clipping 限制梯度范数，防止梯度爆炸，提升训练稳定性。

### 120. 混合精度训练为什么有用？

混合精度使用 FP16/BF16 等低精度计算，减少显存占用并提升吞吐。通常配合 loss scaling 或 BF16 的更大指数范围保证稳定性。

## 十三、推理与解码

### 121. Greedy Search 是什么？

Greedy Search 每一步选择概率最高的 token。它简单高效，但容易产生重复、保守或局部最优输出。

### 122. Beam Search 是什么？

Beam Search 每步保留多个候选序列，选择总体分数较高的路径。它常用于翻译等确定性任务，但开放式生成中可能不够自然。

### 123. Top-k Sampling 是什么？

Top-k Sampling 每步只在概率最高的 k 个 token 中采样，过滤低概率 token，控制生成随机性。

### 124. Top-p Sampling 是什么？

Top-p 又叫 nucleus sampling，每步选择累积概率达到 p 的最小 token 集合并采样。它比固定 k 更自适应。

### 125. Temperature 的作用是什么？

Temperature 调整 logits 分布的平滑程度。温度低输出更确定，温度高输出更随机。

### 126. repetition penalty 是什么？

repetition penalty 通过惩罚已经出现过的 token，减少模型重复生成同一短语或句子。

### 127. length penalty 是什么？

length penalty 用于调整序列长度对得分的影响，常见于 beam search，避免模型偏向过短或过长输出。

### 128. 为什么生成会出现重复？

可能因为模型概率分布过尖、贪心解码陷入循环、训练数据有重复模式、上下文过长导致注意力退化，或缺少重复惩罚。

### 129. stop token 的作用是什么？

stop token 表示生成结束。模型生成到 EOS 或指定停止序列时，推理框架停止继续解码。

### 130. logits processor 是什么？

logits processor 是解码前对 logits 进行修改的组件，例如禁用某些 token、应用 repetition penalty、限制格式或做温度缩放。

## 十四、KV Cache 与推理优化

### 131. KV Cache 是什么？

KV Cache 是自回归推理时缓存历史 token 的 Key 和 Value，避免每步重复计算完整前缀的 K、V。

### 132. KV Cache 为什么能加速推理？

生成第 t 个 token 时，历史 token 的 K、V 不会变化。缓存它们后，每步只需计算新 token 的 Q、K、V，并与历史 KV 做 attention。

### 133. KV Cache 的显存复杂度是什么？

KV Cache 显存大致与 `batch_size * seq_len * num_layers * num_kv_heads * head_dim * 2` 成正比，其中 2 表示 K 和 V。

### 134. MQA/GQA 如何减少 KV Cache？

MHA 每个 query head 都有独立 K、V；MQA/GQA 让多个 query head 共享 K、V，因此 `num_kv_heads` 更少，显著降低缓存显存。

### 135. Prefill 和 Decode 阶段有什么区别？

Prefill 是处理 prompt 的阶段，可并行计算整个输入的 KV；Decode 是逐 token 生成阶段，每步追加一个 token，主要受 KV Cache 和内存带宽影响。

### 136. 为什么 decode 阶段常受内存带宽限制？

每生成一个 token 都要读取大量历史 KV Cache，而计算量相对较小，因此瓶颈常在显存带宽和缓存访问。

### 137. FlashAttention 是什么？

FlashAttention 是一种 IO-aware 的 attention 实现，通过分块计算减少 HBM 读写和避免显式保存完整 attention 矩阵，提高速度并降低显存。

### 138. FlashAttention 改变 Attention 数学结果吗？

理论上不改变标准 attention 的数学形式，只是更高效地计算。由于数值精度和实现细节，结果可能有极小浮点差异。

### 139. PagedAttention 是什么？

PagedAttention 用分页方式管理 KV Cache，减少内存碎片，提升多请求并发推理时的显存利用率。vLLM 中广泛使用这一思想。

### 140. Continuous Batching 是什么？

Continuous Batching 是推理服务中动态合并不同请求的 token 计算，允许新请求插入、完成请求移除，提高 GPU 利用率。

## 十五、复杂度与长上下文

### 141. 标准 Attention 为什么难以处理超长序列？

因为注意力矩阵大小是 `seq_len * seq_len`，序列长度翻倍，attention 计算和显存通常接近四倍增长。

### 142. 长上下文模型主要难点是什么？

难点包括 attention 复杂度高、KV Cache 显存大、位置编码外推、长距离信息检索困难、训练数据长度不足和评测不充分。

### 143. Sparse Attention 是什么？

Sparse Attention 只让 token 关注部分位置，例如局部窗口、全局 token 或稀疏模式，从而降低计算复杂度。

### 144. Sliding Window Attention 是什么？

Sliding Window Attention 让每个 token 只关注附近窗口内的 token，适合长文本局部依赖建模，但可能削弱远距离信息交互。

### 145. Global Attention token 有什么作用？

Global token 可以被所有位置关注，或关注所有位置，用于在稀疏 attention 中传递全局信息。

### 146. Longformer 的核心思想是什么？

Longformer 使用局部窗口注意力结合少量全局注意力，把复杂度从平方级降低到近似线性级，适合长文档建模。

### 147. BigBird 的核心思想是什么？

BigBird 使用局部、随机和全局注意力组合，在降低复杂度的同时保留较强表达能力，并提供一定理论保证。

### 148. Transformer-XL 解决了什么问题？

Transformer-XL 引入 segment-level recurrence 和相对位置编码，让模型能复用前一段隐藏状态，改善长依赖建模。

### 149. 长上下文是否等于长距离推理能力强？

不等于。能放入更长上下文只是容量条件，模型还需要在训练中学会检索、聚合和利用远距离信息。

### 150. Needle-in-a-Haystack 测试是什么？

它是在长上下文中插入一条关键信息，测试模型能否从大量无关文本中准确找回该信息，用于评估长上下文检索能力。

## 十六、参数量、显存与计算

### 151. Transformer 参数量主要来自哪里？

主要来自 embedding、Attention 的 QKV/O 投影、FFN/MLP 线性层和 LM Head。其中 MLP 往往占比较大。

### 152. 单层 MHA 参数量如何估算？

标准 MHA 中 Q、K、V、O 各是 `d_model * d_model`，合计约 `4 * d_model^2`，不计 bias。

### 153. 单层 FFN 参数量如何估算？

普通 FFN 约为 `2 * d_model * d_ff`。若 `d_ff=4*d_model`，则约 `8 * d_model^2`。

### 154. 训练显存主要由哪些部分组成？

训练显存包括模型参数、梯度、优化器状态、激活值、临时 buffer 和数据 batch。AdamW 的优化器状态通常占用很大。

### 155. 推理显存主要由哪些部分组成？

推理显存主要包括模型权重、KV Cache、临时计算 buffer 和 batch 输入输出。其中长上下文场景下 KV Cache 很关键。

### 156. 为什么训练比推理更耗显存？

训练需要保存激活用于反向传播，还要存梯度和优化器状态；推理通常只需要权重和缓存，不需要反向图。

### 157. Activation Checkpointing 是什么？

Activation Checkpointing 训练时不保存部分中间激活，反向传播时重新计算它们，以计算换显存。

### 158. ZeRO 优化解决什么问题？

ZeRO 把优化器状态、梯度和参数在数据并行进程间分片，降低单卡显存压力，支持更大模型训练。

### 159. Tensor Parallelism 是什么？

Tensor Parallelism 把单层矩阵计算切分到多张 GPU 上，例如按列或按行切分线性层，用于训练或推理大模型。

### 160. Pipeline Parallelism 是什么？

Pipeline Parallelism 把模型不同层放到不同 GPU 上，按流水线执行 micro-batches，适合模型层数很深、单卡放不下的情况。

## 十七、现代大模型结构变化

### 161. LLaMA 类模型相对原始 Transformer 有哪些变化？

常见变化包括 Decoder-only 架构、Pre-RMSNorm、RoPE、SwiGLU、无 bias 或少 bias、GQA/MQA、以及更适合大规模训练的数据和 tokenizer。

### 162. 为什么很多大模型使用 Decoder-only？

Decoder-only 结构简单统一，适合大规模自回归预训练，也能通过 prompt 和指令微调适配各种生成与理解任务。

### 163. 为什么现代模型常去掉 bias？

去掉 bias 可以减少参数和计算，简化实现。在大规模模型中，bias 的收益通常不明显。

### 164. 为什么现代大模型常用 RMSNorm 而不是 LayerNorm？

RMSNorm 计算更简单、速度更快、显存更省，并且在大模型训练中效果稳定。

### 165. 为什么现代大模型常用 SwiGLU？

SwiGLU 相比普通 ReLU/GELU FFN 表达能力更强，常在相近计算量下带来更好效果。

### 166. 为什么 GQA 在大模型中常见？

GQA 能显著减少 KV Cache，提高推理吞吐，同时相比 MQA 更能保留 MHA 的表达能力，是效果和效率的折中。

### 167. 什么是 MoE Transformer？

MoE Transformer 在 FFN 部分引入多个专家网络，每个 token 只路由到少数专家。它增加总参数量但控制每 token 计算量。

### 168. MoE 的优点是什么？

MoE 可以在相近计算成本下扩大模型容量，提高知识存储和任务适配能力，适合大规模训练。

### 169. MoE 的难点是什么？

难点包括路由负载均衡、专家并行通信、训练稳定性、推理部署复杂度和部分专家利用不足。

### 170. Dense 模型和 MoE 模型区别是什么？

Dense 模型每个 token 使用全部参数参与计算；MoE 模型每个 token 只激活部分专家参数，因此总参数大但激活参数较少。

## 十八、视觉 Transformer 与多模态

### 171. ViT 是什么？

ViT 是 Vision Transformer，把图像切成 patch，把每个 patch 当作 token 输入 Transformer，用 self-attention 建模图像 patch 间关系。

### 172. ViT 中 patch embedding 是什么？

Patch embedding 把图像 patch 展平后线性投影成 token 向量。实现上常用卷积核大小等于 patch size、stride 等于 patch size 的卷积。

### 173. ViT 为什么需要大量数据？

ViT 缺少 CNN 的局部归纳偏置，更依赖数据学习视觉结构。数据不足时可能不如 CNN 稳定。

### 174. Swin Transformer 的核心思想是什么？

Swin Transformer 使用窗口注意力和 shifted window，降低计算复杂度并实现跨窗口信息交互，适合视觉层级特征建模。

### 175. DETR 和 Transformer 有什么关系？

DETR 使用 CNN 提取图像特征，再用 Transformer Encoder-Decoder 和 object queries 直接预测目标集合，减少传统检测中的 anchor 和 NMS 依赖。

### 176. 多模态模型如何把图像接入 LLM？

常见做法是用视觉编码器提取图像特征，再通过 projector、Q-Former 或 resampler 映射到 LLM hidden space，作为视觉 token 输入语言模型。

### 177. CLIP 中 Transformer 的作用是什么？

CLIP 可使用视觉 Transformer 或文本 Transformer 分别编码图像和文本，通过对比学习对齐图文表示空间。

### 178. Q-Former 是什么？

Q-Former 是 BLIP-2 中的查询 Transformer，用少量 learnable queries 从视觉特征中提取与语言相关的信息，再接入语言模型。

### 179. 视觉 token 太多会带来什么问题？

视觉 token 多会增加 LLM 的上下文长度和 attention 成本，降低推理速度并占用更多显存。

### 180. 多模态模型为什么需要 projector？

视觉编码器输出维度和语义空间通常与 LLM 不一致，projector 用于把视觉特征映射到 LLM 可接受的 hidden space。

## 十九、常见变体与替代方案

### 181. Performer 的核心思想是什么？

Performer 用核方法近似 softmax attention，把标准 attention 的平方复杂度近似降到线性复杂度。

### 182. Linformer 的核心思想是什么？

Linformer 认为注意力矩阵低秩，可以对 K、V 在序列维度做投影，降低长序列 attention 复杂度。

### 183. Reformer 的核心思想是什么？

Reformer 使用 LSH Attention 和可逆残差层，降低长序列计算和显存开销。

### 184. RetNet 是什么？

RetNet 是一种结合并行训练和递归推理能力的序列建模架构，试图在 Transformer 表达能力和 RNN 推理效率之间折中。

### 185. RWKV 是什么？

RWKV 是结合 RNN 推理形式和 Transformer 训练方式的架构，目标是获得线性时间推理和较强语言建模能力。

### 186. Mamba 是什么？

Mamba 是基于选择性状态空间模型的序列模型，使用线性复杂度处理长序列，是 Transformer 的重要替代方向之一。

### 187. SSM 相比 Transformer 的优势是什么？

SSM 通常具有线性复杂度和更好的长序列效率，推理时缓存更小。但在通用大语言模型生态中，Transformer 仍更成熟。

### 188. Transformer 是否一定比 RNN 好？

不一定。Transformer 在大规模并行训练和通用建模上优势明显，但 RNN/SSM 在低延迟、流式、超长序列等场景可能更高效。

### 189. Attention 是否可以替代所有归纳偏置？

不能。Attention 表达灵活，但缺少任务特定归纳偏置时可能需要更多数据。视觉、语音等领域仍常结合卷积、局部窗口或层级结构。

### 190. 为什么 Transformer 仍是主流？

因为它可扩展性强、并行训练高效、软硬件生态成熟、任务迁移能力强，并且在大规模数据和参数下表现稳定。

## 二十、面试深挖题

### 191. 如果让你手写 Self-Attention，会注意哪些细节？

要注意输入形状、QKV 投影、head reshape、scale、mask 加到 softmax 前、dropout、与 V 相乘、head 拼接、输出投影，以及数值稳定性。

### 192. 为什么 Attention 的复杂度是平方级？

因为每个 query 都要和所有 key 计算相似度，长度为 n 时有 `n*n` 对 token 关系，因此 attention score 矩阵是平方大小。

### 193. 为什么 Transformer 层数越深不一定越好？

层数更深会增加表达能力，但也带来优化困难、过拟合、梯度不稳定、推理延迟上升和显存开销。需要配合归一化、初始化和数据规模。

### 194. Transformer 中信息混合发生在哪里？

token 间信息混合主要发生在 Attention；每个 token 内部特征变换主要发生在 FFN；跨层信息流动依赖残差连接。

### 195. 为什么说 Attention 是动态权重？

Attention 权重由当前输入的 Q、K 计算得到，不是固定卷积核。不同样本、不同位置会产生不同的权重分布。

### 196. 如果模型输出胡言乱语，可能和 Transformer 哪些部分有关？

可能与训练数据、解码策略、上下文长度、位置编码外推、attention mask 错误、tokenizer 不匹配、权重加载错误或数值精度问题有关。

### 197. 如何判断 Attention mask 写错了？

可检查训练 loss 是否异常低或异常高、生成是否泄漏未来信息、padding 是否被关注、attention 权重可视化是否异常，以及小样本单元测试是否符合预期。

### 198. Transformer 面试中最常要求推导的公式是什么？

最常见是 Scaled Dot-Product Attention：`Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`，以及多头注意力的 reshape、concat 和输出投影流程。

### 199. 如何向非技术人员解释 Transformer？

可以说 Transformer 是一种读文本的方法：它会让每个词同时查看句子里其他词，判断哪些词更重要，然后综合这些信息理解当前词的含义。

### 200. 学 Transformer 最重要的主线是什么？

主线是：输入如何表示、Attention 如何让 token 交互、位置如何注入、Block 如何堆叠、训练目标如何塑造能力、推理时如何逐 token 生成，以及工程上如何优化显存和速度。

