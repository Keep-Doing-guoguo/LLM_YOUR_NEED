# LoRA 原理与微调讲解

LoRA 是 Low-Rank Adaptation 的缩写，是大模型微调中非常常用的参数高效微调方法。

一句话概括：

```text
LoRA 不直接更新原始大模型权重，而是在某些线性层旁边增加低秩可训练矩阵，只训练这部分小参数。
```

## 一、为什么需要 LoRA

一个 7B 模型有几十亿参数。如果全量微调，需要：

- 保存所有参数梯度；
- 保存优化器状态；
- 占用大量显存；
- 训练成本高；
- 每个任务都保存一份完整模型权重，存储成本高。

LoRA 的思路是：原始模型大部分能力已经存在，微调时只需要学习一个小的增量。

```text
原始权重 W 冻结
只训练一个增量 ΔW
最终使用 W + ΔW
```

## 二、LoRA 的核心公式

对于一个线性层：

```text
y = xW
```

LoRA 不直接更新 W，而是加入一个低秩增量：

```text
y = xW + xΔW
```

其中：

```text
ΔW = BA
```

假设原始权重：

```text
W: [in_features, out_features]
```

LoRA 使用两个小矩阵：

```text
A: [in_features, r]
B: [r, out_features]
```

r 是 rank，通常远小于 hidden size。

例如：

```text
in_features = 4096
out_features = 4096
r = 8
```

原始 W 参数量：

```text
4096 * 4096 = 16,777,216
```

LoRA 参数量：

```text
4096 * 8 + 8 * 4096 = 65,536
```

参数量大幅减少。

## 三、LoRA 的结构

可以把 LoRA 看成原始线性层旁边加了一个旁路分支：

```text
Input x
  -> 原始 Linear W，冻结
  -> LoRA A: 降维到 r
  -> LoRA B: 升维回输出维度
  -> 原始输出 + LoRA 输出
```

结构图：

```text
                 -> W frozen ---------
Input x --------                     + -> Output
                 -> A train -> B train
```

训练时：

```text
W 不更新
A、B 更新
```

推理时可以选择：

```text
方式 1: 保留 LoRA 分支
方式 2: 把 BA 合并进 W
```

合并后：

```text
W_merged = W + BA * scaling
```

## 四、LoRA 的 scaling

LoRA 通常会使用缩放系数：

```text
ΔW = BA * alpha / r
```

其中：

| 参数 | 含义 |
|---|---|
| r | LoRA rank，低秩矩阵的中间维度 |
| alpha | LoRA 缩放系数 |
| alpha / r | 实际缩放比例 |

如果 r 很小，LoRA 表达能力弱；如果 r 很大，参数量和过拟合风险增加。

常见设置：

```text
r = 8 / 16 / 32 / 64
alpha = 16 / 32 / 64 / 128
```

## 五、LoRA 通常加在哪里

在 Transformer 中，LoRA 常加在线性层上。

常见 target modules：

| 模块 | 说明 |
|---|---|
| q_proj | Attention 的 Query 投影 |
| k_proj | Attention 的 Key 投影 |
| v_proj | Attention 的 Value 投影 |
| o_proj | Attention 输出投影 |
| gate_proj | MLP 门控投影 |
| up_proj | MLP 升维投影 |
| down_proj | MLP 降维投影 |

很多训练会优先选择：

```text
q_proj, v_proj
```

如果想提升效果，可以覆盖更多层：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

覆盖模块越多，训练参数越多，效果可能更好，但显存和过拟合风险也更高。

## 六、LoRA 和全量微调的区别

| 对比项 | 全量微调 | LoRA |
|---|---|---|
| 原始模型参数 | 更新 | 冻结 |
| 训练参数量 | 很大 | 很小 |
| 显存占用 | 高 | 低 |
| 保存模型 | 保存完整权重 | 保存 adapter |
| 多任务切换 | 成本高 | 加载不同 LoRA adapter |
| 极限效果 | 通常更强 | 接近但不一定超过全量 |

## 七、LoRA、QLoRA、Adapter 的区别

### 1. LoRA

基础 LoRA 通常在 FP16/BF16 模型上训练低秩矩阵。

```text
Base Model: FP16 / BF16
Trainable: LoRA A/B
```

### 2. QLoRA

QLoRA 会把基础模型量化到 4bit，再训练 LoRA。

```text
Base Model: 4bit quantized
Trainable: LoRA A/B
Compute: BF16 / FP16
```

优点是显存更低，适合单卡微调较大模型。

### 3. Adapter

Adapter 通常是在 Transformer Block 中插入小型网络：

```text
Input
  -> 原始层
  -> Adapter 小模块
  -> Output
```

LoRA 是改造线性层权重增量，Adapter 是插入额外网络结构。

## 八、LoRA 训练流程

典型流程：

```text
1. 加载基础模型
2. 冻结基础模型参数
3. 指定 target_modules
4. 插入 LoRA 层
5. 准备训练数据
6. 只训练 LoRA 参数
7. 保存 adapter 权重
8. 推理时加载 base model + adapter
```

伪代码：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

## 九、LoRA 常见参数

| 参数 | 说明 | 常见值 |
|---|---|---|
| r | 低秩维度，控制 LoRA 表达能力 | 8、16、32、64 |
| lora_alpha | 缩放系数 | 16、32、64、128 |
| lora_dropout | LoRA 分支 dropout | 0、0.05、0.1 |
| target_modules | 插入 LoRA 的层名 | q_proj/v_proj 或更多 |
| bias | 是否训练 bias | none 常见 |
| task_type | 任务类型 | CAUSAL_LM |

## 十、LoRA 合并与不合并

### 1. 不合并

```text
Base Model + LoRA Adapter
```

优点：

- 多个任务可以切换不同 adapter；
- adapter 文件很小；
- 方便继续训练。

缺点：

- 推理框架需要支持 adapter；
- 可能有少量额外计算。

### 2. 合并

```text
W_merged = W + BA * alpha / r
```

优点：

- 推理时就是普通模型；
- 部署更简单。

缺点：

- 合并后不方便切换多个 adapter；
- 量化模型合并时要注意精度和格式。

## 十一、LoRA 常见问题

### 1. r 越大越好吗

不一定。r 越大表达能力越强，但参数更多，也更容易过拟合。小数据集上 r 太大可能反而变差。

### 2. target_modules 越多越好吗

不一定。覆盖更多模块会提高可训练参数量，效果可能提升，但训练更慢，也更容易学坏原模型行为。

### 3. LoRA 能学知识吗

可以学一部分任务知识和格式偏好，但如果要注入大量新知识，数据质量、训练轮数、学习率和模型容量都会成为瓶颈。LoRA 更适合风格适配、指令格式适配、领域任务适配。

### 4. LoRA 和 RAG 怎么选

如果知识经常变化，用 RAG 更合适；如果是固定任务格式、固定风格、固定领域行为，用 LoRA 更合适。

## 十二、总结

LoRA 的核心是：

```text
冻结原始模型 W
训练低秩增量 ΔW = BA
最终得到 W + ΔW
```

它的优势是：

- 显存低；
- 训练快；
- adapter 文件小；
- 多任务切换方便；
- 适合个人和中小团队做大模型微调。

最需要理解的参数是：

```text
r, lora_alpha, lora_dropout, target_modules
```

