# LoRA 和 QLoRA 微调详解

LoRA 和 QLoRA 都属于参数高效微调方法。它们解决的核心问题是：大模型参数量很大，如果全量微调，显存、训练成本和保存成本都很高。

## 1. LoRA 是什么

LoRA，全称 Low-Rank Adaptation，中文可以理解为低秩适配微调。

它的核心思想是：

```text
冻结原始大模型参数，只训练额外插入的小矩阵。
```

原始全量微调是直接更新权重：

```text
W_new = W + ΔW
```

LoRA 不直接训练完整的 `ΔW`，而是把它拆成两个低秩矩阵：

```text
ΔW = B × A
```

其中：

| 符号 | 含义 |
|------|------|
| `W` | 原始模型权重，冻结不训练 |
| `A` | 降维矩阵，可训练 |
| `B` | 升维矩阵，可训练 |
| `r` | rank，低秩维度 |

## 2. 为什么 LoRA 省显存

假设原始线性层权重是：

```text
W: 4096 × 4096
```

全量微调需要训练：

```text
4096 × 4096 = 16777216 个参数
```

如果 LoRA rank `r=8`：

```text
A: 8 × 4096
B: 4096 × 8
参数量 = 8 × 4096 + 4096 × 8 = 65536
```

参数量下降非常明显。

## 3. LoRA 的优点和缺点

| 优点 | 说明 |
|------|------|
| 显存占用低 | 只训练 adapter |
| 训练速度快 | 反向传播参数少 |
| 保存文件小 | 只保存 LoRA 权重 |
| 多任务方便 | 不同任务保存不同 adapter |
| 部署灵活 | 可加载 adapter，也可 merge |

| 缺点 | 说明 |
|------|------|
| 能力上限有限 | 一般低于全量微调 |
| 依赖 target_modules | 插入位置选不好效果差 |
| adapter 管理复杂 | 多任务时版本要管理好 |
| 合并部署要注意 | QLoRA 合并尤其要谨慎 |

## 4. 项目中的相关文件

当前项目中 LoRA / QLoRA 相关文件：

| 文件 | 说明 |
|------|------|
| `llm/finetune/Qwen/finetune_lora_single_gpu.py` | Qwen 单卡 LoRA 微调 |
| `llm/finetune/Qwen/finetune_lora_ds.py` | Qwen + LoRA + DeepSpeed |
| `llm/finetune/Qwen/finetune_qlora_single_gpu.py` | Qwen 单卡 QLoRA |
| `llm/finetune/Qwen/finetune_qlora_ds.py` | Qwen + QLoRA + DeepSpeed |
| `llm/finetune/Qwen2-DPO/main_train.py` | 包含 LoRA / QLoRA 创建逻辑 |

## 5. LoRA 核心代码

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## 6. LoRA 重要参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `r` | 低秩矩阵 rank | 8、16、32、64 常见 |
| `lora_alpha` | 缩放系数 | 常见为 16、32、64 |
| `lora_dropout` | dropout | 数据少时可设 0.05 |
| `target_modules` | 插入 LoRA 的模块 | attention + MLP 线性层 |
| `bias` | 是否训练 bias | 通常设为 `none` |

`r` 越大，可训练参数越多，表达能力越强，但显存和过拟合风险也更高。

## 7. target_modules 怎么选

不同模型的线性层名字不同。

Qwen 老版本常见：

```python
["c_attn", "c_proj", "w1", "w2"]
```

LLaMA / Qwen2 / Qwen2.5 常见：

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

可以自动扫描线性层：

```python
import torch.nn as nn

def find_all_linear_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(name.split(".")[-1])
    names.discard("lm_head")
    return list(names)
```

项目里的 `find_all_linear_names()` 也是这个思路。

## 8. QLoRA 是什么

QLoRA 可以理解为：

```text
4bit 量化加载基座模型 + LoRA adapter 训练
```

它进一步节省显存，适合单卡训练较大模型。

QLoRA 的核心不是训练 4bit 权重，而是：

```text
基座模型 4bit 冻结；
LoRA adapter 正常训练。
```

## 9. QLoRA 核心代码

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

## 10. QLoRA 重要参数

| 参数 | 说明 |
|------|------|
| `load_in_4bit=True` | 4bit 加载基座模型 |
| `bnb_4bit_quant_type="nf4"` | 使用 NF4 量化 |
| `bnb_4bit_compute_dtype` | 计算时使用 fp16 或 bf16 |
| `bnb_4bit_use_double_quant=True` | 双重量化，进一步省显存 |

## 11. LoRA 和 QLoRA 区别

| 对比项 | LoRA | QLoRA |
|--------|------|-------|
| 基座模型精度 | fp16 / bf16 常见 | 4bit |
| 显存占用 | 低 | 更低 |
| 训练速度 | 通常更快 | 可能略慢 |
| 环境依赖 | PEFT 即可 | 依赖 bitsandbytes / CUDA |
| 适合场景 | 显存较充足 | 显存紧张 |

## 12. 权重保存和合并

LoRA 训练后默认保存的是 adapter，不是完整模型。

保存 adapter：

```python
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

合并 LoRA：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(merged_output_path)
```

## 13. 项目中常见问题

问题 1：为什么 LoRA 训练后模型文件很小？

答案：

因为保存的是 adapter 权重，不是完整基座模型。推理时需要 `base model + adapter`，或者先 merge 成完整模型。

问题 2：为什么 LoRA 微调效果不好？

答案：

常见原因有：训练数据质量差、chat template 不一致、labels mask 错误、学习率过大或过小、target_modules 没选对、训练步数不够。

问题 3：为什么 QLoRA 容易遇到环境问题？

答案：

QLoRA 依赖 bitsandbytes、CUDA、GPU 架构和 transformers/peft 版本。如果版本不匹配，容易出现 4bit 加载失败或训练报错。

问题 4：为什么不训练 `lm_head`？

答案：

`lm_head` 参数量较大，训练它会增加显存和保存成本。多数 SFT 场景只训练 Transformer 内部线性层即可。

问题 5：QLoRA 合并时要注意什么？

答案：

QLoRA 训练时基座是 4bit 加载，但合并时通常要用非量化基座重新加载，再加载 adapter 并 merge。

## 14. 面试题

问题 1：LoRA 的核心思想是什么？

答案：

冻结原模型权重，只训练插入到部分线性层中的低秩 adapter。它用两个小矩阵近似权重增量，从而大幅减少可训练参数。

问题 2：LoRA 的公式是什么？

答案：

```text
W_new = W + ΔW
ΔW = B × A
```

其中 W 是冻结的原始权重，A 和 B 是可训练低秩矩阵。

问题 3：`r` 和 `lora_alpha` 分别控制什么？

答案：

`r` 控制低秩矩阵的 rank，影响参数量和表达能力；`lora_alpha` 是缩放系数，影响 LoRA adapter 对原始输出的影响强度。

问题 4：LoRA 和全量微调有什么区别？

答案：

全量微调更新所有参数，成本高但能力上限高；LoRA 只训练 adapter，成本低、保存小、部署灵活，但能力上限受 rank 和插入位置影响。

问题 5：QLoRA 和 LoRA 有什么区别？

答案：

QLoRA 使用 4bit 量化加载基座模型，再训练 LoRA adapter，因此比 LoRA 更省显存，但对环境依赖更强，训练速度可能略慢。

问题 6：如何排查 LoRA 微调效果差？

答案：

先检查训练样本 decode 后的格式和 labels mask，再检查推理 prompt 是否和训练一致，然后检查 target_modules、学习率、训练步数、数据质量和是否过拟合。

