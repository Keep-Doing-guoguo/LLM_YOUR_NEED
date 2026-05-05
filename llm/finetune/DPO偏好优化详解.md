# DPO 偏好优化详解

DPO，全称 Direct Preference Optimization，中文可以叫直接偏好优化。它是一种用于大模型对齐的训练方法，常用于 SFT 之后，让模型更偏向人类喜欢的回答。

## 1. DPO 是什么

DPO 的目标是：

```text
同一个 prompt 下，让模型更倾向 chosen 回答，远离 rejected 回答。
```

数据通常长这样：

```json
{
  "prompt": "请解释什么是 LoRA",
  "chosen": "LoRA 是一种参数高效微调方法，只训练低秩 adapter。",
  "rejected": "LoRA 是一种数据库。"
}
```

## 2. DPO 解决什么问题

SFT 只告诉模型：

```text
这个问题应该这样回答。
```

DPO 告诉模型：

```text
这两个回答里，chosen 比 rejected 更好。
```

所以 DPO 更适合优化：

- 回答风格；
- 安全性；
- 拒答边界；
- 事实严谨性；
- 结构化格式；
- 人类偏好。

## 3. DPO 和 SFT 的区别

| 对比项 | SFT | DPO |
|--------|-----|-----|
| 数据格式 | prompt + answer | prompt + chosen + rejected |
| 训练目标 | 模仿标准答案 | 偏向更优答案 |
| 是否需要负样本 | 不需要 | 需要 |
| 是否需要 Reward Model | 不需要 | 不需要 |
| 常见阶段 | 第一阶段 | SFT 之后 |

## 4. DPO 和 PPO 的区别

| 对比项 | DPO | PPO |
|--------|-----|-----|
| 是否需要奖励模型 | 不需要 | 通常需要 |
| 是否需要在线生成 | 通常不需要 | 需要 |
| 工程复杂度 | 较低 | 较高 |
| 稳定性 | 相对更稳定 | 更难调 |
| 显存成本 | 中等 | 高 |
| 落地难度 | 较低 | 较高 |

## 5. DPO 为什么不需要 Reward Model

传统 RLHF 通常是：

```text
SFT 模型
  -> 训练 Reward Model
  -> 使用 PPO 优化 Policy Model
```

DPO 直接使用 chosen / rejected 偏好对构造损失函数。

也就是说：

```text
DPO 把奖励建模和策略优化合并到一个监督式目标里。
```

## 6. DPO 训练流程

```text
准备 SFT 模型
  -> 准备偏好数据 prompt / chosen / rejected
  -> 加载 policy model
  -> 加载 reference model
  -> 计算 chosen / rejected logprob
  -> 计算 DPO loss
  -> 更新 policy model
  -> 保存 LoRA adapter 或完整模型
```

## 7. 项目中的相关文件

当前项目里 DPO 相关文件：

| 文件 | 说明 |
|------|------|
| `llm/finetune/Qwen2-DPO/main_train.py` | Qwen2 DPO / SFT 训练主脚本 |
| `llm/finetune/Qwen2-DPO/train_args/` | 训练参数 |
| `llm/finetune/Qwen2-DPO/utils/data_process.py` | 数据处理 |
| `llm/finetune/Qwen2-DPO/utils/data_collator.py` | data collator |
| `llm/finetune/Qwen2-DPO/utils/eval/` | 评估相关逻辑 |

## 8. DPO 核心代码示例

```python
from trl import DPOTrainer, DPOConfig

dpo_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    beta=0.1,
    num_train_epochs=3,
    bf16=True,
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    args=dpo_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## 9. beta 参数

`beta` 是 DPO 中非常重要的参数。

它控制 policy model 偏离 reference model 的程度。

| beta | 效果 |
|------|------|
| 较大 | 约束更强，模型更保守 |
| 较小 | 模型更容易偏向偏好数据，但更可能跑偏 |

经验：

```text
beta=0.1 是常见起点。
如果模型风格变化过猛，可以增大 beta。
如果偏好学习不明显，可以适当减小 beta。
```

## 10. DPO 数据构造

一个高质量 DPO 样本应该满足：

- prompt 清晰；
- chosen 确实优于 rejected；
- chosen 和 rejected 不只是长度不同；
- rejected 要有代表性错误；
- chosen 不要带明显模板污染；
- 数据要覆盖真实业务场景。

错误示例：

```json
{
  "prompt": "介绍 RAG",
  "chosen": "RAG 是检索增强生成。",
  "rejected": "RAG 是检索增强生成。"
}
```

这类样本没有偏好差异，训练价值很低。

## 11. DPO 项目常见问题

问题 1：DPO 显存为什么比 SFT 高？

答案：

DPO 要处理 prompt、chosen、rejected，并且通常还要计算 policy model 和 reference model 的 logprob，所以显存比 SFT 更高。

解决方式：

- 使用 LoRA / QLoRA；
- 减小 batch size；
- 增加 gradient accumulation；
- 开启 gradient checkpointing；
- 使用 ZeRO；
- 减小 max length。

问题 2：为什么 DPO 后模型回答变短？

答案：

如果 chosen 普遍比 rejected 短，模型会学到“短回答更好”的偏好。

解决方式：

- 清洗偏好数据；
- 控制 chosen / rejected 长度分布；
- 单独统计输出长度；
- 加入更完整的 chosen 样本。

问题 3：为什么 DPO loss 不下降？

答案：

可能原因有：偏好数据质量差、chosen/rejected 差异不明显、学习率过低、beta 不合适、数据格式错误、chat template 不一致。

问题 4：为什么 DPO 后模型跑偏？

答案：

可能是 beta 太小、偏好数据分布太窄、训练步数过多、学习率过大。可以增大 beta、降低学习率、减少 epoch、增加通用样本评估。

问题 5：DPO 是否一定要在 SFT 后做？

答案：

通常建议在 SFT 后做。SFT 先让模型学会基本指令跟随和任务格式，DPO 再做偏好对齐。如果直接 DPO，训练可能不稳定。

## 12. DPO 面试题

问题 1：DPO 是什么？

答案：

DPO 是直接偏好优化方法，使用 prompt、chosen、rejected 偏好对训练模型，让模型提高 chosen 的概率，降低 rejected 的概率，从而学习人类偏好。

问题 2：DPO 为什么不需要奖励模型？

答案：

DPO 直接从偏好对中构造优化目标，把奖励建模和策略优化合并到一个损失函数里，因此不需要额外训练 Reward Model。

问题 3：DPO 中 reference model 的作用是什么？

答案：

reference model 提供稳定基准，限制 policy model 不要过度偏离 SFT 模型。DPO 会比较 policy 和 reference 在 chosen/rejected 上的 logprob 差异。

问题 4：DPO 的 beta 如何理解？

答案：

beta 控制偏离 reference model 的程度。beta 越大，模型越保守；beta 越小，模型越容易学习偏好，但也更容易跑偏。

问题 5：DPO 和 PPO 的最大区别是什么？

答案：

DPO 不需要奖励模型，也不需要复杂的在线 RL 采样；PPO 通常需要 reward model、reference model、value model，训练链路更复杂。

问题 6：DPO 数据质量有什么要求？

答案：

chosen 必须明显优于 rejected，差异应该来自质量、事实性、安全性、格式等，而不是单纯长度差异。否则模型会学到错误偏好。

## 13. 项目经验回答

问题：你在项目中如何做 DPO？

答案：

我会先用 SFT 得到一个基础可用模型，然后收集真实业务中的偏好数据。每条数据包含 prompt、chosen、rejected。训练时加载 policy model 和 reference model，用 DPO loss 优化 policy。如果显存紧张，就用 LoRA 或 QLoRA，并减小 batch size、开启 gradient checkpointing。训练后不仅看 loss，还会抽样比较回答质量、长度、格式、安全性和业务指标，避免模型只学到表面偏好。

