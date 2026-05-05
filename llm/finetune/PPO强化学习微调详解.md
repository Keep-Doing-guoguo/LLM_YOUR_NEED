# PPO 强化学习微调详解

PPO，全称 Proximal Policy Optimization，是 RLHF 中经典的强化学习优化算法。它用于根据奖励模型或奖励函数进一步优化语言模型。

## 1. PPO 是什么

PPO 的目标是：

```text
让模型生成能获得更高奖励的回答，同时不要偏离原始模型太远。
```

它属于强化学习方法。

在大模型对齐中，PPO 常见于 RLHF：

```text
SFT -> Reward Model -> PPO
```

## 2. PPO 解决什么问题

SFT 只能模仿标准答案。

DPO 可以学习偏好对。

PPO 可以直接优化奖励：

```text
回答越符合奖励模型或奖励函数，模型越倾向生成这种回答。
```

适合：

- 奖励函数明确的任务；
- 有 Reward Model 的 RLHF；
- 需要在线采样优化的任务；
- 复杂偏好对齐；
- 需要探索不同回答的任务。

## 3. PPO 中的几个模型

| 角色 | 说明 |
|------|------|
| Policy Model | 当前要训练的模型 |
| Reference Model | 参考模型，用于 KL 约束 |
| Reward Model | 给模型回答打分 |
| Value Model / Critic | 估计价值函数，辅助 PPO 更新 |

## 4. PPO 训练流程

```text
输入 prompt
  -> Policy Model 生成回答
  -> Reward Model 给回答打分
  -> 计算 reward
  -> 计算和 Reference Model 的 KL penalty
  -> Value Model 估计 advantage
  -> PPO clip 更新 Policy Model
```

## 5. PPO 和 DPO 的区别

| 对比项 | DPO | PPO |
|--------|-----|-----|
| 是否需要 Reward Model | 不需要 | 通常需要 |
| 是否在线生成 | 通常不需要 | 需要 |
| 是否需要 Value Model | 不需要 | 通常需要 |
| 工程复杂度 | 较低 | 高 |
| 超参敏感度 | 中等 | 高 |
| 训练稳定性 | 相对稳定 | 更难调 |
| 适合场景 | 偏好对齐 | 复杂奖励优化 |

## 6. PPO 核心代码示意

不同 TRL 版本 API 会变化，下面是概念性示例：

```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=16,
    mini_batch_size=4,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=reference_model,
    tokenizer=tokenizer,
)

for batch in dataloader:
    queries = batch["input_ids"]

    responses = ppo_trainer.generate(
        queries,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    rewards = reward_model_score(queries, responses)

    stats = ppo_trainer.step(
        queries,
        responses,
        rewards,
    )

    ppo_trainer.log_stats(stats, batch, rewards)
```

## 7. KL penalty

PPO 中通常会加入 KL penalty。

它的作用是：

```text
防止当前模型为了追求 reward 过度偏离 reference model。
```

如果没有 KL 约束，模型可能出现：

- 语言质量下降；
- 输出奇怪模式；
- reward hacking；
- 安全性下降；
- 过度迎合奖励模型漏洞。

## 8. PPO 关键超参

| 参数 | 说明 |
|------|------|
| `learning_rate` | 学习率 |
| `batch_size` | rollout batch 大小 |
| `mini_batch_size` | PPO 内部小批次 |
| `kl_coef` | KL 惩罚系数 |
| `clip_range` | PPO 裁剪范围 |
| `max_new_tokens` | 生成长度 |
| `temperature` | 采样随机性 |
| `gamma` | 折扣因子，文本任务中不一定显式使用 |

## 9. Reward Model

Reward Model 的作用是给回答打分。

输入：

```text
prompt + response
```

输出：

```text
reward score
```

Reward Model 通常来自偏好数据训练。

如果 Reward Model 质量差，PPO 会学坏。

## 10. Reward Hacking

Reward Hacking 指模型学会钻奖励函数漏洞。

例子：

```text
奖励函数只检查是否包含“谢谢”，模型就每句话都加“谢谢”。
```

解决方式：

- 设计多维奖励；
- 加入人工评估；
- 加 KL 约束；
- 定期抽样检查；
- 不只看 reward 曲线；
- 使用真实业务指标验证。

## 11. PPO 项目常见问题

问题 1：为什么 PPO 比 DPO 难？

答案：

PPO 要在线生成、打分、计算 advantage、做 KL 约束和 policy 更新，涉及 policy、reference、reward、value 等多个模型或模块，工程链路和超参都更复杂。

问题 2：为什么 PPO 训练不稳定？

答案：

可能是 reward scale 不合适、学习率过大、KL 系数不合理、Reward Model 不稳定、生成采样参数不合适或 batch 太小。

问题 3：为什么 reward 升高但人工看效果变差？

答案：

模型可能 reward hacking 了。它优化了奖励函数漏洞，但没有真正提升业务质量。必须结合人工样本和业务指标评估。

问题 4：为什么需要 reference model？

答案：

reference model 用来计算 KL penalty，限制 policy model 不要偏离原始模型太远，保持语言能力和安全边界。

问题 5：PPO 什么时候值得用？

答案：

当有可靠 Reward Model 或明确可计算奖励，并且任务需要在线采样探索时，PPO 才值得用。如果只是偏好对齐，DPO 通常更简单。

## 12. PPO 面试题

问题 1：PPO 是什么？

答案：

PPO 是一种强化学习策略优化算法，通过裁剪策略更新幅度，让模型在提升奖励的同时保持训练稳定。在 RLHF 中，它用于优化语言模型生成更高奖励的回答。

问题 2：PPO 在 RLHF 中的流程是什么？

答案：

先用 SFT 得到初始模型，再训练 Reward Model，然后让 Policy Model 生成回答，Reward Model 打分，结合 KL penalty 和 value 估计，用 PPO 更新 Policy Model。

问题 3：PPO 为什么要 clip？

答案：

clip 是为了限制每次策略更新幅度，防止模型因为一次过大的更新导致训练崩溃或性能急剧下降。

问题 4：KL penalty 的作用是什么？

答案：

KL penalty 限制当前模型和参考模型的输出分布差异，防止模型为了追求 reward 过度偏离原模型。

问题 5：PPO 和 DPO 哪个更容易落地？

答案：

DPO 更容易落地，因为它不需要训练 Reward Model，也不需要复杂在线 RL 采样。PPO 更灵活，但工程复杂度和调参成本更高。

问题 6：PPO 最大风险是什么？

答案：

最大风险是 reward hacking。模型可能学会利用奖励模型漏洞，让 reward 升高但真实回答质量下降。

## 13. 项目经验回答

问题：如果项目中要做 PPO，你会怎么设计？

答案：

我会先确保 SFT 模型质量足够，再准备偏好数据训练 Reward Model。PPO 阶段会保留 reference model 做 KL 约束，控制 policy 不要偏离太远。训练过程中不仅监控 reward，还要监控 KL、输出长度、人工样本质量和业务指标。如果 reward 升高但人工效果下降，就要怀疑 reward hacking，调整奖励函数或加入人工评估。

## 14. PPO 实战检查清单

训练前：

- Reward Model 是否可靠；
- SFT 模型是否可用；
- prompt 数据是否覆盖业务场景；
- KL 系数是否设置；
- 生成长度和 temperature 是否合理。

训练中：

- reward 是否稳定；
- KL 是否过大；
- 输出长度是否异常；
- 是否出现重复、模板化、胡编；
- 是否出现 NaN；
- 人工抽样是否变好。

训练后：

- 跑验证集；
- 人工评估；
- 业务指标评估；
- 检查 reward hacking；
- 和 SFT / DPO 模型做对比。

