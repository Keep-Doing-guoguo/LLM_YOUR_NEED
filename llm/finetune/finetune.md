# 大模型微调总览

这份文件作为微调目录的总览入口，重点说明分布式训练和显存优化基础。具体的 LoRA / QLoRA、DPO、PPO 已拆成独立文档。

## 1. 数据并行、张量并行和 DDP

数据并行（Data Parallelism, DP）是最常用的并行训练方法。它将数据划分为多个批次，并在多个 GPU 上并行处理每个批次。每个 GPU 拥有完整的模型副本，并在每个批次后同步梯度。

张量并行（Tensor Parallelism, TP）是将模型的参数切分到多个 GPU 上。每个 GPU 只存储模型的一部分参数，并在前向和后向传播时进行通信以完成计算。

分布式数据并行（Distributed Data Parallel, DDP）是 PyTorch 提供的一种数据并行方法，多进程（每 GPU 一个进程），每个进程独立持有模型副本。它把一个批次的数据切分成多份，每个 GPU 处理一部分数据，计算 loss 和梯度，然后同步参数。

总结：

- DP（DataParallel）是单进程控制多张 GPU，自动切分 batch 并聚合梯度，简单但效率和显存利用率较低。
- DDP（DistributedDataParallel）是多进程多 GPU，每个进程独立计算并通过通信同步梯度，速度快且扩展性好。
- DP：多卡多数据，每卡一份完整模型，靠 batch 数据并行训练。
- TP：多卡一模型，把一层的计算拆到多卡上完成。

## 2. DeepSpeed 中的 ZeRO 并行

ZeRO 是 DeepSpeed 提供的内存优化方法，用来把优化器状态、梯度、模型参数切分到多卡。

| 阶段 | 切分内容 | 说明 |
|------|----------|------|
| ZeRO-1 | optimizer states | 切分优化器状态 |
| ZeRO-2 | optimizer states + gradients | 切分优化器状态和梯度 |
| ZeRO-3 | optimizer states + gradients + parameters | 进一步切分模型参数，最省显存 |

优点：

- 可以训练更大的模型；
- 降低单卡显存压力；
- 适合多卡大模型训练。

缺点：

- 通信复杂度增加；
- ZeRO-3 保存权重更复杂；
- 和 LoRA / QLoRA / 量化训练组合时要特别注意参数 gather。

## 3. Accelerate 和 DeepSpeed 的关系

| 特性 | Accelerate | DeepSpeed |
|------|------------|-----------|
| 核心定位 | 分布式训练封装器 | 高性能分布式训练引擎 |
| 作用重点 | 统一封装 DDP、FSDP、DeepSpeed 等后端 | 提供 ZeRO、流水线并行、张量并行等优化 |
| 本质 | 调度 / 管理层 | 执行 / 优化层 |

可以理解成：

```text
Accelerate = 自动挡驾驶系统
DeepSpeed = 高性能发动机
```

Accelerate 可以帮你更简单地启用 DeepSpeed，但真正做显存优化和高性能训练的是 DeepSpeed。

## 4. 微调专题文档

| 文件 | 内容 |
|------|------|
| [LoRA和QLoRA微调详解.md](./LoRA和QLoRA微调详解.md) | LoRA、QLoRA 原理、代码、项目问题、面试题 |
| [DPO偏好优化详解.md](./DPO偏好优化详解.md) | DPO 原理、数据格式、训练流程、项目问题、面试题 |
| [PPO强化学习微调详解.md](./PPO强化学习微调详解.md) | PPO / RLHF 原理、代码流程、Reward Model、项目问题、面试题 |

## 5. 推荐学习顺序

```text
1. 先理解 SFT：监督微调和 labels mask
2. 再理解 LoRA / QLoRA：怎么低成本训练
3. 再理解 DPO：怎么用偏好对优化模型
4. 最后理解 PPO：怎么用奖励模型做强化学习优化
5. 项目训练时再结合 DDP / DeepSpeed / ZeRO 解决显存和速度问题
```

