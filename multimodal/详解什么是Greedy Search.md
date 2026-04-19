

# ✅ 详解：Greedy Search（贪心搜索）

## 一、前言

**Greedy Search 是最简单、最直接的解码策略**，广泛应用于自然语言生成任务中，例如：

- ✅ 机器翻译（Machine Translation）
- ✅ 图像描述生成（Image Captioning）
- ✅ 文本摘要（Summarization）
- ✅ 对话系统（Dialogue Generation）

它的核心思想是：

> “**每一步只选择当前概率最高的 token，不考虑整体序列最优性**。”

本文将从以下方面进行详解：

| 内容 | 是否真实存在 |
|------|----------------|
| ✅ 基本原理 | ✅ 是 |
| ✅ 算法流程 | ✅ 是 |
| ✅ 与 Beam Search 对比 | ✅ 是 |
| ✅ 在 T5 / BLIP / GPT 中的应用 | ✅ 是 |

---

## 二、Greedy Search 的基本原理

### 🧠 核心思想：

Greedy Search 是一种**局部最优解策略**，它在每个时间步仅选择当前得分最高的 token，不保留其他可能路径。

### ⚙️ 示例说明：

假设我们正在生成一句话，模型输出如下：

```
Step 1: 当前词为 "The"
    - 下一个词的概率：
        "cat" → 0.6
        "dog" → 0.3
        "car" → 0.1
    → Greedy 选择 "cat"

Step 2: 当前词为 "The cat"
    - 下一个词的概率：
        "is" → 0.7
        "runs" → 0.2
        "sleeps" → 0.1
    → Greedy 选择 "is"

Step 3: 当前词为 "The cat is"
    - 下一个词的概率：
        "on" → 0.8
        "under" → 0.1
        "beside" → 0.05
    → Greedy 选择 "on"

最终输出："The cat is on"
```

---

## 三、Greedy Search 的完整流程图（文字版）

```
初始化输入 → <BOS> 或 [CLS]
│
├— Step 1: 预测下一个 token（选择最高概率的那个）
│
├— Step 2: 将该 token 加入已生成序列
│
└— Step 3: 重复上述步骤，直到遇到 <EOS> 或达到 max_length
```

---

## 四、Greedy Search 的数学定义（简化版）

给定模型对 token 的联合概率分布 $ P(y_1, y_2, ..., y_T) $，Greedy Search 的目标是在每一步选择当前最优 token：

$$
\hat{y}_t = \arg\max_{y_t} P(y_t | y_{<t})
$$

即：在已知前面 t-1 个 token 的情况下，选择第 t 个 token 使得当前条件概率最大。

---

## 五、Greedy Search 的伪代码（简化逻辑）

```python
def greedy_search(model, start_token, max_len=30):
    generated_tokens = [start_token]  # 起始 token（如 <BOS>）

    for _ in range(max_len):
        logits = model(generated_tokens)  # 获取下个 token 的 logit
        next_token = logits.argmax()     # 贪心选择概率最高的 token
        generated_tokens.append(next_token)

        if next_token == end_token:
            break

    return generated_tokens
```

---

## 六、Greedy Search 的关键参数说明（来自 HuggingFace Transformers）

| 参数 | 含义 | 是否必须 |
|--------|--------|--------------|
| ✅ `num_beams` | 如果设为 1，则等价于 Greedy Search | ✅ 是（beam_size=1） |
| ✅ `do_sample` | 是否采样（False 表示 Greedy） | ✅ 是 |
| ❌ `temperature` | 温度调节（Greedy 不使用） | ❌ 否 |
| ❌ `top_k` / `top_p` | 采样策略参数（Greedy 不使用） | ❌ 否 |

---

## 七、Greedy Search 与其他解码方法对比

| 解码策略 | 是否使用贪心 | 是否 Beam Search 变体 | 是否支持多样性 | 是否稳定 |
|-----------|----------------|--------------------------|------------------|----------------|
| ✅ Greedy Search | ✅ 是 | ❌ 否 | ❌ 否 | ✅ 是 |
| ✅ Beam Search | ❌ 否 | ✅ 是 | ❌ 否 | ✅ 是 |
| ✅ Sampling | ❌ 否 | ❌ 否 | ✅ 是 | ❌ 否 |
| ✅ Top-k Sampling | ❌ 否 | ❌ 否 | ✅ 是 | ❌ 否 |
| ✅ Top-p (Nucleus) Sampling | ❌ 否 | ❌ 否 | ✅ 是 | ❌ 中等 |

---

### 📊 性能对比（假设模型为 T5-base）

| 解码方法 | BLEU 分数 | 输出质量 | 多样性 | 是否稳定 |
|-------------|---------------|------------|------------|--------------|
| ✅ Greedy Search | ~20.1 | 中等 | ❌ 低 | ✅ 高 |
| ✅ Beam Search（beam=5） | ~23.6 | 高 | ❌ 低 | ✅ 高 |
| ✅ Sampling | ~21.2 | 中等 | ✅ 高 | ❌ 中等 |
| ✅ Top-k Sampling | ~22.5 | 高 | ✅ 高 | ✅ 高 |

---

## 八、Greedy Search 在实际模型中的调用方式

### ✅ 1. 在 HuggingFace Transformers 中调用（T5、GPT、OPT 等）

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

input_ids = tokenizer("translate English to German: The cat is on the table", return_tensors="pt").input_ids
outputs = model.generate(input_ids, decoder_start_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(outputs[0]))
```

默认情况下，`num_beams=1` 并且 `do_sample=False` 即表示使用 Greedy Search。

---

### ✅ 2. 在 BLIP / BLIP-2 中使用 Greedy Search（图像描述生成）

```python
from lavis.models import load_model_and_preprocess
from PIL import Image

raw_image = Image.open("test.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0)

model = load_model_and_preprocess(name="blip_caption", model_type="base_coco")
caption = model.generate({"image": image}, use_nucleus_sampling=False, num_beams=1)
print(caption)  # 输出："A red sports car parked on a street."
```

---

## 九、Greedy Search 的优缺点总结

| 特点 | Greedy Search |
|--------|----------------|
| ✅ 优点 |
| ✅ 推理速度快 | 每步只需一次 forward pass |
| ✅ 显存占用低 | 不需要保存多个候选路径 |
| ✅ 实现简单 | 易于调试和部署 |
| ✅ 输出结果稳定 | 相同输入始终得到相同输出 |
| ❌ 缺点 |
| ❌ 局部最优问题 | 每步选当前最优，未必全局最优 |
| ❌ 生成文本缺乏连贯性 | 容易陷入局部循环或无意义输出 |
| ❌ 多样性差 | 相同输入每次输出一样 |
| ❌ 不适合长句生成 | 容易提前终止或语义断裂 |

---

## 十、Greedy Search 的典型应用场景

| 场景 | 是否推荐使用 |
|--------|----------------|
| ✅ 快速原型开发 | ✅ 是 |
| ✅ 测试阶段 | ✅ 是 |
| ✅ 简单任务（如小数据集翻译） | ✅ 是 |
| ✅ 对输出稳定性要求高 | ✅ 是 |
| ❌ 对生成多样性要求高 | ❌ 否 |
| ❌ 长句生成任务 | ❌ 否 |
| ❌ 需要高质量翻译/摘要 | ❌ 否（建议使用 Beam Search） |

---

## 十一、Greedy Search 与 Beam Search 的对比分析

| 比较维度 | Greedy Search | Beam Search |
|-------------|----------------|----------------|
| ✅ 是否贪心 | ✅ 是 | ❌ 否 |
| ✅ 是否 Beam | ✅ 否（beam_size=1） | ✅ 是（beam_size>1） |
| ✅ 是否多路径 | ❌ 否 | ✅ 是 |
| ✅ 显存占用 | ✅ 低 | ❌ 较高 |
| ✅ 输出是否稳定 | ✅ 是 | ✅ 是 |
| ✅ 多样性 | ❌ 低 | ✅ 低（但更合理） |
| ✅ 推荐场景 | ✅ 快速测试 | ✅ 正式部署、SOTA 任务 |

---

## 十二、Greedy Search 在推理中的调用方式（HuggingFace）

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    num_beams=1,          # Greedy 搜索
    do_sample=False,      # 不采样，强制使用 argmax
    early_stopping=True
)

print(tokenizer.decode(output[0]))
```

---

## 十三、Greedy Search 的变种（真实存在）

| 方法 | 描述 | 是否 Greedy Search 改进 |
|--------|--------|------------------------------|
| ✅ Beam Search | 使用多个候选路径 | ❌ 否（属于不同类别） |
| ✅ Sampling | 按概率分布采样 | ❌ 否（属于随机采样） |
| ✅ Temperature Scaling | 控制 softmax 分布温度 | ❌ 否（Greedy 可结合使用） |
| ✅ No Repeat N-Gram | 强制不重复某些 n-gram | ✅ 否（可用于 Greedy） |

---

## 十四、结语

**Greedy Search 是最基础的解码策略**，其核心思想是：

> “**每一步都选择当前概率最高的 token，不考虑未来路径**”

虽然它速度最快、显存最低，但在复杂任务中容易出现局部最优、语义断裂等问题。

如果你的任务对生成质量要求不高、推理速度优先级更高，那么 Greedy Search 是一个合适的解码策略。

掌握 Greedy Search 的原理和使用方式，有助于你理解现代语言模型的推理机制，并为进一步优化打下基础。

---

📌 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于 Transformer、Beam Search、Greedy Search、深度学习等内容！**

