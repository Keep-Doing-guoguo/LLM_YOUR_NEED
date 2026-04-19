

# ✅ 详解：什么是 Beam Search（束搜索）

## 一、前言

**Beam Search 是一种启发式搜索算法**，广泛应用于自然语言生成任务中，例如：

- ✅ 机器翻译（Machine Translation）
- ✅ 图像描述生成（Image Captioning）
- ✅ 文本摘要（Summarization）
- ✅ 对话系统（Dialogue Generation）

它的核心思想是：

> “**在解码过程中保留多个候选路径，选择整体最优的序列输出，而非每步贪心选择最高概率 token。**”

---

## 二、Beam Search 的基本原理

### 🧠 核心思想：

传统的 **Greedy Search** 在每一步只选择当前得分最高的 token；  
而 **Beam Search** 在每一步维护一个固定大小的候选序列集合（称为 `beam_size`），并在每步扩展这些序列，保留 top-k 个最优路径。

---

### ⚙️ 示例说明（假设 beam_size = 2）：

```
初始输入 → <BOS>（beginning of sentence）

Step 1: 预测下一个词：
    - "The" (log_prob = -0.3)
    - "A" (log_prob = -0.4)
    - "This" (log_prob = -0.7)
    - ...
    → 保留 "The" 和 "A"

Step 2: 扩展每个候选：
    - "The cat" (log_prob = -0.3 + -0.2 = -0.5)
    - "The dog" (log_prob = -0.3 + -0.5 = -0.8)
    - "A car" (log_prob = -0.4 + -0.3 = -0.7)
    - "A house" (log_prob = -0.4 + -0.6 = -1.0)
    → 保留 "The cat" 和 "A car"

Step 3: 继续扩展，直到达到最大长度或遇到 <EOS>
```

---

## 三、Beam Search 的完整流程图（文字版）

```
初始化 beam → 开始 token（<BOS> 或 [CLS]）
│
├— Step 1: 预测下一 token，为每个当前序列扩展 k 个候选 token
│        └— 每个 token 的 log_prob 加入总分
│
├— Step 2: 计算每个候选序列的总得分（log-prob）
│        └— 使用 beam width 控制保留多少候选序列（如 beam_size=5）
│
├— Step 3: 合并重复路径，避免冗余
│        └— 如两条路径最后生成相同 token 序列
│
└— Step 4: 重复上述步骤，直到满足终止条件
         └— 如达到 max_length 或所有序列以 <EOS> 结尾
```

---

## 四、Beam Search 的数学定义（简化版）

给定模型对 token 的联合概率分布 $ P(y_1, y_2, ..., y_T) $，目标是找到使整个序列概率最大的句子：

$$
\hat{y}_{1:T} = \arg\max_{y_{1:T}} P(y_1, y_2, ..., y_T)
$$

由于穷举所有可能的句子组合计算量太大，**Beam Search 通过保留 top-k 个候选序列来近似求解最优解**。

---

## 五、Beam Search 的伪代码（简化逻辑）

```python
def beam_search(model, start_token, beam_size=5, max_len=30):
    # 初始化 beam：起始 token + 初始得分为 0
    beams = [{"tokens": [start_token], "score": 0.0}]
    
    for _ in range(max_len):
        all_candidates = []
        
        for beam in beams:
            current_tokens = beam["tokens"]
            probs = model(current_tokens)  # 获取下一个 token 的概率
            
            # 获取 top-k 个最可能的 token
            topk_indices = probs.topk(beam_size).indices
            
            for token in topk_indices:
                new_score = beam["score"] + log(probs[token])
                new_tokens = current_tokens + [token]
                all_candidates.append({"tokens": new_tokens, "score": new_score})
                
        # 排序并保留 top-k 最优路径
        ordered = sorted(all_candidates, key=lambda x: x["score"], reverse=True)
        beams = ordered[:beam_size]
        
    return best path in beams
```

---

## 六、Beam Search 的关键参数说明（来自 HuggingFace Transformers）

| 参数 | 含义 | 是否必须 |
|--------|--------|--------------|
| ✅ `num_beams` | beam width，保留的候选路径数量 | ✅ 是 |
| ✅ `early_stopping` | 达到 beam 数量后是否提前停止 | ✅ 否 |
| ✅ `length_penalty` | 对长句加分或惩罚（默认为 1.0） | ✅ 否 |
| ✅ `no_repeat_ngram_size` | 禁止重复 n-gram（提升流畅性） | ✅ 否 |
| ✅ `num_return_sequences` | 返回多少个生成结果（通常 ≤ num_beams） | ✅ 否 |

---

## 七、Beam Search 与其他解码策略对比

| 解码策略 | 是否使用 Beam |
|-----------|----------------|
| ✅ Greedy Search | ❌ 否 |
| ✅ Beam Search | ✅ 是 |
| ✅ Sampling | ❌ 否 |
| ✅ Top-k Sampling | ❌ 否 |
| ✅ Nucleus Sampling（Top-p） | ❌ 否 |
| ✅ Diverse Beam Search | ✅ 是（变种） |
| ✅ Constrained Beam Search | ✅ 是（带约束） |

---

### 📊 性能对比（假设模型为 T5-base）

| 解码方法 | BLEU 分数 | 输出质量 | 多样性 | 是否稳定 |
|-------------|---------------|------------|------------|--------------|
| ✅ Greedy Search | ~20.1 | 中等 | ❌ 低 | ✅ 高 |
| ✅ Beam Search（beam=5） | ~23.6 | 高 | ❌ 低 | ✅ 高 |
| ✅ Sampling | ~21.2 | 中等 | ✅ 高 | ❌ 中等 |
| ✅ Top-k Sampling | ~22.5 | 高 | ✅ 高 | ✅ 高 |
| ✅ Beam Search + Length Penalty | ~24.0 | ✅ 最高 | ❌ 低 | ✅ 高 |

---

## 八、Beam Search 在实际模型中的应用

### ✅ 1. 在 T5 中的应用

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

input_ids = tokenizer("translate English to German: The cat is on the table", return_tensors="pt").input_ids
outputs = model.generate(input_ids, num_beams=5, max_new_tokens=30, early_stopping=True)
print(tokenizer.decode(outputs[0]))
```

---

### ✅ 2. 在 BLIP-2 中的应用（图像描述生成）

```python
from lavis.models import load_model_and_preprocess
from PIL import Image

raw_image = Image.open("test.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0)

model = load_model_and_preprocess(name="blip2", model_type="pretrain_flant5xl")
caption = model.generate({"image": image}, use_beam_search=True, num_beams=5, max_length=32)
print(caption)  # 输出："A red sports car parked on a street."
```

---

### ✅ 3. 在 GPT / BERT 等自回归模型中的应用

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
output = model.generate(
    input_ids,
    max_new_tokens=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print(tokenizer.decode(output[0]))
```

---

## 九、Beam Search 的改进与变种（真实存在）

| 变种 | 描述 | 是否 Beam Search 改进 |
|--------|--------|--------------------------|
| ✅ Length Penalty | 对短句进行惩罚，鼓励更长输出 | ✅ 是 |
| ✅ No Repeat N-Gram | 避免重复词汇 | ✅ 是 |
| ✅ Diverse Beam Search | 鼓励不同路径的多样性 | ✅ 是 |
| ✅ Constrained Beam Search | 强制包含某些关键词 | ✅ 是 |
| ✅ Group Beam Search | 每组独立 beam，控制多样性 | ✅ 是 |

---

## 十、Beam Search 的优点与缺点总结

| 特点 | Beam Search |
|--------|----------------|
| ✅ 优点 |
| 提升生成质量 | 相比 Greedy 更稳定 |
| 提升翻译/摘要/描述生成性能 | 广泛用于 SOTA 模型 |
| 支持多任务统一接口 | VQA / Captioning / Translation |
| ❌ 缺点 |
| 显存占用较高 | 需要保存多个候选路径 |
| 生成文本缺乏多样性 | 默认 beam search 会收敛到相似路径 |
| 计算开销略大 | 需要多次 forward pass |

---

## 十一、Beam Search 在推理中的调用方式（HuggingFace）

```python
output = model.generate(
    input_ids,
    num_beams=5,
    max_new_tokens=50,
    length_penalty=1.0,
    no_repeat_ngram_size=2,
    num_return_sequences=1,
    early_stopping=True
)
```

---

## 十二、结语

**Beam Search 是目前主流 NLP 与多模态模型中使用最广泛的解码策略之一**，它通过维护多个候选路径，在每一步选择最有希望的 token，最终输出整体最优的文本。

掌握 Beam Search 的原理和使用方式，有助于你理解现代语言模型的推理机制，并为进一步优化生成质量打下基础。
