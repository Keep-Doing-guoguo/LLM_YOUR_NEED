
# ✅ BLIP-2 模型结构详解

## 一、前言

**BLIP-2（Bootstrapped Language-Image Pre-training）** 是由 **Salesforce Research 团队于 2023 年提出的一种多模态预训练模型**。其核心思想是：

> “**利用已有的强大视觉和语言模型，通过一个轻量模块连接二者，避免端到端训练带来的高昂成本**”。

本文将**只详解其模型结构**，包括：
- ✅ 视觉编码器；
- ✅ Q-Former 模块；
- ✅ 语言解码器；

所有内容均来自：
- ✅ [BLIP-2 论文](https://arxiv.org/abs/2301.13659)
- ✅ [GitHub: Salesforce/LAVIS](https://github.com/salesforce/LAVIS)

---

## 二、BLIP-2 的完整模型结构流程图（输入图像：224×224）

```
Input Image → ViT Encoder (Frozen) → Patch Embeddings
                   ↓
           Q-Former（可训练）
                   ↓
Language Model (OPT / FlanT5, Frozen) → Output Text
```

---

## 三、BLIP-2 的三大核心组件

### ✅ 1. 视觉编码器（ViT-B/16、ViT-Giant/14 或 EVA-CLIP）

- ✅ 输入图像尺寸：224 × 224；
- ✅ 输出 patch embeddings：`[B, N, D]`，其中：
  - `B`: batch size；
  - `N`: patch 数量（如 ViT-Giant 输出为 `[B, 256, 1408]`）；
  - `D`: embedding 维度（1408 或 1664）；

> ⚠️ 注意：**该模块在训练过程中完全冻结**，不更新参数。

---

### ✅ 2. Q-Former 模块（Querying Transformer）

#### 📌 来源依据：
- [论文 Section 3.1](https://arxiv.org/abs/2301.13659)
- [GitHub: LAVIS 中的 `Qformer.py`](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/Qformer.py)

#### 🧠 核心结构说明：

Q-Former 是一个小型 Transformer 编码器，包含以下关键组件：

| 组件 | 功能 |
|--------|--------|
| ✅ Learnable Query Tokens | 可学习的 queries（数量通常为 32~64） |
| ✅ Cross Attention | 查询图像特征，建立跨模态联系 |
| ✅ Self Attention | 在 query tokens 之间建模关系 |
| ✅ FeedForward Network（FFN） | 提升表达能力 |
| ✅ LayerNorm + Linear Projection | 对齐语言模型维度 |

---

#### ⚙️ 输入输出维度示例（以 ViT-Giant + FlanT5-xl 为例）：

| 阶段 | 输入 | 输出 |
|--------|--------|--------|
| ✅ 图像编码 | `[B, 3, 224, 224]` | `[B, 256, 1408]` |
| ✅ Q-Former | `[B, 64, 1408]`（query tokens）+ `[B, 256, 1408]`（ViT 输出） | `[B, 64, 2048]`（适配 T5 输入） |

---

#### 📌 示例伪代码（简化版）：

```python
class QFormer(nn.Module):
    def __init__(self, num_query_tokens=64, vision_width=1408, lm_width=2048):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, vision_width))
        self.encoder = BertEncoder()  # 改进的 BERT 编码器
        self.ln = BertLayerNorm(vision_width)
        self.proj = nn.Linear(vision_width, lm_width)  # 投影到语言模型输入维度

    def forward(self, image_embeds):
        """
        image_embeds: [B, N, D] ← ViT 输出的 patch embeddings
        """
        batch_size = image_embeds.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # [B, Q, D]

        # Step 1: Cross Attention with Vision Embeddings
        attention_output = self.encoder(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
        )

        # Step 2: LayerNorm + Projection
        output = self.ln(attention_output.last_hidden_state)
        output = self.proj(output)

        return output  # 返回适配语言模型的 embeddings
```

---

### ✅ 3. 语言模型（OPT / FlanT5 / BLOOM）

- ✅ 使用现成的语言模型（如 OPT-2.7b/6.7b、FlanT5-xl）；
- ✅ 参数在整个训练阶段保持冻结；
- ✅ 接收 Q-Former 输出作为 prefix 输入；
- ✅ 解码生成文本（caption、answer 等）；

> ⚠️ 注意：这部分不在 BLIP-2 的“结构创新”中，而是直接使用已有 LM，因此不在本详解重点。

---

## 四、BLIP-2 的完整模型结构总结（仅结构）

```
Input Image → ViT Encoder（Frozen）→ Patch Embeddings [B, 256, 1408]
                  ↓
          Q-Former（Trainable）
           ↗ Cross-Attention ↘
   [B, 64, 1408] → [B, 64, 2048] → 输入语言模型（Frozen）
                  ↓
          Language Model（Frozen）
                  ↓
           Output Caption / Answer
```

---

## 五、Q-Former 的详细模块结构（来自论文与代码）

Q-Former 的结构如下（简化自论文与 GitHub 实现）：

```text
Q-Former Block × N:
    ├─ Cross-Attention Layer
    │    └— Queries ← Learnable Tokens
    │    └— Keys/Values ← ViT Patch Embeddings
    ├— Self-Attention Layer
    │    └— Queries ← 上一层输出
    ├— FeedForward Network
    └— LayerNorm
```

---

## 六、Q-Former 的输入输出解析

| 输入 | 维度 | 来源 |
|--------|--------|--------|
| ✅ ViT Patch Embeddings | `[B, N, D]`（如 `[1, 256, 1408]`） | ViT Encoder 输出 |
| ✅ Learnable Query Tokens | `[Q, D]`（如 `[64, 1408]`） | Q-Former 初始化参数 |

| 输出 | 维度 | 说明 |
|--------|--------|--------|
| ✅ Q-Former Output | `[B, Q, D']`（如 `[1, 64, 2048]`） | 适配语言模型输入维度 |
| ✅ 输入语言模型 | `[B, Q, D']` | 作为 prefix 输入给 LM |

---

## 七、Q-Former 的关键组件详解（来自源码）

| 组件 | 类型 | 是否训练 |
|--------|--------|--------------|
| ✅ query_tokens | `nn.Parameter` | ✅ 是 |
| ✅ cross_attn | `nn.MultiheadAttention` 或 `F.scaled_dot_product_attention` | ✅ 是 |
| ✅ self_attn | `nn.MultiheadAttention` | ✅ 是 |
| ✅ FFN | `nn.Sequential`（Linear + GELU + Linear） | ✅ 是 |
| ✅ LayerNorm | `BertLayerNorm` | ✅ 是 |
| ✅ ViT 主干网络 | `timm.create_model("eva_giant_patch14_224")` | ❌ 否（冻结） |
| ✅ 语言模型 | `T5ForConditionalGeneration` | ❌ 否（冻结） |

---

## 八、Q-Former 的实际结构配置（来自 LAVIS 开源实现）

```yaml
qformer:
  num_query_tokens: 64
  cross_attention_freq: 2
  use_learnable_queries: True
  bert_config_name: bert-base-uncased
  freeze_vit: True
  freeze_lm: True
```

> ✅ 注：以上配置可在 `configs/models/blip2.yaml` 中找到。

---

## 九、Q-Former 的作用总结（仅结构层面）

| 作用 | 描述 |
|--------|--------|
| ✅ 跨模态信息压缩 | 将图像 patch embeddings 压缩为少量 query |
| ✅ 多任务通用接口 | 所有下游任务共享统一表示 |
| ✅ 显存友好 | 不更新视觉和语言主干 |
| ✅ 可插拔式设计 | 可替换不同 ViT 和 LM |

---

## 十、结语

BLIP-2 的模型结构可以概括为：

![BLIP-2 Q-Former](./assets/blip2-qformer.svg)

```
ViT Encoder（冻结） → Q-Former（可训练） → 冻结语言模型（如 FlanT5/OPT） → 文本生成
```

其核心在于：
- ✅ **Q-Former 模块的设计**；
- ✅ **查询机制替代全连接层**；
- ✅ **仅训练中间模块，降低训练成本**；


---

📌 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于 BLIP、YOLO、Transformer 等深度学习内容！**
