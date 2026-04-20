# LLM_YOUR_NEED

面向 AI 工程学习的实践仓库，覆盖 **大语言模型、RL/RLHF、多模态、计算机视觉、Python 工程基础** 等方向。

项目目标不是做一个单一应用，而是沉淀一套可持续学习和复现实验的工程资料库：每个专题尽量同时保留原理说明、代码示例、训练流程、数据格式和工程实践。

## 快速导航

| 方向 | 路径 | 内容 |
| --- | --- | --- |
| LLM 基础 | [llm/foundations](./llm/foundations) | Word2Vec、GloVe、词向量基础算法 |
| LLM 微调 | [llm/finetune](./llm/finetune) | Qwen 系列 SFT、LoRA、QLoRA、分布式训练 |
| RL / RLHF | [llm/finetune](./llm/finetune) | DPO、PPO、GRPO、偏好优化，后续重点补充 |
| LangChain | [llm/langchain](./llm/langchain) | Prompt、Loader、Splitter、Chain、Agent、Memory、Milvus |
| 多模态 | [multimodal](./multimodal) | CLIP、BLIP、BLIP-2、视觉语言模型、解码策略 |
| 计算机视觉 | [computer-vision](./computer-vision) | YOLO、DETR、图像分割、训练推理、部署 |
| Python 基础 | [python](./python) | Python 基础语法、动态导入、工程补充 |
| 项目文档 | [docs](./docs) | 环境说明、路线图、参考资料等项目级文档 |

## 当前重点

本仓库目前重点维护两个方向：

### 1. RL / RLHF

后续会围绕大模型训练后的偏好优化和强化学习训练流程继续补充：

- RLHF 基础流程
- 奖励模型训练
- 偏好数据构造
- DPO / PPO / GRPO 原理与代码实践
- SFT 到 RLHF 的完整训练链路
- 训练稳定性、显存优化和常见问题

### 2. 图像分割

计算机视觉部分后续会重点补充图像分割方向：

- 语义分割、实例分割、全景分割
- Mask、Polygon、RLE、COCO Segmentation 等数据格式
- mIoU、Dice、Mask AP、PQ 等评价指标
- YOLO-Seg、Mask R-CNN、Mask2Former、SegFormer
- SAM / SAM2 / SAM3 相关模型与自动标注流程
- ONNX、TensorRT 等部署实践

## 目录结构

```text
LLM_YOUR_NEED/
├── README.md                   # 项目总览、模块导航和学习路线
├── docs/                       # 项目级文档、环境说明、路线图、参考资料
├── assets/
│   └── imgs/                   # README、算法说明和笔记中使用的图片资源
├── llm/
│   ├── foundations/            # NLP/LLM 基础算法：Word2Vec、GloVe
│   ├── finetune/               # Qwen 微调、分布式训练、RL/RLHF 示例
│   └── langchain/              # LangChain 基础组件与应用示例
├── multimodal/                 # CLIP、BLIP、BLIP-2、多模态发展与解码策略
├── computer-vision/            # 目标检测、图像分割、训练推理、部署与数据格式
└── python/                     # Python 工程基础补充
```

## 模块说明

### LLM 基础

[llm/foundations](./llm/foundations) 用于学习 NLP 和 LLM 的基础算法实现。

| 文件 | 说明 |
| --- | --- |
| `word2vec.md` / `word2vec.py` | Word2Vec 原理与实现，包含 CBOW / Skip-Gram 思路 |
| `glove.md` / `glove.py` | GloVe 原理与实现，包含共现矩阵和词向量训练 |

### LLM 微调与训练

[llm/finetune](./llm/finetune) 包含 Qwen 系列模型微调、分布式训练和 RL 相关实践。

| 子目录 | 说明 |
| --- | --- |
| `Qwen/` | Qwen 基础微调、LoRA、QLoRA、DeepSpeed 示例 |
| `Qwen1.5/` | Qwen1.5 数据样例、训练与推理脚本 |
| `Qwen2/` | Qwen2 微调示例 |
| `Qwen2.5/` | Qwen2.5 微调示例 |
| `Qwen2-DPO/` | DPO 训练相关代码 |
| `Qwen3/` | Qwen3 相关实验内容 |
| `Qwen3-8B-GPRO/` | Qwen3 8B GRPO 实验笔记与数据 |
| `05-Distributed Training/` | DP、DDP、Accelerate、DeepSpeed 分布式训练示例 |

说明：`Qwen3-8B-GPRO` 是历史目录名，实际语义对应 GRPO（Group Relative Policy Optimization）。后续整理 RL 模块时可以统一改为 `Qwen3-8B-GRPO`。

### LangChain

[llm/langchain](./llm/langchain) 用于学习 LangChain 基础组件和常见组合方式。

| 子目录 / 文件 | 说明 |
| --- | --- |
| `01PROMPT/` | PromptTemplate、ChatPromptTemplate |
| `02LOAD/` | 文档加载 |
| `03SPLITTER/` | 文本切分 |
| `04AGENT/` / `AGENT/` | Agent、ReAct、Tools 调用 |
| `05CHAIN/` / `CHAIN/` | Chain、Memory、Retrieval、SSE、SQL 示例 |
| `06MEMORY/` | 对话记忆 |
| `7-milvus.md` | Milvus 向量数据库基础 |

### 多模态

[multimodal](./multimodal) 主要整理视觉语言模型和解码策略：

- CLIP 模型模拟与理解
- BLIP / BLIP-2 模型结构与训练流程
- 多模态发展路线
- Greedy Search、Beam Search 等解码策略

### 计算机视觉

[computer-vision](./computer-vision) 主要整理目标检测、图像分割和视觉模型部署。

| 子目录 | 说明 |
| --- | --- |
| `00-基础概念/` | 数据格式、坐标体系、置信度、类别概率、框类型 |
| `01-YOLO版本演进/` | YOLOv1 到 YOLOv13、YOLOX 等版本说明 |
| `02-专题机制/` | IoU、GIoU、DIoU、CIoU、DFL 等机制 |
| `03-训练推理流程/` | 各 YOLO 版本训练与推理流程 |
| `04-部署与数据转换/` | VOC/YOLO 数据转换、推理部署方式 |
| `05-DETR系列/` | DETR、RT-DETR 系列 |
| `06-图像分割/` | 语义分割、实例分割、SAM、Mask2Former、SegFormer、评价指标、部署 |

## 推荐学习路线

```text
Python 基础
  -> NLP 基础算法
  -> Qwen SFT / LoRA / QLoRA
  -> 分布式训练
  -> LangChain / RAG / Agent
  -> RLHF / DPO / PPO / GRPO
  -> 多模态 / 图像分割 / 部署
```

建议顺序：

1. 先阅读 [python](./python)，补齐 Python 工程基础。
2. 阅读 [llm/foundations](./llm/foundations)，理解 Word2Vec、GloVe 和词向量训练思路。
3. 从 [llm/finetune/Qwen1.5](./llm/finetune/Qwen1.5) 或 [llm/finetune/Qwen2](./llm/finetune/Qwen2) 入手，学习小模型微调。
4. 再学习 LoRA、QLoRA、DeepSpeed、DDP、Accelerate 等训练方式。
5. 按 Prompt、Loader、Splitter、Chain、Memory、Agent 的顺序学习 [llm/langchain](./llm/langchain)。
6. 继续补充 RLHF、DPO、PPO、GRPO 等偏好优化和强化学习训练内容。
7. 视觉方向先学习 YOLO / DETR，再重点学习图像分割和部署。

## 工程约定

- 文档和代码放在同一主题模块下，项目级说明放入 `docs/`
- 图片、流程图等公共资源放入 `assets/imgs/`
- 大模型权重、缓存文件、训练产物和大数据集不要直接提交到 Git
- 示例数据尽量小型化，完整数据集在文档中说明下载方式
- 新增专题时优先放入已有一级主题，例如 `llm/`、`multimodal/`、`computer-vision/`
- 新增代码时补充对应 README 或 Markdown 说明，记录运行方式、依赖、输入和输出

## 项目进展

| 方向 | 当前状态 |
| --- | --- |
| LLM 基础 | 已整理 Word2Vec、GloVe 的原理说明和代码实现 |
| Qwen 微调 | 已包含 Qwen、Qwen1.5、Qwen2、Qwen2.5、Qwen3 相关示例 |
| 分布式训练 | 已包含 DP、DDP、Accelerate、DeepSpeed 示例 |
| RL / RLHF | 已包含 DPO、PPO、GRPO 相关实验内容，后续重点补充 |
| LangChain | 已覆盖 Prompt、Loader、Splitter、Chain、Memory、Agent、Milvus 等内容 |
| 多模态 | 已整理 CLIP、BLIP、BLIP-2、搜索解码策略 |
| 计算机视觉 | 已整理 YOLO、DETR、图像分割、训练推理和部署相关内容 |

## 后续计划

优先级最高：

- 系统补充 RL 方向内容：RLHF、DPO、PPO、GRPO、奖励模型、偏好数据构造、训练流程和常见问题
- 系统补充图像分割方向内容：语义分割、实例分割、全景分割、SAM 系列、Mask2Former、SegFormer、数据格式、评价指标和部署实践

工程整理：

- 统一部分历史目录命名，例如 `GPRO` -> `GRPO`
- 合并 LangChain 中重复的 `04AGENT` / `AGENT`、`05CHAIN` / `CHAIN` 目录
- 为主要模块补充独立 README
- 增加统一环境说明，例如 PyTorch、Transformers、Accelerate、DeepSpeed、LangChain 等版本建议
- 补充训练脚本运行命令、数据格式说明和常见问题
- 增加 `.gitignore`，忽略 `.DS_Store`、IDE 配置、模型权重、缓存和训练输出

## 致谢

核心贡献者：

- [张文文 - 项目负责人和开发人](https://github.com/Keep-Doing-guoguo)

项目学习和参考了 Hugging Face、Qwen 以及多个优秀开源项目。部分 LoRA 代码和讲解参考：

- https://github.com/datawhalechina/self-llm
- https://github.com/mst272/LLM-Dojo
- https://github.com/QwenLM/Qwen
- https://github.com/philschmid/deep-learning-pytorch-huggingface

欢迎通过 Issue 或 Pull Request 参与讨论和贡献。
