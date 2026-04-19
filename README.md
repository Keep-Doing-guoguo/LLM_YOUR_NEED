# LLM_YOUR_NEED

`LLM_YOUR_NEED` 是一个面向 AI 工程学习的实践仓库，内容覆盖大语言模型、多模态、计算机视觉和 Python 工程基础。项目以“原理说明 + 可运行示例 + 工程实践”为主线，适合用于系统学习 LLM 相关技术栈，也适合沉淀日常实验代码和专题笔记。

当前重点包括：

- LLM 基础算法：Word2Vec、GloVe
- Qwen 系列模型微调：SFT、LoRA、QLoRA、分布式训练
- RL 训练与偏好优化：DPO、PPO、GRPO 等方法的原理、训练流程和代码实践
- LangChain 基础组件：Prompt、Loader、Splitter、Agent、Chain、Memory、Retrieval、Milvus
- 多模态模型：CLIP、BLIP、BLIP-2、解码策略
- 计算机视觉：YOLO、DETR、图像分割、训练推理和部署
- Python 工程基础：基础语法、动态导入等补充内容

后续主要补充方向：

- RL 方向：系统补充 RLHF、DPO、PPO、GRPO、奖励模型、偏好数据构造和训练流程
- 图像分割方向：系统补充语义分割、实例分割、全景分割、SAM 系列、Mask2Former、SegFormer、数据格式、评价指标和部署实践

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

## 模块导航

### LLM 基础

路径：[llm/foundations](./llm/foundations)

该模块用于学习 NLP 和 LLM 的基础算法实现：

- `word2vec.md` / `word2vec.py`：Word2Vec 原理与实现
- `glove.md` / `glove.py`：GloVe 原理与实现

适合先理解词向量、上下文窗口、共现矩阵、CBOW、Skip-Gram 等基础概念，再进入模型微调和应用框架。

### LLM 微调

路径：[llm/finetune](./llm/finetune)

该模块包含 Qwen 系列模型微调和训练实践：

- `Qwen/`：Qwen 基础微调、LoRA、QLoRA、DeepSpeed 示例
- `Qwen1.5/`：Qwen1.5 数据样例、训练与推理脚本
- `Qwen2/`：Qwen2 微调示例
- `Qwen2.5/`：Qwen2.5 微调示例
- `Qwen2-DPO/`：DPO 训练相关代码
- `Qwen3/`：Qwen3 相关实验内容
- `Qwen3-8B-GPRO/`：Qwen3 8B GRPO 实验笔记与数据
- `05-Distributed Training/`：Data Parallel、DDP、Accelerate、DeepSpeed 分布式训练示例

说明：目录中 `GPRO` 命名保留了当前历史路径，实际语义对应 GRPO（Group Relative Policy Optimization）。后续 RL 模块整理时可统一重命名为 `Qwen3-8B-GRPO`。

### LangChain 实践

路径：[llm/langchain](./llm/langchain)

该模块用于学习 LangChain 的基础组件和常见组合方式：

- `01PROMPT/`：PromptTemplate、ChatPromptTemplate
- `02LOAD/`：文档加载
- `03SPLITTER/`：文本切分
- `04AGENT/` 和 `AGENT/`：Agent、ReAct、Tools 调用
- `05CHAIN/` 和 `CHAIN/`：Chain、Memory、Retrieval、SSE、SQL 示例
- `06MEMORY/`：对话记忆
- `7-milvus.md`：Milvus 向量数据库基础

该模块还保留了一些手写 ReAct Agent、工具调用、搜索、天气查询、SQL 和 SSE 示例，便于对比框架能力与底层实现思路。

### 多模态

路径：[multimodal](./multimodal)

该模块用于学习视觉语言模型和多模态基础：

- CLIP 模型模拟与理解
- BLIP、BLIP-2 模型结构与训练流程
- 多模态发展路线
- Greedy Search、Beam Search 等解码策略

### 计算机视觉

路径：[computer-vision](./computer-vision)

该模块用于系统学习目标检测、图像分割和视觉模型部署。其中图像分割是后续重点补充方向。

- `00-基础概念/`：数据格式、坐标体系、置信度、类别概率、框类型
- `01-YOLO版本演进/`：YOLOv1 到 YOLOv10、YOLOX 等版本说明
- `02-专题机制/`：IoU、GIoU、DIoU、CIoU、DFL 等机制
- `03-训练推理流程/`：各 YOLO 版本训练与推理流程
- `04-部署与数据转换/`：VOC/YOLO 数据转换、推理部署方式
- `05-DETR系列/`：DETR、RT-DETR 系列
- `06-图像分割/`：语义分割、实例分割、全景分割、SAM、Mask2Former、SegFormer、评价指标、ONNX/TensorRT 部署

### Python 工程基础

路径：[python](./python)

该模块用于补充 Python 工程开发中的基础知识，例如基础语法、动态导入等内容。

## 推荐学习路线

1. Python 基础
   - 先阅读 [python](./python) 中的基础内容，补齐脚本开发和工程组织能力。

2. NLP 基础算法
   - 阅读 [llm/foundations](./llm/foundations)，理解 Word2Vec、GloVe 和词向量训练思路。

3. LLM 微调
   - 从 [llm/finetune/Qwen1.5](./llm/finetune/Qwen1.5) 或 [llm/finetune/Qwen2](./llm/finetune/Qwen2) 的小模型示例入手。
   - 再学习 LoRA、QLoRA、DeepSpeed 和分布式训练。

4. LangChain 应用
   - 按 Prompt、Loader、Splitter、Chain、Memory、Agent 的顺序阅读 [llm/langchain](./llm/langchain)。
   - 再结合 Retrieval、Milvus、Tools 等示例理解 RAG 和 Agent 应用。

5. RL 训练与偏好优化
   - 阅读 DPO、PPO、GRPO 相关脚本，理解 SFT 后的偏好优化流程。
   - 后续重点补充奖励模型、偏好数据构造、策略优化和训练稳定性相关内容。

6. 多模态与计算机视觉
   - 先学习 [computer-vision](./computer-vision) 中的数据格式、YOLO、DETR 和分割基础。
   - 图像分割部分后续会作为重点方向，建议按语义分割、实例分割、全景分割、SAM 系列和部署实践逐步学习。
   - 再学习 [multimodal](./multimodal) 中的 CLIP、BLIP、BLIP-2 和解码策略。

## 工程规范

为了保持该仓库适合作为长期学习项目，建议遵循以下约定：

- 文档和代码放在同一主题模块下，项目级说明放入 `docs/`
- 图片、流程图等公共资源放入 `assets/imgs/`
- 大模型权重、缓存文件、训练产物和大数据集不要提交到 Git
- 示例数据尽量小型化，完整数据集使用 README 说明下载方式
- 新增专题时优先放入已有一级主题，例如 `llm/`、`multimodal/`、`computer-vision/`
- 新增代码时尽量补充对应 README 或 Markdown 说明，记录运行方式、依赖和输入输出

## 项目进展

### LLM

- 已完成 Qwen 系列 LoRA / QLoRA 微调脚本
- 已补充分布式训练示例，包括 DP、DDP、Accelerate、DeepSpeed
- 已加入 DPO、PPO、GRPO 等 RL/RLHF 相关实验内容
- 已完成 Word2Vec、GloVe 基础算法实现
- 后续重点补充 RL 训练、偏好优化、奖励模型和训练流程实践

### LangChain

- 已覆盖 Prompt、Loader、Splitter、Chain、Memory、Agent 等基础模块
- 已包含工具调用、搜索、天气查询、SQL、SSE、Retrieval、Milvus 等示例

### 多模态

- 已整理 CLIP、BLIP、BLIP-2 相关模型说明
- 已补充 Greedy Search、Beam Search 等解码策略说明

### 计算机视觉

- 已整理 YOLO 系列版本演进、核心机制、训练推理流程
- 已补充 DETR / RT-DETR 系列说明
- 已整理图像分割基础、SAM、Mask2Former、SegFormer、部署和评价指标等内容
- 后续重点补充图像分割的数据格式、训练流程、评价指标、自动标注和部署实践

## 后续计划

- 系统补充 RL 方向内容，包括 RLHF、DPO、PPO、GRPO、奖励模型、偏好数据构造、训练流程和常见问题
- 系统补充图像分割方向内容，包括语义分割、实例分割、全景分割、SAM 系列、Mask2Former、SegFormer、数据格式、评价指标和部署实践
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
