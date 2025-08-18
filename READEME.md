
## 1.LLM_YOUR_NEED

本项目包含多种大语言模型（LLM）相关的微调、训练和基础算法实现示例，涵盖 Qwen 系列模型的多版本微调（包括分布式训练、LoRA、GRPO 等），以及经典词向量算法（GloVe、Word2Vec）的实现与说明。


## 2.目录结构
```java
LLM_YOUR_NEED/
├── finetune/                   # 各版本 Qwen 模型微调代码
│   ├── 05-Distributed Training # 分布式训练示例
│   ├── Qwen                    # Qwen 基础微调
│   ├── Qwen1.5
│   ├── Qwen2
│   ├── Qwen2.5
│   ├── Qwen3
│   ├── Qwen3-8B-GPRO           # Qwen3 8B GRPO 示例
│   ├── [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
│   ├── [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
│   └── finetune.md             # 微调说明文档
│
├── foundations/                # NLP 基础算法
│   ├── glove.md / glove.py      # GloVe 算法原理与实现
│   ├── word2vec.md / word2vec.py# Word2Vec 算法原理与实现
│
├── imgs/                       # 项目配图
├── readme.md                   # 项目说明


```

## 3.功能特点
	1.	Qwen 系列模型微调
	•	支持 Qwen、Qwen1.5、Qwen2、Qwen2.5、Qwen3 等多个版本
	•	支持 单机单卡 / 单机多卡 / 多机多卡 训练
	•	支持 LoRA 微调 与 全量微调
	•	集成 GRPO（Group Relative Policy Optimization） 强化训练示例
	2.	分布式训练示例
	•	torchrun 启动
	•	DeepSpeed 集成优化
	•	支持多节点 GPU 集群训练
	3.	词向量算法实现
	•	GloVe：从文本语料构建共现矩阵并训练词向量
	•	Word2Vec：CBOW / Skip-Gram 模型的完整实现


## 4.项目进展

主要包括四类：

**1、SFT（单机单卡、单机多卡、多机多卡、accelerate、deepspeed）**

    •	✅ 完成 Qwen 系列 LoRA 微调脚本
	•	✅ 完成分布式训练配置
	•	✅ 添加 GloVe / Word2Vec 实现
	•	⏳ 待添加更多 RLHF / DPO 示例
    •	

**2、RLHF**
    •	✅ 完成 Qwen3 的GPRO脚本
    •	✅ 完成 Qwen3 的DPO脚本
    •	✅ 完成 Qwen3 的PPO脚本

**3、VL ：**

**4、LLM Basic ：**
    •	✅ 完成 词向量算法实现（Word2Vec、GloVe）

## 🤝 致谢！

### 核心贡献者

- [张文文-项目负责人和开发人](https://github.com/Keep-Doing-guoguo) 
- [黄家嘉-实际开发人](https://github.com/869924024) 

项目学习了优秀开源项目，感谢huggingface、流萤及一些国内外开源项目。
部分lora代码和讲解参考仓库：

- 1.https://github.com/datawhalechina/self-llm
- 2.https://github.com/mst272/LLM-Dojo
- 3.https://github.com/QwenLM/Qwen
- 4.https://github.com/philschmid/deep-learning-pytorch-huggingface

🪂 无论是提出问题（Issue）还是贡献代码（Pull Request），都是对项目的巨大支持。
***
