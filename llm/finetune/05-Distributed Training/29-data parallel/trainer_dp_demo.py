#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/8/13 11:04
@source from:

DP（DataParallel）是单进程控制多张 GPU，自动切分 batch 并聚合梯度，简单但效率和显存利用率较低。
DDP（DistributedDataParallel）是多进程多 GPU，每个进程独立计算并通过通信同步梯度，速度快且扩展性好。

这个是DP
这种是 单进程+多 GPU，由 PyTorch 自动做梯度聚合，简单但显存利用率低、速度慢。
命令（假设你要用 4 张卡）：
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py

特点：
	•	只能在单机上用
	•	不支持多节点
	•	Trainer 会自动调用 torch.nn.DataParallel
	•	每个 batch 会被切分到不同 GPU，但还是单进程调度

这个是DDP
这种是 多进程+多 GPU，也是 HuggingFace 推荐的方式，效率更高，显存利用率好。
命令（单机 4 卡）：
torchrun --nproc_per_node=4 train.py

特点：
	•	每张卡一个进程
	•	HuggingFace Trainer 会检测 torch.distributed 环境，自动初始化 DDP
	•	可以轻松扩展到多机多卡（加 --nnodes、--node_rank、--master_addr 等参数）

"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)

datasets = dataset.train_test_split(test_size=0.1)

import torch

tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)

model = BertForSequenceClassification.from_pretrained("hfl/rbt3")

print(model.config)

import evaluate

acc_metric = evaluate.load("./metric_accuracy.py")
f1_metirc = evaluate.load("./metric_f1.py")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

hasattr(train_args, "_n_gpu")
train_args.__dict__

from transformers import DataCollatorWithPadding
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()

trainer.evaluate(tokenized_datasets["test"])