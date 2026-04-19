#!/usr/bin/env python
# coding=utf-8
"""
通用训练脚本：单机单卡 / 单机多卡 / 多机多卡（torchrun），可选Deepspeed，LoRA 微调
数据：输入旧 jsonl（每行含 {"text","category","output"}），会转换成 instruction 格式再训练
"""

import os, json, math, logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import pandas as pd
from datasets import Dataset

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# 数据适配：把旧 jsonl 转换为 instruction 格式
# ------------------------------
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            context = d["text"]
            category = d["category"]
            label = d["output"]
            messages.append({
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本:{context},类型选项:{category}",
                "output": label,
            })
    with open(new_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# ------------------------------
# HF 参数类
# ------------------------------
@dataclass
class ModelArgs:
    model_name_or_path: str = field(default="Qwen/Qwen1.5-0.5B")
    trust_remote_code: bool = True
    use_bf16: bool = True
    use_fp16: bool = False
    gradient_checkpointing: bool = True


@dataclass
class DataArgs:
    train_jsonl: str = field(default="./train.jsonl")
    eval_jsonl: Optional[str] = field(default=None)
    work_dir: str = field(default="./output")
    max_length: int = field(default=2048)
    seed: int = field(default=42)


@dataclass
class LoRAArgs:
    use_lora: bool = True
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    # 适配 Qwen/Llama 常见命名；如需精细控制可自行调整
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


# ------------------------------
# 组装训练数据（聊天模板）
# ------------------------------
def build_chat_example(tokenizer, instr: str, user_inp: str, answer: str):
    sys = "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型"
    # 统一走 chat template，兼容 Qwen 等模型
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"{user_inp}"},
        {"role": "assistant", "content": f"{answer}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, add_special_tokens=False)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    # 把 system+user 之前的 token mask 掉（label=-100），只训练 assistant 段
    # 这里用简化策略：再次构建到 assistant 前的 prompt，并计算长度用于 mask
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

    return input_ids, attn_mask, labels


def build_hf_dataset(tokenizer, jsonl_path: str, max_len: int):
    df = pd.read_json(jsonl_path, lines=True)
    hf_ds = Dataset.from_pandas(df)

    def _map_fn(example):
        inp_ids, attn, labels = build_chat_example(
            tokenizer,
            instr=example["instruction"],
            user_inp=example["input"],
            answer=example["output"],
        )
        # 截断
        inp_ids = inp_ids[:max_len]
        attn = attn[:max_len]
        labels = labels[:max_len]
        return {"input_ids": inp_ids, "attention_mask": attn, "labels": labels}

    return hf_ds.map(_map_fn, remove_columns=hf_ds.column_names)


# ------------------------------
# 训练主流程
# ------------------------------
def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArguments, LoRAArgs))
    model_args, data_args, train_args, lora_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.work_dir, exist_ok=True)
    set_seed(data_args.seed)

    # 1) 如目标 new_* 文件不存在，则自动转换
    new_train = os.path.join(data_args.work_dir, "train.instruct.jsonl")
    if not os.path.exists(new_train):
        dataset_jsonl_transfer(data_args.train_jsonl, new_train)
    new_eval = None
    if data_args.eval_jsonl:
        new_eval = os.path.join(data_args.work_dir, "eval.instruct.jsonl")
        if not os.path.exists(new_eval):
            dataset_jsonl_transfer(data_args.eval_jsonl, new_eval)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=False
    )
    # 处理 pad token
    if tokenizer.pad_token_id is None:
        # Qwen 常用 eod_id 作为 pad
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, "eod_id"):
            tokenizer.pad_token_id = tokenizer.eod_id

    # 3) 模型（精度与梯度检查点）
    torch_dtype = None
    if model_args.use_bf16:
        torch_dtype = torch.bfloat16
    elif model_args.use_fp16:
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if model_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # 允许输入梯度以配合 checkpoint
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # 4) LoRA（可关可开）
    if lora_args.use_lora:
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_args.target_modules,
            inference_mode=False,
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            lora_dropout=lora_args.dropout,
        )
        # 非量化 LoRA，无需 prepare_model_for_kbit_training；如果是 QLoRA 再启用
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    # 5) 数据集
    train_ds = build_hf_dataset(tokenizer, new_train, data_args.max_length)
    eval_ds = build_hf_dataset(tokenizer, new_eval, data_args.max_length) if new_eval else None

    # 6) DataCollator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # 7) Trainer（torchrun 自动 DDP；传 --deepspeed 则自动 DeepSpeed）
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_state()
    # 保存（LoRA 时保存 adapter；全量时保存权重）
    trainer.save_model(train_args.output_dir)


if __name__ == "__main__":
    main()

