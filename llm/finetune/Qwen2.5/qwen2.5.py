#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/8/13 17:26
@source from: 
"""
#!/usr/bin/env python
# coding=utf-8
"""
Qwen2.5-1.5B-Instruct LoRA 微调（NER Few-shot，带 SwanLab 日志）
- 单机单卡：python finetune_qwen25_ner_lora.py
- 单机多卡：torchrun --nproc_per_node=4 finetune_qwen25_ner_lora.py
- 多机多卡：torchrun --nnodes=N --node_rank=R --nproc_per_node=K --master_addr=... --master_port=... finetune_qwen25_ner_lora.py
- DeepSpeed：在上述命令行最后加 --deepspeed ds_config_zero2.json
"""

import os
import json
import argparse
from typing import Dict, List, Optional

import torch
import pandas as pd
from datasets import Dataset

# 仅用于“下载到本地”，加载仍用 transformers
from modelscope import snapshot_download

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

# SwanLab：只在 rank0 初始化与记录
try:
    import swanlab
    from swanlab.integration.huggingface import SwanLabCallback
    _has_swanlab = True
except Exception:
    _has_swanlab = False


# =========================
# 数据转换：旧 jsonl -> 新 jsonl（instruction/input/output）
# =========================
def dataset_jsonl_transfer(origin_path: str, new_path: str) -> None:
    """
    将原始数据集转换为大模型微调所需格式（instruction/input/output）
    期望旧的每行 json 至少包含：
      - text: 原文
      - entities: 实体列表，每个元素包含 entity_text, entity_label
    """
    messages: List[Dict] = []
    with open(origin_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            input_text = data["text"]
            entities = data.get("entities", [])

            if len(entities) == 0:
                entity_sentence = "没有找到任何实体"
            else:
                # 每个实体写成一行 JSON（保证是合法 JSON）
                rows = []
                for ent in entities:
                    ent = dict(ent)
                    rows.append({
                        "entity_text": ent.get("entity_text", ""),
                        "entity_label": ent.get("entity_label", ""),
                    })
                # 输出为多行 JSON 拼接的纯文本（按你原脚本风格）
                entity_sentence = "\n".join([json.dumps(r, ensure_ascii=False) for r in rows])

            instruction = (
                "你是一个文本实体识别领域的专家，你需要从给定的句子中提取"
                " mic / dru / pro / ite / dis / sym / equ / bod / dep 这些实体。"
                ' 以 JSON 格式逐行输出，如 {"entity_text":"房室结消融","entity_label":"procedure"}。\n'
                '注意：1) 每一行必须是正确的 JSON；2) 找不到任何实体时，输出"没有找到任何实体"。'
            )

            messages.append({
                "instruction": instruction,
                "input": input_text,
                "output": entity_sentence,
            })

    with open(new_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# =========================
# ChatML 样本构造（Qwen 风格）
# =========================
def build_chatml_sample(tokenizer, instruction: str, user_input: str, assistant_out: str,
                        max_len: int = 384) -> Dict[str, List[int]]:
    """
    构造符合 Qwen ChatML 的监督训练样本：
      <|im_start|>system\n{instruction}<|im_end|>\n
      <|im_start|>user\n{user_input}<|im_end|>\n
      <|im_start|>assistant\n{assistant_out}<|im_end|>\n
    labels：system/user/assistant 前缀均为 -100，仅 assistant_out 参与 loss
    """
    sys_part = f"<|im_start|>system\n{instruction}<|im_end|>\n"
    user_part = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    asst_prefix = "<|im_start|>assistant\n"
    asst_out = f"{assistant_out}<|im_end|>\n"

    sys_ids = tokenizer(sys_part, add_special_tokens=False)
    user_ids = tokenizer(user_part, add_special_tokens=False)
    asst_pref_ids = tokenizer(asst_prefix, add_special_tokens=False)
    asst_out_ids = tokenizer(asst_out, add_special_tokens=False)

    input_ids = sys_ids["input_ids"] + user_ids["input_ids"] + asst_pref_ids["input_ids"] + asst_out_ids["input_ids"]
    attn_mask = sys_ids["attention_mask"] + user_ids["attention_mask"] + asst_pref_ids["attention_mask"] + asst_out_ids["attention_mask"]

    prefix_len = len(sys_ids["input_ids"]) + len(user_ids["input_ids"]) + len(asst_pref_ids["input_ids"])
    labels = [-100] * prefix_len + asst_out_ids["input_ids"]

    # 截断
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        attn_mask = attn_mask[:max_len]
        labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }


def process_func(example, tokenizer, max_len: int = 384) -> Dict[str, List[int]]:
    return build_chatml_sample(
        tokenizer=tokenizer,
        instruction=example.get("instruction", ""),
        user_input=example.get("input", ""),
        assistant_out=example.get("output", ""),
        max_len=max_len,
    )


# =========================
# 推理（可选快速验证）
# =========================
def predict(messages, model, tokenizer, max_new_tokens: int = 256, device: str = "cuda") -> str:
    """
    messages: [{"role":"system"|"user"|"assistant","content": "..."}]
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        gen = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
    gen = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct", type=str)
    parser.add_argument("--cache_dir", default="./", type=str)
    parser.add_argument("--work_dir", default="./work_qwen25_ner", type=str)

    parser.add_argument("--train_jsonl", default="cmeee.jsonl", type=str)
    parser.add_argument("--take_n", default=2000, type=int)  # 只取前 N 条
    parser.add_argument("--train_ratio", default=0.9, type=float)

    # 训练超参
    parser.add_argument("--output_dir", default="./output/Qwen2.5_NER", type=str)
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--logging_steps", default=5, type=int)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=5, type=int)
    parser.add_argument("--model_max_length", default=384, type=int)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--seed", default=42, type=int)

    # LoRA
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    parser.add_argument("--lora_target_modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # 分布式/DeepSpeed
    parser.add_argument("--deepspeed", default=None, type=str, help="DeepSpeed 配置 JSON 路径，可为空")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", default=False)

    # SwanLab
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--swan_project", default="Qwen2.5-NER-finetune", type=str)
    parser.add_argument("--swan_exp", default="Qwen2.5-1.5B-Instruct", type=str)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 从 ModelScope 拉到本地，再用 transformers 加载（稳定）
    model_dir = snapshot_download(args.model_id, cache_dir=args.cache_dir, revision="master")

    # 2) 加载分词器与模型（训练阶段不要 device_map="auto"）
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    # 如果后续改为 QLoRA，可启用下面一行
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    model = get_peft_model(model, lora_cfg)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    # 3) 处理数据
    new_path = os.path.join(args.work_dir, "cmeee_train.jsonl")
    if not os.path.exists(new_path):
        dataset_jsonl_transfer(args.train_jsonl, new_path)

    total_df = pd.read_json(new_path, lines=True)
    if args.take_n > 0:
        total_df = total_df.iloc[: args.take_n].copy()

    # 划分 train / test
    split = int(len(total_df) * args.train_ratio)
    train_df = total_df.iloc[:split].reset_index(drop=True)
    test_df = total_df.iloc[split:].reset_index(drop=True)

    train_ds_hf = Dataset.from_pandas(train_df)
    train_dataset = train_ds_hf.map(
        lambda ex: process_func(ex, tokenizer, max_len=args.model_max_length),
        remove_columns=train_ds_hf.column_names
    )

    # 4) 训练参数
    hfa = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        evaluation_strategy="no",  # 如需评估可改 "steps" 并提供 eval_dataset
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,

        report_to=["none"],  # SwanLab 通过 callbacks 接入
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        deepspeed=args.deepspeed,  # None 或 ds_config_zero2.json
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    callbacks = []
    # 仅在主进程用 SwanLab
    is_rank0 = (int(os.environ.get("RANK", "0")) == 0)
    if _has_swanlab and args.use_swanlab and is_rank0:
        exp_desc = "Qwen2.5-1.5B-Instruct 在 CMeEE 前 2000 条数据上的 NER LoRA 微调"
        swcb = SwanLabCallback(
            project=args.swan_project,
            experiment_name=args.swan_exp,
            description=exp_desc,
            config={
                "model": args.model_id,
                "dataset": "qgyd2021/few_shot_ner_sft - cmeee.jsonl",
                "take_n": args.take_n,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lr": args.learning_rate,
                "gacc": args.gradient_accumulation_steps,
                "bf16": args.bf16,
                "deepspeed": bool(args.deepspeed),
            },
        )
        callbacks.append(swcb)

    trainer = Trainer(
        model=model,
        args=hfa,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)

    # 5) 简单推理验证（只在 rank0 做）
    if is_rank0 and len(test_df) > 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        small = test_df.sample(n=min(5, len(test_df)), random_state=42)
        pred_logs = []
        for _, row in small.iterrows():
            msg = [
                {"role": "system", "content": row["instruction"]},
                {"role": "user", "content": row["input"]},
            ]
            out = predict(msg, model, tokenizer, device=device)
            pred_logs.append(f"输入: {row['input']}\n输出: {out}")

        if _has_swanlab and args.use_swanlab:
            texts = [swanlab.Text(t) for t in pred_logs]
            swanlab.log({"Prediction": texts})
            swanlab.finish()

        print("\n=== Quick Predictions ===")
        for t in pred_logs:
            print(t)
            print("-" * 60)


if __name__ == "__main__":
    main()