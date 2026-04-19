#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/8/13 17:05
@source from: 
"""
#!/usr/bin/env python
# coding=utf-8
"""
Qwen LoRA 微调（单机单卡 / 单机多卡 / 多机多卡 / 可选 DeepSpeed）
- 默认 HuggingFace 加载；可选用 modelscope 拉取到本地后再加载
- 训练阶段不使用 device_map，统一交给 DDP/DeepSpeed 管理设备
- 保留你原有的 ChatML 模板处理逻辑
"""
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pandas as pd
import torch
from datasets import Dataset

# 可选：需要用到 modelscope 拉模型时再导入
try:
    from modelscope import snapshot_download as ms_snapshot_download
    _has_modelscope = True
except Exception:
    _has_modelscope = False

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,  # QLoRA 时使用
)


# -----------------------------
# 数据工具：把旧 jsonl 转成 instruction/input/output
# -----------------------------
def dataset_jsonl_transfer(origin_path: str, new_path: str):
    """
    将原始数据集转换为 instruction/input/output 的新 JSONL
    期望旧格式每行包含：
      - text: 文本
      - category: 备选类别
      - output: 正确类别
    """
    messages = []
    with open(origin_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            context = data["text"]
            category = data["category"]
            label = data["output"]
            messages.append({
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本:{context},类型选项:{category}",
                "output": label,
            })

    with open(new_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# -----------------------------
# ChatML -> 模型监督训练样本构造
# -----------------------------
def build_chatml_sample(tokenizer, instruction: str, user_input: str, assistant_out: str,
                        max_len: int = 384) -> Dict[str, List[int]]:
    """
    构造 ChatML 序列，并生成 input_ids / attention_mask / labels
      system: 给定指令
      user:   输入
      assistant: 标签（可训练）
    """
    sys_prompt = (
        "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，"
        "请输出文本内容的正确类型"
    )
    # 如果外部传了更具体的 instruction，就以外部为准
    if instruction and instruction.strip():
        sys_prompt = instruction.strip()

    # ChatML 拼接（Qwen2 风格）
    # 注意：这里不依赖 tokenizer.apply_chat_template，手动构造更直观可控
    sys_part = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
    user_part = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    asst_prefix = "<|im_start|>assistant\n"

    # 编码
    sys_ids = tokenizer(sys_part, add_special_tokens=False)
    user_ids = tokenizer(user_part, add_special_tokens=False)
    asst_pref_ids = tokenizer(asst_prefix, add_special_tokens=False)
    asst_out_ids = tokenizer(assistant_out, add_special_tokens=False)

    input_ids = sys_ids["input_ids"] + user_ids["input_ids"] + asst_pref_ids["input_ids"] + asst_out_ids["input_ids"] + [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id]
    attn_mask = sys_ids["attention_mask"] + user_ids["attention_mask"] + asst_pref_ids["attention_mask"] + asst_out_ids["attention_mask"] + [1]

    # labels：系统+用户+assistant前缀不参与 loss（置为 -100），assistant_out 参与 loss
    labels = (
        [-100] * (len(sys_ids["input_ids"]) + len(user_ids["input_ids"]) + len(asst_pref_ids["input_ids"]))
        + asst_out_ids["input_ids"]
        + [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id]
    )

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


def process_func(example, tokenizer, max_len: int = 384):
    return build_chatml_sample(
        tokenizer=tokenizer,
        instruction=example.get("instruction", ""),
        user_input=example.get("input", ""),
        assistant_out=example.get("output", ""),
        max_len=max_len,
    )


# -----------------------------
# 评测（推理）辅助
# -----------------------------
def predict(messages, model, tokenizer, max_new_tokens: int = 256, device: str = "cuda"):
    """
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        gen_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
    # 只取新增的
    gen_trim = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, gen_ids)]
    return tokenizer.batch_decode(gen_trim, skip_special_tokens=True)[0]


# -----------------------------
# CLI 配置
# -----------------------------
@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="Qwen/Qwen2-1.5B-Instruct")
    # 如果用 modelscope 先拉到本地，再从本地加载模型/分词器
    use_modelscope: bool = field(default=False)
    modelscope_repo: str = field(default="qwen/Qwen2-1.5B-Instruct")
    modelscope_cache_dir: str = field(default="./qwen_ms_cache")

    train_jsonl: str = field(default="train.jsonl")
    test_jsonl: str = field(default="test.jsonl")
    work_dir: str = field(default="./work")
    out_dir: str = field(default="./output/Qwen2-1_5B-Instruct-LoRA")

    # 训练超参
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=5)
    model_max_length: int = field(default=384)
    bf16: bool = field(default=True)   # A100/4090 建议 bf16；否则用 fp16
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    seed: int = field(default=42)

    # 分布式/DeepSpeed
    deepspeed_config: Optional[str] = field(default=None)   # 例如 "finetune/ds_config_zero2.json"
    ddp_find_unused_parameters: bool = field(default=False)

    # LoRA 超参
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


def main():
    parser = argparse.ArgumentParser()
    # 让 argparse 能接住 dataclass 默认参数
    for k, v in ScriptArgs().__dict__.items():
        arg_type = type(v)
        if arg_type is bool:
            # bool 用 action 更友好
            parser.add_argument(f"--{k}", default=v, action="store_true" if v is False else "store_false")
        else:
            parser.add_argument(f"--{k}", default=v, type=arg_type)
    args_ns = parser.parse_args()
    args = ScriptArgs(**vars(args_ns))

    set_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 可选：用 modelscope 把模型拉到本地，再用 transformers 从本地加载
    model_path = args.model_name_or_path
    if args.use_modelscope:
        if not _has_modelscope:
            raise RuntimeError("未安装 modelscope，请先 `pip install modelscope` 或把 --use_modelscope 关掉。")
        local_dir = ms_snapshot_download(args.modelscope_repo, cache_dir=args.modelscope_cache_dir, revision="master")
        model_path = local_dir

    # 加载分词器 / 模型（训练阶段不要 device_map）
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # 对 CausalLM，pad_token 常用 eos_token 兜底
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    # LoRA 配置
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )

    # 若后续切换到 QLoRA，可放开下面这行
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    model = get_peft_model(model, lora_cfg)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ---------- 数据 ----------
    # 若新数据不存在，先转换
    new_train = os.path.join(args.work_dir, "new_train.jsonl")
    new_test = os.path.join(args.work_dir, "new_test.jsonl")
    if not os.path.exists(new_train):
        dataset_jsonl_transfer(args.train_jsonl, new_train)
    if not os.path.exists(new_test):
        dataset_jsonl_transfer(args.test_jsonl, new_test)

    # 加载训练集
    train_df = pd.read_json(new_train, lines=True)
    train_raw = Dataset.from_pandas(train_df)
    train_ds = train_raw.map(
        lambda ex: process_func(ex, tokenizer, max_len=args.model_max_length),
        remove_columns=train_raw.column_names
    )

    # 也可以准备一个小的验证集（这里用同一份/或另给 dev）
    eval_df = pd.read_json(new_test, lines=True)
    eval_raw = Dataset.from_pandas(eval_df)
    eval_ds = eval_raw.map(
        lambda ex: process_func(ex, tokenizer, max_len=args.model_max_length),
        remove_columns=eval_raw.column_names
    )

    # ---------- 训练参数 ----------
    hf_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        evaluation_strategy="no",   # 如需评估可改成 "steps" 并设置 eval_steps
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,

        report_to=["none"],
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        deepspeed=args.deepspeed_config,   # None 或 JSON 路径
        # 为了更稳地跑分布式，避免 Trainer 去管 device_map
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=hf_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=None,  # 若需要评估可改为 eval_ds
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    # 保存 LoRA 适配器权重
    trainer.save_model(args.out_dir)

    # 简单推理验证（可选）
    try:
        sample = eval_df.iloc[0].to_dict()
        msg = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]},
        ]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        out = predict(msg, model, tokenizer, device=device)
        print("\n[Sanity Check] Sample generation:\n", out)
    except Exception as e:
        print("Skip quick generation check:", e)


if __name__ == "__main__":
    main()

'''
单机单卡：

python train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
  --train_jsonl train.jsonl --test_jsonl test.jsonl \
  --out_dir ./output/Qwen2_LoRA

单机多卡（DDP）：（按卡数改 --nproc_per_node）

torchrun --nproc_per_node=4 train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
  --train_jsonl train.jsonl --test_jsonl test.jsonl \
  --out_dir ./output/Qwen2_LoRA \
  --bf16 True --ddp_find_unused_parameters False

单机多卡 + DeepSpeed（推荐 ZeRO-2）：
torchrun --nproc_per_node=4 train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
  --train_jsonl train.jsonl --test_jsonl test.jsonl \
  --out_dir ./output/Qwen2_LoRA \
  --deepspeed_config finetune/ds_config_zero2.json \
  --bf16 True --ddp_find_unused_parameters False
  
  
  
多机多卡（2 机 × 8 卡为例）：

MASTER_ADDR=10.0.0.10 MASTER_PORT=6001 \
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
  --train_jsonl train.jsonl --test_jsonl test.jsonl \
  --out_dir ./output/Qwen2_LoRA \
  --deepspeed_config finetune/ds_config_zero2.json \
  --bf16 True --ddp_find_unused_parameters False
  
MASTER_ADDR=10.0.0.10 MASTER_PORT=6001 \
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
  --train_jsonl train.jsonl --test_jsonl test.jsonl \
  --out_dir ./output/Qwen2_LoRA \
  --deepspeed_config finetune/ds_config_zero2.json \
  --bf16 True --ddp_find_unused_parameters False

小贴士
	•	显存不够：把 per_device_train_batch_size 降低、开 gradient_checkpointing=True、用 ZeRO-2。
	•	QLoRA：如需 4bit 量化微调，改用 bitsandbytes 或 GPTQ 流程，并启用 prepare_model_for_kbit_training。
	•	评估：把 evaluation_strategy="no" 改为 "steps"，并传 eval_dataset=eval_ds。
	•	数据格式：如果你已有 HF chat 格式，也可以改用 tokenizer.apply_chat_template 直接生成输入。
'''