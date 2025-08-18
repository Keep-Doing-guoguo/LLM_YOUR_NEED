https://blog.csdn.net/SoulmateY/article/details/145287971

# 1.	•	单机单卡
python finetune_qwen25_ner_lora.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --train_jsonl cmeee.jsonl \
  --take_n 2000 \
  --bf16
# 2.	•	单机多卡（DDP）
torchrun --nproc_per_node=4 finetune_qwen25_ner_lora.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --train_jsonl cmeee.jsonl \
  --take_n 2000 \
  --bf16 --ddp_find_unused_parameters


# 3. 	•	单机/多机 + DeepSpeed（ZeRO-2）
torchrun --nproc_per_node=4 finetune_qwen25_ner_lora.py \
  --model_id Qwen/Qwen2.5-1.5B-Instruct \
  --train_jsonl cmeee.jsonl \
  --take_n 2000 \
  --bf16 \
  --deepspeed ds_config_zero2.json

多机时再加：--nnodes N --node_rank R --master_addr IP --master_port 6001

备注与建议

	•	ChatML 模板我已统一成 Qwen 推荐格式；你原来的 <im_sep> 不是标准 token。
	•	SwanLab：我做了主进程保护（只在 RANK==0 初始化&log）。
	•	显存吃紧：可以把 per_device_train_batch_size 降低、保持 gradient_checkpointing=True、再配合 DeepSpeed ZeRO-2。
	•	想改成 QLoRA：加上 prepare_model_for_kbit_training 与 4-bit 量化加载；如需我帮你切到 QLoRA 版，也可以直接给我你目标显卡&显存我来配。