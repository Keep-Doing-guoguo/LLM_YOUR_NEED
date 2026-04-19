# 1.参考链接：
[qwen1.5微调](https://blog.csdn.net/Kashiwa123/article/details/139438533)

# 2.单机单卡：

python train.py \
  --model_name_or_path Qwen/Qwen1.5-0.5B \
  --train_jsonl /path/to/old_train.jsonl \
  --output_dir ./output_qwen \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --use_bf16 True \
  --gradient_checkpointing True

# 3.多机多卡：
在两台机器各自执行，替换 MASTER_ADDR 为 rank0 的 IP。

Rank 0


torchrun \
  --nproc_per_node 8 \
  --nnodes 2 \
  --node_rank 0 \
  --master_addr 192.168.1.10 \
  --master_port 6001 \
  train.py \
  --model_name_or_path Qwen/Qwen1.5-0.5B \
  --train_jsonl /path/to/old_train.jsonl \
  --output_dir ./output_qwen \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --use_bf16 True \
  --gradient_checkpointing True

Rank 1

torchrun \
  --nproc_per_node 8 \
  --nnodes 2 \
  --node_rank 1 \
  --master_addr 192.168.1.10 \
  --master_port 6001 \
  train.py \
  --model_name_or_path Qwen/Qwen1.5-0.5B \
  --train_jsonl /path/to/old_train.jsonl \
  --output_dir ./output_qwen \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --use_bf16 True \
  --gradient_checkpointing True


# 4.单机多卡：
torchrun --nproc_per_node 8 train.py \
  --model_name_or_path Qwen/Qwen1.5-0.5B \
  --train_jsonl /path/to/old_train.jsonl \
  --output_dir ./output_qwen \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --use_bf16 True \
  --gradient_checkpointing True


# 5.说明 & 小贴士

	•	DDP 自动化：用 torchrun 启动后，Trainer 会自动切到 DDP；如果你加 --deepspeed ds_config.json，则自动用 Deepspeed（比如 ZeRO-2/3）。
	•	LoRA：默认已开启 --use_lora（见 LoRAArgs.use_lora=True）。如果要全量微调，把启动命令加 --use_lora False 即可。
	•	精度：上面用 --use_bf16 True。如需 FP16 改为 --use_bf16 False --use_fp16 True（并确保硬件/驱动支持）。
	•	数据：脚本会把旧 jsonl（text/category/output）自动转为 instruction 格式并缓存到 --work_dir 下。
	•	有效批大小：global_batch = per_device_train_batch_size × n_gpus × gradient_accumulation_steps，按算力调参。

