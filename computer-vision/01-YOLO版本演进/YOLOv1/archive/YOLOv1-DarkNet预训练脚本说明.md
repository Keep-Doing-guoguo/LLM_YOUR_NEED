# YOLOv1 DarkNet 预训练脚本说明

本文整理 `train_darknet.py` 的主要功能，用于说明如何基于 PyTorch 训练 DarkNet 分类模型。

## 1. 主要功能

|功能|说明|
|------|------|
|模型结构|使用自定义 DarkNet，可选择是否启用 BatchNorm|
|数据集|使用 `ImageFolder` 加载自定义分类数据集|
|分布式训练|可基于 `torch.multiprocessing` 启动多进程训练|
|TensorBoard|使用 `SummaryWriter` 记录训练和验证指标|
|Resume|支持从 checkpoint 恢复训练|
|Top-K Accuracy|计算 top-1 和 top-5 分类准确率|
|Checkpoint|保存当前模型和最佳模型 `model_best.pth.tar`|
|参数配置|训练参数通过 argparse 或 namespace 配置|

## 2. 快速运行示例

```bash
python train_darknet.py
```

脚本入口：

```python
if __name__ == '__main__':
    main()
```

典型参数示例：

```python
args = SimpleNamespace(
    data='/models/other_code/acvhhhhh/archive',
    bn=True,
    gpu=0,
    tb_log_dir='debug_run',
    ...
)
```

## 3. TensorBoard 可视化

```bash
tensorboard --logdir=results/darknet/debug_run
```

默认访问地址：

```text
http://localhost:6006
```

常见记录指标包括：

- `train/loss`
- `train/top1`
- `train/top5`
- `test/loss`
- `test/top1`
- `test/top5`
- `lr`

## 4. 数据要求

`args.data` 对应的数据集目录建议使用 `ImageFolder` 结构：

```text
archive/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...
```

如果没有独立验证集，也可以临时将 `valdir` 指向 `train`，但正式训练时不建议这样做。

## 5. 推荐文章结构

如果将该脚本整理成博客，可以按以下顺序展开：

1. DarkNet 简介，以及为什么先训练分类模型。
2. 自定义分类数据集准备，重点讲 `ImageFolder` 目录结构。
3. 训练主流程，说明 PyTorch 官方 ImageNet 示例的借鉴点。
4. 训练参数配置，包括 batch size、learning rate、BatchNorm 等。
5. TensorBoard 可视化训练日志。
6. 模型保存、恢复训练与最佳模型选择。
7. 推理测试与准确率评估。
8. 后续优化方向，例如 AMP、模型导出、更多数据增强。
