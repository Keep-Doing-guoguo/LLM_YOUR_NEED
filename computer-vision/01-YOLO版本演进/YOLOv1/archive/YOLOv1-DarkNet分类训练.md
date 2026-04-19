# 手把手训练 DarkNet 分类网络（PyTorch 实现）

本文说明如何先训练一个 DarkNet 分类网络，为后续 YOLOv1 检测模型提供预训练 Backbone。

## 1. 网络结构简介

DarkNet 是 YOLO 系列早期使用的主干网络。这里将它作为分类模型训练，结构大致包括：

- 卷积模块：`Conv2d + BatchNorm2d + LeakyReLU`
- 下采样模块：`MaxPool2d`
- 特征提取：交替使用 `1x1` 和 `3x3` 卷积
- 分类头：`AvgPool2d + Squeeze + Linear`

可以通过参数控制是否使用 BatchNorm，以及是否只保留卷积层用于后续检测任务。

```python
class DarkNet(nn.Module):
    ...
    def _make_fc_layers(self):
        return nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, num_classes)
        )
```

## 2. 准备训练流程

### 2.1 定义模型与优化器

```python
model = DarkNet(conv_only=False, bn=True, init_weight=True)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

### 2.2 加载数据

下面以 CIFAR10 为例，实际训练时也可以换成自定义 `ImageFolder` 数据集。

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    batch_size=64,
    shuffle=False
)
```

### 2.3 开始训练

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/darknet_cifar10")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    acc = correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)

writer.close()
```

## 3. 验证准确率

```python
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total
```

## 4. 使用 TensorBoard 查看训练过程

```bash
tensorboard --logdir=runs
```

默认访问地址：

```text
http://localhost:6006
```

## 5. 总结

- 可以自定义 DarkNet 网络结构，并控制是否使用 BatchNorm。
- 可以先完成分类任务训练，再把卷积层权重迁移到 YOLOv1 检测模型。
- TensorBoard 可以用于观察 loss 和 accuracy 曲线。
- 后续可将该流程扩展到自定义分类数据集或 YOLO Backbone 预训练。
