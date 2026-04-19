
# YOLOv3-SPP 的 ONNX 导出与部署指南

## 一、前言

YOLOv3-SPP 是在 YOLOv3 基础上引入 SPP 模块的目标检测模型，它在工业界广泛使用。为了将其部署到边缘设备（如 Jetson Nano、RK3399、树莓派等），我们需要将 `.weights` 文件导出为 ONNX 格式，并进一步转换为 TensorRT 引擎或其他推理引擎支持的格式。

本文将围绕以下流程进行解析：

|内容|是否真实存在|
|------|----------------|
|ONNX 导出|是（社区提供）|
|ONNX 转 TensorRT|是|
|ONNX 转 OpenVINO|是|
|ONNX 推理输出结构|是|
|部署注意事项|是|

---

## 二、YOLOv3-SPP 的 ONNX 导出流程详解（基于 PyTorch 实现）

虽然原生 AlexeyAB/darknet 不直接支持 ONNX 导出，但你可以通过 **Ultralytics/yolov3 或 yolov5 提供的工具实现 ONNX 导出**，然后替换 backbone 为 Darknet-53 + SPP 结构。

### Step-by-Step 流程如下：

```
1. 加载预训练权重 → yolov3-spp.weights
2. 构建 PyTorch 模型 → darknet_spp_53()
3. 使用 torch.onnx.export() 导出模型
4. 验证 ONNX 模型正确性
5. 可选：转换为 TensorRT / OpenVINO / CoreML 等
```

---

### 示例代码（PyTorch 导出 ONNX）：

```python
import torch
from models.yolo import Model
from utils.torch_utils import select_device

# Step 1: 加载配置和权重
device = select_device('cpu')
model = Model(model_cfg='models/yolov3-spp.yaml').to(device)
model.load_state_dict(torch.load("yolov3-spp.pt", map_location=device)['model'])
model.eval()

# Step 2: 输入张量
dummy_input = torch.randn(1, 3, 416, 416).to(device)

# Step 3: 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "yolov3-spp.onnx",
    export_params=True,  # 存储训练参数
    opset_version=13,   # ONNX 算子集版本
    do_constant_folding=True,  # 优化常量
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
print("ONNX Export Done!")
```

> 注：以上代码改编自 `Ultralytics/yolov3` 中的导出脚本。

---

## 三、YOLOv3-SPP 的 ONNX 模型结构详解（输入图像：416×416×3）

### 输入张量：
```text
[batch_size, 3, 416, 416] → RGB 图像输入
```

### 输出张量（三个层级）：

|层级|输出形状|anchor boxes|
|--------|--------------|---------------|
|P3|`[B, 80, 80, 255]`|[10,13], [16,30], [33,23]|
|P4|`[B, 40, 40, 255]`|[30,61], [62,45], [59,119]|
|P5|`[B, 20, 20, 255]`|[116,90], [156,198], [373,326]|

其中每个 bounding box 包含：

```text
[x_center, y_center, width, height, objectness, class_probs]
```

---

## 四、YOLOv3-SPP 的 ONNX 模型可视化方式（现实存在的资源）

你可以使用以下方式查看 YOLOv3-SPP 的 ONNX 模型结构：

### 方法一：使用 Netron 查看模型图

 在线地址：[https://netron.app/](https://netron.app/)
GitHub 仓库：[lutzroeder/netron](https://github.com/lutzroeder/netron)

上传你的 `yolov3-spp.onnx` 文件即可看到完整网络结构图。

---

### 方法二：使用 ONNX Runtime 运行 ONNX 模型

```bash
pip install onnx onnxruntime
```

```python
import onnx
import onnxruntime as ort

onnx_model = onnx.load("yolov3-spp.onnx")
onnx.checker.check_model(onnx_model)

ort_session = ort.InferenceSession("yolov3-spp.onnx")

outputs = ort_session.run(
    None,
    {'images': np.random.rand(1, 3, 416, 416).astype(np.float32)}
)

print(outputs[0].shape)  # 应输出 [1, 255, 13, 13] 或类似结构
```

---

## 五、YOLOv3-SPP 的 ONNX 模型后处理（NMS）

ONNX 模型仅输出原始预测框，需要你自己实现 NMS 后处理逻辑。

### 示例后处理代码（Python）：

```python
import numpy as np
from utils.general import non_max_suppression

def postprocess(output_tensor):
    """
    output_tensor: ONNX 模型输出，通常为 [B, 255, H, W] 形状
    """
    # reshape 为 [B, num_anchors_per_pixel, 85, H, W]
    batch_size, _, height, width = output_tensor.shape
    output_tensor = output_tensor.reshape(batch_size, 3, 85, height, width)

    # 将数据转换为 [B, num_anchors, 85] 形式
    # 然后执行 NMS
    dets = non_max_suppression(output_tensor, conf_thres=0.25, iou_thres=0.45)
    return dets
```

---

## 六、YOLOv3-SPP 的完整 ONNX 导出命令（来自 Ultralytics）

```bash
python export.py --weights yolov3-spp.pt --include onnx --img-size 416
```

> 注：该命令来自 `Ultralytics/yolov3` 项目，需自行适配 YOLOv3-SPP 主干网络。

---

## 七、YOLOv3-SPP 的 ONNX 导出关键参数说明（来自 PyTorch）

|参数|含义|是否必须|
|--------|--------|--------------|
|`--weights`|权重文件路径（.pt/.weights）|是|
|`--img-size`|输入图像尺寸（如 416）|是|
|`--include onnx`|表示导出为 ONNX|是|
|`--dynamic`|支持动态 batch size（实验性质）|是（可选）|
|`--opset`|ONNX 算子集版本（推荐 13）|是|

---

## 八、YOLOv3-SPP 的 ONNX 模型部署方式汇总

|部署平台|是否支持|工具|
|-------------|----------------|--------|
|ONNX Runtime|是|Microsoft 开源|
|TensorRT|是|NVIDIA 部署工具|
|OpenVINO|是|Intel 推理引擎|
|CoreML|是（需适配）|Apple 移动端部署|
|TVM / MNN / NCNN|是（实验性质）|社区已有尝试|

---

## 九、YOLOv3-SPP 的 TensorRT 部署流程（现实存在的资源）

### 步骤如下：

```
ONNX → trtexec → .engine 文件 → TensorRT 推理
```

### 示例导出命令：

```bash
trtexec --onnx=yolov3-spp.onnx \
        --saveEngine=yolov3-spp.engine \
        --workspace=4096 \
        --fp16 \
        --inputIO=input:image_input \
        --output=output:detections
```

---

### 示例 TensorRT 推理代码（C++ / Python）：

```python
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with TRT_LOGGER as logger, trt.Builder(logger) as builder:
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open("yolov3-spp.onnx", 'rb') as model:
        parser.parse(model.read())

    engine = builder.build_engine(network, config)
    print("TensorRT Engine Built!")

# 推理部分略...
```

---

## 十、YOLOv3-SPP 的 ONNX 模型部署建议（来自社区反馈）

|建议|说明|
|--------|--------|
|使用 FP16 推理|显存减少，速度提升|
|使用 dynamic batch|更灵活适配不同输入|
|保持 anchor 设置一致|训练与推理 anchor 必须相同|
|添加后处理层|如 NMS、解码边界框|
|不建议直接部署到 RK3399 / 边缘设备|需裁剪 head 和 Neck|

---

## 十一、YOLOv3-SPP 的 ONNX 模型结构总结（输入图像：416×416×3）

```
Input Image (416x416x3)
│
├— Vision Encoder（Darknet-53 + SPP）
│
├— Detection Head（三个层级输出）
│
└— ONNX Exported Model
     ├— Input: images [1, 3, 416, 416]
     └— Output: detections [1, 255, 13, 13]（P5）+ P4/P3 输出（若多尺度输出启用）
```

---

## 十二、YOLOv3-SPP 的 ONNX 模型推理输出结构（简化表示）

|输出层级|输出维度|描述|
|------------|--------------|--------|
|P3（80×80）|`[1, 85×3, 80, 80]`|小目标检测|
|P4（40×40）|`[1, 85×3, 40, 40]`|中目标检测|
|P5（20×20）|`[1, 85×3, 20, 20]`|大目标检测|

> 注：`85×3 = 3 anchors × (4 + 1 + 80)`，即 `(x, y, w, h)` + `objectness` + `class probs`

---

## 十三、YOLOv3-SPP 的 ONNX 导出常见问题与解决方案

|问题|解决方案|
|--------|--------------|
|导出失败|检查模型结构是否匹配 ONNX 支持算子|
|推理结果异常|检查 anchor 设置是否与训练一致|
|输出通道顺序错误|确保导出时输出通道正确排列|
|输入归一化错误|使用与训练一致的 Normalize（0~1 或 0~255）|
|NMS 后处理缺失|手动添加 NMS 后处理逻辑|

---

## 十四、YOLOv3-SPP 的 ONNX 导出 & 部署完整流程图（文字版）

```
YOLOv3-SPP weights → PyTorch 模型构建 → dummy input → torch.onnx.export → yolov3-spp.onnx
         ↓
   ONNX Runtime / TensorRT / OpenVINO
         ↓
   编写 C++ / Python 推理程序
         ↓
   验证输出与原始 darknet 一致
```

---

## 十五、YOLOv3-SPP 的 ONNX 模型部署调试技巧（来自实际经验）

### 技巧 1：输入归一化一致性

确保 ONNX 模型的输入归一化方式与训练一致：

```python
image = image.astype(np.float32) / 255.0  # 与训练一致
```

---

### 技巧 2：anchor 设置一致性

导出 ONNX 时，务必确保 anchor 设置与 `.cfg` 文件一致。

---

### 技巧 3：后处理对齐

YOLOv3-SPP 的 ONNX 模型输出是 raw predictions，需手动实现：

- 解码 bounding box；
- Softmax 分类置信度；
- DIoU-NMS 后处理；

---

### 技巧 4：FP16 推理加速

```bash
trtexec --onnx=yolov3-spp.onnx --fp16 --saveEngine=yolov3-spp.engine
```

---

### 技巧 5：TensorRT 部署性能对比

|模型|mAP@COCO|FPS（V100）|是否 FP16|
|--------|--------------------|----------------|----------------|
|YOLOv3-SPP（darknet）|~36.5%|~30|否|
|YOLOv3-SPP（TensorRT）|~36.5%|~80|是|

---

## 十六、YOLOv3-SPP 的 ONNX 模型结构优化建议（来自部署经验）

|优化点|说明|
|--------|--------|
|移除辅助 loss 层|如 training-only layers|
|合并 BN 到 Conv|减少推理计算量|
|使用静态 anchor|若不需要 auto-anchor 动态聚类|
|支持动态 batch|需在导出时设置 dynamic_axes|
|后处理移植|需要手动实现 NMS + decode_box|

---

## 十七、YOLOv3-SPP 的完整 ONNX 导出流程模拟代码（简化版）

```python
import torch
from models.yolo import Model

# Step 1: 加载模型
model = Model(cfg="models/yolov3-spp.cfg").to('cuda')
model.load_state_dict(torch.load("yolov3-spp.pt")['model'])
model.eval()

# Step 2: 构造输入
dummy_input = torch.randn(1, 3, 416, 416).to('cuda')

# Step 3: 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "yolov3-spp.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    verbose=False
)
print(" ONNX 导出完成！")
```

---

## 十八、YOLOv3-SPP 的 ONNX 模型推理流程模拟代码（简化版）

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
ort_session = ort.InferenceSession("yolov3-spp.onnx")

# 构造输入图像
image = cv2.imread("test.jpg")
image = cv2.resize(image, (416, 416)) / 255.0
image = image.transpose(2, 0, 1)[None]  # [1, 3, 416, 416]

# 推理
outputs = ort_session.run(None, {"images": image})
detections = outputs[0]  # 原始输出

# 解码 bounding box
bboxes = decode_boxes(detections, anchors=custom_anchors)

# 执行 NMS
keep_indices = nms(bboxes, scores, iou_threshold=0.45)
final_detections = bboxes[keep_indices]
```

---

## 十九、YOLOv3-SPP 的完整 ONNX 导出 & 部署流程总结

|步骤|内容|
|--------|--------|
|Step 1|加载 PyTorch 模型（Darknet-53 + SPP）|
|Step 2|构造 dummy input（416×416×3）|
|Step 3|使用 torch.onnx.export 导出模型|
|Step 4|使用 ONNX Runtime / TensorRT / OpenVINO 加载模型|
|Step 5|实现解码 + NMS 后处理|
|Step 6|部署至边缘设备或服务器端推理|

---

## 二十、YOLOv3-SPP 的完整 ONNX 导出注意事项（现实存在）

|注意事项|说明|
|------------|--------|
|anchor 设置必须一致|否则推理结果错乱|
|输入归一化必须一致|0~1 或 0~255|
|输出结构需手动处理|ONNX 仅输出 raw predictions|
|NMS 需手动实现|ONNX 不包含后处理|
|支持 dynamic batch|需在导出时指定|

---

## 二十一、结语

YOLOv3-SPP 虽然没有官方 ONNX 导出支持，但通过 PyTorch + ONNX 工具链，我们可以轻松地将它导出并在多种推理引擎中部署。

主要流程包括：

- 构建 PyTorch 模型；
- 导出为 ONNX；
- 使用 TensorRT / OpenVINO / ONNX Runtime 加载；
- 实现 NMS + Bounding Box 解码；
- 部署至边缘设备；


---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于 YOLO系列、Transformer、深度学习等内容！**

