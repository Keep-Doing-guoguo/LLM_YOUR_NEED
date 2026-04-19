# YOLO 的推理部署方法全景指南

YOLO 系列模型经过训练后，通常需要部署到线上环境中进行推理（inference）。下面是常见的 YOLO 推理部署方式：

---

## 1⃣ PyTorch 原生部署

- 使用原始 PyTorch 模型 `.pt` 文件
- 直接调用 `model(input)` 进行推理
- 优点：简单、灵活、易于调试
- 缺点：推理速度较慢，不适合生产环境

---

## 2⃣ ONNX 导出 + 推理

- 将 YOLO 模型导出为 `.onnx` 格式
- 使用 ONNX Runtime、TensorRT、OpenVINO 等推理框架部署
- 优点：跨平台、高性能、支持多后端
- 缺点：导出模型需兼容性处理，部分 ops 不支持

---

## 3⃣ TensorRT 部署

- 将 ONNX 模型转换为 `.engine` 文件
- 使用 NVIDIA TensorRT 进行推理加速
- 优点：极致推理速度，适用于边缘设备（如 Jetson）
- 缺点：部署复杂，需特定 GPU 支持

---

## 4⃣ OpenVINO 部署（Intel 平台）

- 将模型转换为 OpenVINO IR 格式
- 用于 CPU、VPU、FPGA 等 Intel 硬件部署
- 优点：适配 Intel 系列硬件
- 缺点：转换过程复杂，ops 支持有限

---

## 5⃣ NCNN/MNN 部署（移动端）

- 将 YOLO 模型转换为适用于移动端的格式
- 支持 Android/iOS 部署
- 优点：轻量级、低功耗
- 缺点：精度和兼容性可能下降

---

## 6⃣ Web 部署（WebAssembly / ONNX.js）

- 可通过 ONNX.js、WebDNN 等方案实现浏览器推理
- 优点：无需安装、轻量级
- 缺点：性能受限，适合小模型

---

## 7⃣ 服务化部署（FastAPI / Flask / Triton）

- 将模型封装成 API 接口，部署为 HTTP 服务
- 优点：便于前后端集成、支持扩展负载
- 缺点：需维护服务器资源

---

## 小结对比

|方法|性能|部署复杂度|适用平台|
|--------------|------|--------------|-----------|
|PyTorch|||通用平台|
|ONNX|||通用平台|
|TensorRT|||NVIDIA GPU|
|OpenVINO|||Intel 设备|
|NCNN/MNN|||移动设备|
|Web|||浏览器|
|API 服务化|||服务端|



