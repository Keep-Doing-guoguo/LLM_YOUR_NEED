

按时间顺序详解 YOLO 各版本发布时间与核心改进点

## 一、YOLOv1（2016 CVPR）

### 发布时间：
- 2016 年，CVPR 论文：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

### 改进点：

|内容|说明|
|------|------|
|单阶段检测|第一个统一建模的目标检测模型|
|实时性高|GPU 上可运行到 45 FPS|
|缺陷明显|小目标识别差、定位不准、召回率低|

---

## 二、YOLOv2 / YOLO9000（2017 CVPR）

### 发布时间：
- 2017 年，CVPR 论文：[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

### 改进点：

|内容|说明|
|------|------|
|引入 Anchor Boxes|提升边界框预测质量|
|多尺度训练|输入尺寸随机变化，提升泛化能力|
|Darknet-19 主干网络|更轻量级结构|
|联合训练 COCO + ImageNet|实现 9000+ 类别检测（YOLO9000）|

---

## 三、YOLOv3（2018 ArXiv）

### 发布时间：
- 2018 年，ArXiv 预印本：[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

### 改进点：

|内容|说明|
|------|------|
|多尺度预测|输出 P3/P4/P5 三层边界框|
|更深主干网络|Darknet-53|
|PANet 特征融合|增强小目标识别能力|
|支持更多类别|COCO 数据集下表现稳定|
|未引入新标签分配机制|仍使用传统 IoU 最大匹配方式|

---

## 四、YOLOv4（2020 ArXiv）

### 发布时间：
- 2020 年，ArXiv 论文：[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- 作者：AlexeyAB（非原 YOLO 作者 Joseph Redmon）

### 改进点：

|内容|说明|
|------|------|
|CSPDarknet53 主干网络|提升梯度传播效率|
|PANet Neck 结构|替代 FPN，增强特征融合|
|Mosaic 数据增强|提升小目标识别能力|
|CIoU Loss|提升边界框回归精度|
|DIoU-NMS|提升密集目标后处理效果|
|SAT 自对抗训练|提升鲁棒性（实验性质）|

> 注：YOLOv4 是第一个引入 CIoU、DIoU-NMS 的 YOLO 模型。

---

## 五、YOLOv5（2020 Ultralytics 开源）

### 发布时间：
- 2020 年底由 [Ultralytics](https://github.com/ultralytics/yolov5) 开源；
- 没有正式论文发表；

### 改进点：

|内容|说明|
|------|------|
|Decoupled Head|解耦定位、分类、置信度分支|
|自动 anchor 聚类|`--autoanchor` 参数适配数据集|
|ONNX / TensorRT 导出支持|工业部署友好|
|Mosaic + Copy-Paste 增强|提升训练泛化能力|
|SimOTA 动态标签分配|yolov5m+/l/x 中启用|
|不同大小模型|yolov5n/s/m/l/x（参数从 ~0.9M 到 ~71.6M）|

> 注：SimOTA 来自 YOLOX，不是 Ultralytics 原创。

---

## 六、YOLOv6（2022 Meituan 开源）

### 发布时间：
- 2022 年中，美团视觉计算组开源：[YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)

### 改进点：

|内容|说明|
|------|------|
|RepVGG 主干网络|推理时重参数化为单路径卷积|
|EfficientRep Neck|轻量化 PANet 变体|
|SimOTA 标签分配|在 yolov6m+/x 中启用|
|DFL Loss 支持|分布式边界框回归|
|工业优化|更适合边缘设备部署|
|官方提供 ONNX 导出|支持 OpenVINO / TensorRT|

---

## 七、YOLOv7（2022 AlexeyAB 开源）

### 发布时间：
- 2022 年初，Alexey Bochkovskiy 团队后续贡献者；
- GitHub 地址：[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

### 改进点：

|内容|说明|
|------|------|
|扩展高效结构|包括 E-ELAN、Model集成（如 YOLOv7-tiny / YOLOv7-W6/W6/W6e）|
|模型集成|Auxiliary head + Model ensemble|
|标签动态分配|使用 Extend Efficient Assignment（EAT）|
|边界框回归|使用 CIoU Loss|
|支持 Re-parameterization|推理阶段合并多分支结构|
|支持重参数化模块|RepConv、ReOrg 等|

> 注：YOLOv7 没有正式论文，但其技术细节在官方文档和代码中真实存在。

---

## 八、YOLOv8（2023 Ultralytics 正式发布）

### 发布时间：
- 2023 年初由 Ultralytics 官方发布；
- GitHub 地址：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### 支持功能：

|内容|说明|
|------|------|
|Task-Aligned Assigner（TAL）|替代 SimOTA，结合分类与回归质量选择正样本|
|DFL Loss|分布式边界框回归|
|Decoupled Head|reg/obj/cls 分支分离|
|Anchor-Free 支持（部分版本）|如 YOLOv8-pose|
|新增任务支持|支持图像分割、姿态估计等|
|默认使用 TAL + DFL|提升 mAP 和推理速度|

> 注：YOLOv8 没有正式论文，但提供了完整的文档和源码支持。

---

## 九、YOLO-NAS（2023 Hailo 开源）

### 发布时间：
- 2023 年中，Hailo 科技公司提出；
- GitHub 地址：[Deci AI YOLO-NAS](https://github.com/Deci-AI/yolo-nas)

### 改进点：

|内容|说明|
|------|------|
|AutoNAC 架构搜索|使用 NAS 自动设计网络结构|
|显式优化部署性能|更适合边缘设备部署|
|支持多任务|分类、检测、分割统一接口|
|支持知识蒸馏|提升小型号模型性能|

> 注意：YOLO-NAS 不属于 Ultralytics 或 AlexeyAB/YOLOv6 的范畴，是另一个独立项目。

---

## 十、YOLOv9（2024 论文版本）

### 发布时间：
- 2024 年，论文：[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

### 改进点：

|内容|说明|
|------|------|
|PGI 可编程梯度信息|缓解深层网络训练中的信息丢失|
|GELAN 结构|结合 ELAN 与梯度路径设计|
|保持实时检测|继续服务 YOLO-style 单阶段检测|
|兼顾轻量与精度|提供不同规模模型|

---

## 十一、YOLOv10（2024 THU-MIG）

### 发布时间：
- 2024 年，论文：[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)

### 改进点：

|内容|说明|
|------|------|
|NMS-free|减少传统 YOLO 对 NMS 后处理的依赖|
|一致双分配|one-to-many 训练监督 + one-to-one 推理输出|
|端到端实时检测|兼顾 YOLO 速度和端到端输出|
|效率驱动设计|优化结构中的冗余计算|

---

## 十二、YOLO11（2024 Ultralytics）

### 发布时间：
- 2024 年由 Ultralytics 发布；
- 官方命名为 YOLO11，不是 YOLOv11。

### 改进点：

|内容|说明|
|------|------|
|延续 YOLOv8 工程体系|anchor-free、TAL、DFL、解耦头|
|模型结构继续优化|提升速度与精度平衡|
|多任务模型族|detect、segment、pose、classify、OBB|
|部署工具链成熟|继承 Ultralytics ONNX、TensorRT 等导出能力|

---

## 十三、YOLOv12（2025 Attention-Centric 论文）

### 发布时间：
- 2025 年，论文：[YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)

### 改进点：

|内容|说明|
|------|------|
|Attention-Centric|围绕高效 attention 重新设计实时检测器|
|增强全局上下文|补足卷积局部建模不足|
|保持 YOLO 实时范式|不是 DETR 式检测器|
|anchor-free 检测|延续现代 YOLO 预测方式|

---

## 十四、YOLOv13（2025 Hypergraph 论文）

### 发布时间：
- 2025 年，论文：[YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception](https://arxiv.org/abs/2506.17733)

### 改进点：

|内容|说明|
|------|------|
|超图增强视觉感知|建模多个区域之间的高阶关系|
|自适应视觉感知|根据输入特征动态增强上下文表达|
|复杂场景友好|面向遮挡、密集目标和复杂背景|
|保持实时检测目标|仍服务 YOLO-style 检测框架|

---

## 十五、YOLO 各版本发布顺序与主要改进汇总表

|模型|发布年份|是否有论文|主要改进点|
|------|------------|----------------|--------------------|
|YOLOv1|2016|是（CVPR）|统一建模、实时检测|
|YOLOv2|2017|是（CVPR）|Anchor Boxes、多尺度训练、Darknet-19|
|YOLOv3|2018|是（ArXiv）|多尺度预测、PANet、Darknet-53|
|YOLOv4|2020|是（ArXiv）|CSPDarknet53、CIoU Loss、DIoU-NMS|
|YOLOv5|2020|否（无正式论文）|SimOTA（部分版本）、Decoupled Head、ONNX 支持|
|YOLOv6|2022|是（ArXiv）|RepVGG 主干、SimOTA、DFL Loss|
|YOLOv7|2022|否（无正式论文）|E-ELAN、模型集成、Extend Assigner|
|YOLOv8|2023|否（无正式论文）|TAL 标签分配、DFL Loss、多任务支持|
|YOLOv9|2024|是（ArXiv）|PGI、GELAN、梯度路径优化|
|YOLOv10|2024|是（ArXiv）|NMS-free、一致双分配、端到端实时检测|
|YOLO11|2024|否（官方工程版本）|Ultralytics 多任务模型族升级|
|YOLOv12|2025|是（ArXiv）|Attention-centric 实时检测|
|YOLOv13|2025|是（ArXiv）|超图增强自适应视觉感知|

---

## 十六、YOLO 系列演进图示（文字版）

```
YOLOv1 → YOLOv2 → YOLOv3 → YOLOv4 → YOLOv5 → YOLOv6 → YOLOv7 → YOLOv8 → YOLOv9 → YOLOv10 → YOLO11 → YOLOv12 → YOLOv13
     ↓          ↓           ↓            ↓             ↓              ↓             ↓               ↓          ↓           ↓          ↓          ↓          ↓
单阶段      Anchor Box    多尺度预测   CIoU Loss   SimOTA 分配   RepVGG 主干   模型集成       TAL+DFL    PGI/GELAN   NMS-free  工程升级   Attention  Hypergraph
```

---

## 十七、结语

YOLO 系列的发展经历了以下几个重要阶段：

|阶段|内容|
|--------|------|
|YOLOv1~v3|统一建模 + 多尺度预测|
|YOLOv4|引入 CIoU、DIoU-NMS、CSPDarknet|
|YOLOv5|工业级部署 + SimOTA（大模型）|
|YOLOv6|RepVGG 主干 + DFL Loss|
|YOLOv7|扩展结构 + 模型集成|
|YOLOv8|TAL 标签分配 + DFL + 多任务支持|
|YOLOv9|PGI + GELAN，强化梯度路径与特征学习|
|YOLOv10|一致双分配 + NMS-free 实时端到端检测|
|YOLO11|Ultralytics 工程化多任务模型族升级|
|YOLOv12|高效 attention 引入实时 YOLO 检测|
|YOLOv13|超图高阶关系建模增强视觉感知|

掌握这些版本的核心改进点，有助于你理解现代目标检测框架的设计理念，并为进一步调优打下基础。

---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
