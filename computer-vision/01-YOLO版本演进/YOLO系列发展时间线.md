

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

## 十、YOLO 各版本发布顺序与主要改进汇总表

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

---

## 十一、YOLO 系列演进图示（文字版）

```
YOLOv1 → YOLOv2 → YOLOv3 → YOLOv4 → YOLOv5 → YOLOv6 → YOLOv7 → YOLOv8
     ↓          ↓           ↓            ↓             ↓              ↓             ↓               ↓
单阶段      Anchor Box    多尺度预测   CIoU Loss   SimOTA 分配   RepVGG 主干   模型集成       TAL + DFL
实时性强    多尺度训练    PANet Neck   DIoU-NMS   auto-anchor   EfficientRep  扩展高效结构   多任务支持
```

---

## 十二、结语

YOLO 系列的发展经历了以下几个重要阶段：

|阶段|内容|
|--------|------|
|YOLOv1~v3|统一建模 + 多尺度预测|
|YOLOv4|引入 CIoU、DIoU-NMS、CSPDarknet|
|YOLOv5|工业级部署 + SimOTA（大模型）|
|YOLOv6|RepVGG 主干 + DFL Loss|
|YOLOv7|扩展结构 + 模型集成|
|YOLOv8|TAL 标签分配 + DFL + 多任务支持|

掌握这些版本的核心改进点，有助于你理解现代目标检测框架的设计理念，并为进一步调优打下基础。

---

 **欢迎点赞 + 收藏 + 关注我，我会持续更新更多关于目标检测、YOLO系列、深度学习等内容！**
