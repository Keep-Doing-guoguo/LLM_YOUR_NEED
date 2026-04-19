# 🕒 RT‑DETR 系列发展时间顺序

RT‑DETR 系列是由百度提出的一系列基于 Transformer 的实时端到端目标检测器，以下列出了从提出到演化的主要 milestone：

---

## 时间线概览

|版本|时间|主要改进 / 特点|
|-------------|--------------|------------------|
|**DETR**|2020–05（论文）  [oai_citation:0‡labellerr.com](https://www.labellerr.com/blog/rt-detr-the-real-time-end-to-end-object-detector/?utm_source=chatgpt.com) [oai_citation:1‡arxiv.org](https://arxiv.org/abs/2005.12872?utm_source=chatgpt.com)|Transformer 架构首次用于端到端检测，无需 NMS|
|**RT‑DETR**|2023–04（CVPR′24 接收） ()|实时 DETR，混合 Encoder + IoU-aware 初始化，RTX‑T4 上可达 108 FPS（R50/53.1 AP）|
|**RT‑DETRv2**|2024–07（arXiv 发布） ()|优化训练策略（Bag‑of‑Freebies）、解耦多尺度、部署友好改动|
|**RT‑DETRv3**|2024–09（arXiv 发布） ()|引入 Dense Positive Supervision、Self‑Attention Perturbation，增强训练密度，R18 AP 提升约 1.6 %|

---

## 版本亮点解析

1. **DETR（2020）**
   突破传统两阶段结构，提出 bipartite matching + transformer decoder，实现 NMS-free 结构  [oai_citation:2‡arxiv.org](https://arxiv.org/abs/2005.12872?utm_source=chatgpt.com)。

2. **RT‑DETR（2023/2024）**
   - 混合 Encoder：高效处理多尺度特征
   - IoU-aware query selection：提高初始化质量
   - 实时解码：RTX‑T4 上 RT‑DETR‑R50 达 108 FPS、53.1 AP；R101 可达 74 FPS、54.3 AP  [oai_citation:3‡arxiv.org](https://arxiv.org/abs/2304.08069?utm_source=chatgpt.com)。

3. **RT‑DETRv2（2024‑07）**
   - Bag‑of‑Freebies：训练增强策略
   - 解耦多尺度采样、离散采样优化，提升训练与部署效率  [oai_citation:4‡developers.arcgis.com](https://developers.arcgis.com/python/latest/guide/rt-detrv2-object-detector/?utm_source=chatgpt.com)。

4. **RT‑DETRv3（2024‑09）**
   - Dense Positive Supervision：CNN + Transformer 双分支监督
   - Attention Perturbation：提升 query 表达能力
   - 性能提升明显（如 R18 AP 提升 1.6%）  [oai_citation:5‡arxiv.org](https://arxiv.org/abs/2409.08475?utm_source=chatgpt.com) [oai_citation:6‡openaccess.thecvf.com](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.pdf?utm_source=chatgpt.com)。

---

## 总结

RT‑DETR 代表了 Transformer 检测器向实时性能发展的趋势，通过一系列工程和训练优化，实现了比主流 YOLO 系列更优的速度–精度平衡。最新的 RT‑DETRv3 在训练统一性上继续深入，是当前端到端检测的先进方向。
