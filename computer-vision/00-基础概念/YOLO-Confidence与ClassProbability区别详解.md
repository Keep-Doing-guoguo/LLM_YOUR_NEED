



# YOLO 中的 Confidence 与 Class Probability 区别详解

## 1. Confidence（置信度）
- 定义：某个预测框 **包含目标的概率 × 预测框与真实框的 IOU（重合程度）**
- 公式：
  $$
  \text{Confidence} = \Pr(\text{object}) \times \text{IOU}_{\text{pred, truth}}
  $$
- 含义：这个框是不是目标框，位置准不准。

## 2. Class Probability（类别概率）
- 定义：在该预测框中，目标属于某个特定类别的概率。
- 多个类别下为 softmax 输出，如：
  $$
  \Pr(\text{class}_i \mid \text{object})
  $$

## 3. Final Score（用于 NMS 的分数）
- 定义：是上述两者的乘积。
- 公式：
  $$
  \text{Score}_{\text{class}_i} = \Pr(\text{class}_i) \times \text{Confidence}
  $$
  （或 YOLOv1 中为：
  $$
  \Pr(\text{object}) \times \text{IOU} \times \Pr(\text{class}_i)
  $$）

---

## 举例说明
假设：
- 模型预测这个框有 80% 的概率含有物体（Pr(object) = 0.8）
- IOU = 0.7
- 对应类别是 "dog"，其类别概率为 0.9

则：
- Confidence = 0.8 × 0.7 = 0.56
- Final score for "dog" = 0.56 × 0.9 = **0.504**

---

## 总结

|名称|含义|位置相关|类别相关|
|------|------|-----------|-----------|
|Confidence|该框含有目标 + 位置是否准确|是|否|
|Class Probability|如果有目标，是哪个类别|否|是|
|Final Score|综合考虑目标有无 + 属于哪个类|是|是|
