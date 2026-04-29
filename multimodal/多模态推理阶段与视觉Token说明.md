# 多模态推理阶段与视觉 Token 说明

多模态推理和纯文本推理最大的区别是：图片、视频会被转换成视觉 token，并占用上下文窗口。

![多模态推理视觉 Token](./assets/visual-token-inference.svg)

## 一、图片不是免费输入

在 VLM 中，图片通常会经过：

```text
Image
  -> Vision Encoder
  -> Visual Features
  -> Projector / Adapter
  -> Visual Tokens
```

这些 visual tokens 会和文本 token 一起进入 LLM。

```text
总输入 token = 文本 token + 视觉 token
```

## 二、Prefill 为什么更慢

纯文本 prompt 可能只有几十到几百 token。

多模态 prompt 可能是：

```text
文本 token: 100
视觉 token: 1000
```

模型生成第一个 token 前，需要先处理完整上下文，所以视觉 token 会增加首 token 延迟。

## 三、高分辨率图像的成本

图片越大，patch 越多，视觉 token 越多。

动态分辨率模型会保留更多图像细节，但代价是：

- prefill 更慢；
- KV Cache 更大；
- 并发能力下降；
- 多图输入更贵。

## 四、视频输入的成本

视频可以理解为多张图片加时间维度。

```text
video
  -> sampled frames
  -> visual tokens per frame
  -> temporal position encoding
```

如果抽帧太多，token 成本很高；如果抽帧太少，模型可能看不到关键动作。

所以视频模型需要平衡：

```text
帧数
分辨率
视觉 token 数
时间信息
推理成本
```

## 五、KV Cache 也会包含视觉上下文

视觉 token 进入 LLM 后，prefill 阶段会为它们生成 K/V。

后续 decode 时，模型会复用这些 K/V。

这意味着视觉 token 不仅影响计算，也影响显存。

## 六、工程建议

实际部署时可以考虑：

- 不需要细节时降低图片分辨率；
- 文档/OCR 场景保留足够分辨率；
- 多图任务控制图片数量；
- 视频任务控制抽帧数量；
- 对固定图片或固定文档做缓存；
- 根据任务区分普通问答和精细 OCR/定位。

