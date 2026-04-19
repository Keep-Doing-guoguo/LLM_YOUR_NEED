#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/8/11 18:51
@source from: 
"""
#!/usr/bin/env python
# coding: utf-8
"""
A tiny GloVe implementation in PyTorch (for demo & visualization).
Trains on a toy corpus and plots 2D embeddings.
"""

import math
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt


# -----------------------
# 1) Toy corpus
# -----------------------
sentences = [
    "apple banana fruit",
    "banana orange fruit",
    "orange banana fruit",
    "dog cat animal",
    "cat monkey animal",
    "monkey dog animal",
]
sentences = [s.lower().split() for s in sentences]

# -----------------------
# 2) Build vocab
# -----------------------
word_counts = Counter(w for s in sentences for w in s)
itos = list(word_counts.keys())
stoi = {w: i for i, w in enumerate(itos)}
vocab_size = len(itos)

# -----------------------
# 3) Build co-occurrence matrix X (symmetric within a window)
# -----------------------
window_size = 2  # symmetric context window
X = defaultdict(float)  # (i,j) -> X_ij
for s in sentences:
    ids = [stoi[w] for w in s]
    for center_idx, wi in enumerate(ids):
        # look around within window
        for dist in range(1, window_size + 1):
            left = center_idx - dist
            right = center_idx + dist
            if left >= 0:
                wj = ids[left]
                X[(wi, wj)] += 1.0 / dist  # distance weighting（可选）
                X[(wj, wi)] += 1.0 / dist
            if right < len(ids):
                wj = ids[right]
                X[(wi, wj)] += 1.0 / dist
                X[(wj, wi)] += 1.0 / dist

pairs = [(i, j, x) for (i, j), x in X.items() if x > 0.0]

# -----------------------
# 4) GloVe parameters & model
# -----------------------
embedding_dim = 2  # 设 2 维便于可视化
x_max = 100.0
alpha = 0.75

# 主词向量 w，和上下文词向量 w_tilde，以及偏置 b、b_tilde
w = nn.Embedding(vocab_size, embedding_dim)
w_tilde = nn.Embedding(vocab_size, embedding_dim)
b = nn.Embedding(vocab_size, 1)
b_tilde = nn.Embedding(vocab_size, 1)

# 参数初始化（可选：更稳定）
for emb in [w, w_tilde]:
    nn.init.xavier_uniform_(emb.weight)
for bias in [b, b_tilde]:
    nn.init.zeros_(bias.weight)

params = list(w.parameters()) + list(w_tilde.parameters()) + list(b.parameters()) + list(b_tilde.parameters())
optimizer = optim.Adagrad(params, lr=0.05)  # 经典做法是用 Adagrad

def weight_fn(x):
    # f(x) = (x/x_max)^alpha for x < x_max else 1
    if x < x_max:
        return (x / x_max) ** alpha
    return 1.0

weights = torch.tensor([weight_fn(x) for (_, _, x) in pairs], dtype=torch.float32)
logs = torch.tensor([math.log(x) for (_, _, x) in pairs], dtype=torch.float32)

# -----------------------
# 5) Training loop
# -----------------------
epochs = 2000
for ep in range(1, epochs + 1):
    optimizer.zero_grad()

    i_idx = torch.tensor([i for (i, _, _) in pairs], dtype=torch.long)
    j_idx = torch.tensor([j for (_, j, _) in pairs], dtype=torch.long)

    wi = w(i_idx)            # [N, D]
    wj = w_tilde(j_idx)      # [N, D]
    bi = b(i_idx).squeeze(1)         # [N]
    bj = b_tilde(j_idx).squeeze(1)   # [N]

    # 预测项：w_i^T w_j + b_i + b_j
    pred = (wi * wj).sum(dim=1) + bi + bj
    diff = pred - logs  # [N]

    loss = (weights * (diff ** 2)).sum()

    loss.backward()
    optimizer.step()

    if ep % 400 == 0:
        print(f"Epoch {ep:04d} | loss = {loss.item():.4f}")

# -----------------------
# 6) Get final embeddings (sum is common choice)
# -----------------------
with torch.no_grad():
    final_emb = w.weight + w_tilde.weight  # [V, D]

# -----------------------
# 7) Plot 2D embeddings
# -----------------------
emb_np = final_emb.cpu().numpy()
plt.figure(figsize=(6, 6))
for idx, word in enumerate(itos):
    x, y = emb_np[idx, 0], emb_np[idx, 1]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), xytext=(5, 2), textcoords="offset points")
plt.title("GloVe (2D) on toy corpus")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.show()