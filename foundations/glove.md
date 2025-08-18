文本预处理
```java
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
```
模型相关参数
```python
# -----------------------
# 4) GloVe parameters & model
# -----------------------
embedding_dim = 2  # 设 2 维便于可视化
x_max = 100.0
alpha = 0.75

```

假设计算出来的共现矩阵为：
```python

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


```
![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove1.png)

条件概率矩阵为：

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove2.png)

Glove三元组表log值：

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove3.png)

下一步glove的优化目标就是：

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove4.png)



拿一个很简单的例子来说明：ice、water、steam
共现矩阵为：
![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove5.png)
glove优化的目标就是使得wi和wj的点积等于log(共现矩阵)，即：

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove6.png)
```java
P(water | ice)  = 0.98
P(water | steam) = 0.89
log比例 ≈ 0.098
```
![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/glove7.png)
意思就是：
在 “water” 这个上下文维度上，ice 和 steam 的向量差应该表现出这种 0.098 的统计差异。