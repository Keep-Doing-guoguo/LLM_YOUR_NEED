文本预处理
```java
	sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])
```
模型相关参数
```java
	batch_size = 2 # mini-batch size
    embedding_size = 2 # embedding size
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
```

假设所有文本分词，转为索引之后的 list 如下图所示

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/imgs/img.png)

根据论文所述，我这里设定 window size=2，即每个中心词左右各取 2 个词作为背景词，那么对于上面的 list，窗口每次滑动，选定的中心词和背景词如下图所示

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/imgs/img_1.png)

那么 skip_grams 变量里存的就是中心词和背景词一一配对后的 list，例如中心词 2，有背景词 0,1,0,1，一一配对以后就会产生 [2,0],[2,1],[2,0],[2,1]。skip_grams 如下图所示


![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/imgs/img_2.png)

由于 Word2Vec 的输入是 one-hot 表示，所以我们先构建一个对角全 1 的矩阵，利用 np.eye(rows) 方法，其中的参数 rows 表示全 1 矩阵的行数，对于这个问题来说，语料库中总共有多少个单词，就有多少行

然后根据 skip_grams 每行第一列的值，取出相应全 1 矩阵的行。将这些取出的行，append 到一个 list 中去，最终的这个 list 就是所有的样本 X。标签不需要 one-hot 表示，只需要类别值，所以只用把 skip_grams 中每行的第二列取出来存起来即可

最后第三步就是构建 dataset，然后定义 DataLoader

构建模型



训练


由于我这里每个词是用的 2 维的向量去表示，所以可以将每个词在平面直角坐标系中标记出来，看看各个词之间的距离

[本文参考链接](https://wmathor.com/index.php/archives/1430/)

![](/Volumes/PSSD/NetThink/pythonProject/7-19-Project/before_embedding/imgs/img_3.png)


1️⃣ 模型结构

对于FastText来说
FastText 基于 Word2Vec 框架（Skip-gram / CBOW），但在输入层和输出层加了 subword n-gram 表示。

	•	输入层

一个词不是单独一个向量，而是该词所有 n-gram 子词向量的和/平均。

	•	例如 apple（n=3,4） → {<ap, app, ppl, ple, le>, <app, appl, pple, apple>}
	•	每个 n-gram 在词典中都有一个唯一 ID 和向量
	•	输出层

和 Word2Vec 相同（预测上下文词或类别标签），但输出端依然用的是完整词的向量表示（不是子词）

2️⃣ 核心流程

以 Skip-gram 模式为例：
	1.	子词分解
	•	把输入中心词拆成所有 n-gram 子词
	•	查找每个子词的向量
	2.	向量合成
	•	把这些子词向量求和/平均 → 得到中心词向量
	3.	预测上下文词
	•	用中心词向量预测上下文词（输出层是词向量，而不是子词向量）
	4.	损失计算
	•	用 Negative Sampling 或 Hierarchical Softmax 计算损失
	5.	反向传播
	•	更新子词向量和输出词向量
3️⃣ 总结
FastText 的实现原理核心就是：


	•	输入层 = 子词向量的加和（解决 OOV + 捕捉词形特征）
	•	训练过程 = Word2Vec 架构 + 负采样/层次softmax
	•	输出层 = 完整词向量（或分类标签）
	•	工程优化 = 子词哈希 + 多线程 + 流式训练