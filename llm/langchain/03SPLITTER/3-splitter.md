

🧩 LangChain 常见文本分块器（Text Splitters）

文档加载（Loader）之后的第一步，就是将长文档拆成合适的段落（chunks），
这样每一段都能独立生成向量，提升语义检索效果。



### 1.🧠 什么是 TextSplitter？

TextSplitter 是 LangChain 中的抽象类，用于 按一定规则拆分文档文本。
拆分的目标：
	•	控制每个文本块的长度；
	•	避免句子被截断；
	•	提高嵌入（Embedding）和检索（Retrieval）的语义质量。

所有 Splitter 的输出都是：

```
from langchain.docstore.document import Document

Document(page_content="分块后的文本", metadata={})


```

| Splitter 名称 | 功能说明 | 推荐场景 |
|------------------------------|-------------------------------------------|------------------------------|
| RecursiveCharacterTextSplitter | 递归按字符、标点、段落拆分（最常用） | 通用文本拆分（中英文都可） |
| CharacterTextSplitter | 简单按字符或换行符拆分 | 日志 / 文本行分隔 |
| MarkdownHeaderTextSplitter | 按 Markdown 标题层级拆分 | Markdown、技术文档 |
| SpacyTextSplitter | 基于 NLP 句法结构拆分 | 中文 / 英文自然语言句子 |
| TokenTextSplitter | 按 token 数量拆分（兼容 tokenizer） | 精确控制 token 长度（LLM 输入） |



### 🧩 1️⃣ RecursiveCharacterTextSplitter —— 最推荐的通用拆分器

🔹 从段落 → 句子 → 单词逐级递归分割；
🔹 保证语义完整的同时控制块大小。

✅ 示例代码
```
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

text = "LangChain 是一个用于构建基于大语言模型（LLM）的应用框架..."
chunks = splitter.split_text(text)

print(f"分成 {len(chunks)} 块：")
print(chunks[:2])

这个是是用splitter切分后的结果：
[
Document(page_content='新乡工程学院（原河南科技学院新科学院）是经教育部批准设立的全日制普通本科高等学校。学校地处中原名城新乡市，现有南、北和大学科技园三个校区，规划占地面积2200亩，总建筑面积110万平方米。学校设有教学学院12个，涵盖经济学、管理学、法学、文学、理学、工学、农学、艺术学、教育学等九大学科门类。其中，新一轮河南省重点学科3个，河南省“综合改革试点专业”2个，“河南省一流本科专业建设点”2个，建设产教融合类专业建设点3个，“河南省民办普通高等学校专业建设资助项目”5个，获批省级工程（技术）研究中心和市' metadata={'source': '/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/knowledge_base/test.txt'}),

Document(page_content='合类专业建设点3个，“河南省民办普通高等学校专业建设资助项目”5个，获批省级工程（技术）研究中心和市厅级科技创新平台9个，“河南省特色产业、行业学院”2个。学校获批河南省硕士学位授予单位立项建设单位，先后荣获“河南省‘五好’基层党组织”“河南省先进基层党组织”“河南省优秀民办高校”“河南省高等教育教学工作先进集体”“河南省民办教育先进单位”“河南省五四红旗团委”“博鳌亚洲论坛年会志愿者最佳合作单位”“新乡市全面深入实施‘十大战略’先进集体”“河南省5A级社会组织”等荣誉称号。' metadata={'source': '/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/knowledge_base/test.txt'})
] 

```
📘 特点
	•	默认按段落、换行符、标点等优先级递归切分；
	•	适合大部分中文 / 英文文本；
	•	通用性最强。



### 🧩 2️⃣ CharacterTextSplitter —— 简单字符拆分器

🔹 按固定字符长度拆分，支持自定义分隔符。
🔹 不做语义切分，逻辑最简单。

✅ 示例代码
```
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separator="\n"  # 按换行符拆
)

text = "第一行\n第二行\n第三行"
chunks = splitter.split_text(text)
print(chunks)
```
📘 特点

	•	简单快速；
	•	适合结构化文本（如日志）；
	•	不建议用于自然语言长文。



### 🧩 3️⃣ MarkdownHeaderTextSplitter —— 按标题层级拆分 Markdown

🔹 特别适合技术文档、笔记类知识；
🔹 能自动识别标题层次（#、##、###）。

✅ 示例代码
```
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "一级标题"),
        ("##", "二级标题"),
    ]
)

markdown_text = """
# LangChain 简介
LangChain 是一个强大的 LLM 应用框架。

## 组件
- Prompt
- LLM
- Memory
"""
docs = splitter.split_text(markdown_text)
print(f"共 {len(docs)} 段：", docs[0].page_content)
```
📘 特点

	•	自动根据标题分层；
	•	对结构化 Markdown 支持极好；
	•	拆分后保留标题上下文。



### 🧩 4️⃣ SpacyTextSplitter —— 语言感知拆分器

🔹 使用 NLP 工具 spaCy 分句；
🔹 能按语义边界（句号、连词）智能分割。

✅ 示例代码
```
from langchain.text_splitter import SpacyTextSplitter

splitter = SpacyTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    pipeline="zh_core_web_sm"  # 中文模型
)

text = "今天是个好日子，我们学习LangChain的TextSplitter。它非常强大。"
chunks = splitter.split_text(text)
print(chunks)
```
📘 特点
```
	•	依赖 spacy 语言模型（需下载：python -m spacy download zh_core_web_sm）；
	•	拆分粒度智能；
	•	非常适合自然语言句子分析。



### 🧩 5️⃣ TokenTextSplitter —— 按 Token 数量拆分

🔹 按 tokenizer 计算的 token 长度进行切分；
🔹 能严格控制输入长度（防止超出模型上下文）。

✅ 示例代码
```
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

text = "这是一个非常长的文本，用于测试Token级别的拆分功能..."
chunks = splitter.split_text(text)
print(f"共 {len(chunks)} 块")
```
📘 特点

	•	适合 LLM 输入限制场景；
	•	精准按 token 拆分；
	•	常用于生成 prompt 前的预处理。



实战建议

| 场景 | 推荐 Splitter |
|------|-----------------------------|
| 通用文本（中文 / 英文） | RecursiveCharacterTextSplitter |
| 结构化文本（日志 / 表格） | CharacterTextSplitter |
| Markdown 技术文档 | MarkdownHeaderTextSplitter |
| 中文自然语言句子 | SpacyTextSplitter |
| 精准控制 Token 长度 | TokenTextSplitter |

