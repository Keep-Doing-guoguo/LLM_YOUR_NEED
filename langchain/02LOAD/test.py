from langchain.document_loaders import UnstructuredFileLoader

# 传入任意文件路径
loader = UnstructuredFileLoader("../data/example.txt")
docs = loader.load()

print(f"共加载 {len(docs)} 段文档")
print(docs)  # 打印前200字
print(docs[0].page_content[:200])  # 打印前200字
