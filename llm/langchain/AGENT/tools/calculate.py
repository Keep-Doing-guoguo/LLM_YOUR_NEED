from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain
from langchain_community.chat_models import ChatOpenAI

STRICT_PROMPT = PromptTemplate.from_template(
    # 只生成一个三引号代码块，语言标签必须是 text，且只含一行可执行表达式
    "请将下面的数学问题转换为可由 Python `numexpr` 执行的**单行表达式**。"
    "只输出一个代码块，格式严格为：\n\n"
    "```text\n<单行表达式>\n```\n\n"
    "不要输出其它任何文字。\n\n"
    "问题：{question}"
)

def calculate(query: str):
    model = ChatOpenAI(
        temperature=0,
        openai_api_key="你的key",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus"
    )
    llm_math = LLMMathChain.from_llm(model, verbose=True, prompt=STRICT_PROMPT)
    return llm_math.run(query)

if __name__ == "__main__":
    print("答案:", calculate("2的三次方"))