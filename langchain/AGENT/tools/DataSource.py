#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/14 10:09
@source from: 
"""
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}
问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

class FugeDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def find_product_description(self, product_name: str) -> str:
        """模拟公司产品的数据库"""
        product_info = {
            "好快活": "好快活是一个营销人才平台，以社群+公众号+小程序结合的运营模式展开，帮助企业客户连接并匹配充满才华的营销人才。",
            "Rimix": """Rimix通过采购流程数字化、完备的项目数据存储记录及标准的供应商管理体系，帮助企业实现采购流程, 透明合规可追溯，大幅节约采购成本。Rimix已为包括联合利华，滴滴出行等多家广告主提供服务，平均可为客户节约采购成本30%。""",
            "Bid Agent": "Bid Agent是一款专为中国市场设计的搜索引擎优化管理工具，支持5大搜索引擎。Bid Agent平均为广告主提升18%的投放效果，同时平均提升47%的管理效率。目前已为阳狮广告、GroupM等知名4A公司提供服务与支持。",
        }
        return product_info.get(product_name, "没有找到这个产品")

    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让llm根据抓取信息回答问题"""
        context = """
        关于产品："让广告技术美而温暖"是复歌的产品理念。在努力为企业客户创造价值的同时，也希望让使用复歌产品的每个人都能感受到技术的温度。
        我们关注用户的体验和建议，我们期待我们的产品能够给每个使用者的工作和生活带来正面的改变。
        我们崇尚技术，用科技的力量使工作变得简单，使生活变得更加美好而优雅，是我们的愿景。
        企业文化：复歌是一个非常年轻的团队，公司大部分成员是90后。
        工作上，专业、注重细节、拥抱创新、快速试错。
        协作中，开放、坦诚、包容、还有一点点举重若轻的幽默感。
        以上这些都是复歌团队的重要特质。
        在复歌，每个人可以平等地表达自己的观点和意见，每个人的想法和意愿都会被尊重。
        如果你有理想，并拥有被理想所驱使的自我驱动力，我们期待你的加入。
        """
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        return self.llm(prompt)
llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        api_key="你的key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
fuge_data_source = FugeDataSource(llm)