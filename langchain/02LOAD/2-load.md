

LangChain 常见 Loader 



### ① UnstructuredFileLoader —— 通用文件加载器

🔹 适用范围最广，可自动识别 txt / pdf / docx / html / md 等多种格式。
🔹 内部使用 unstructured 库，能自动提取正文、标题、表格、列表等结构化信息。

✅ 示例代码
```
from langchain.document_loaders import UnstructuredFileLoader

# 传入任意文件路径
loader = UnstructuredFileLoader("data/example.txt")
docs = loader.load()

print(f"共加载 {len(docs)} 段文档")
print(docs[0].page_content[:200])  # 打印前200字



这里的数据如果你将text.txt作为输入的话，这个将会是load后的结果输出。

[Document(page_content='新乡工程学院（原河南科技学院新科学院）是经教育部批准设立的全日制普通本科高等学校。学校地处中原名城新乡市，现有南、北和大学科技园三个校区，规划占地面积2200亩，总建筑面积110万平方米。学校设有教学学院12个，涵盖经济学、管理学、法学、文学、理学、工学、农学、艺术学、教育学等九大学科门类。其中，新一轮河南省重点学科3个，河南省“综合改革试点专业”2个，“河南省一流本科专业建设点”2个，建设产教融合类专业建设点3个，“河南省民办普通高等学校专业建设资助项目”5个，获批省级工程（技术）研究中心和市厅级科技创新平台9个，“河南省特色产业、行业学院”2个。学校获批河南省硕士学位授予单位立项建设单位，先后荣获“河南省‘五好’基层党组织”“河南省先进基层党组织”“河南省优秀民办高校”“河南省高等教育教学工作先进集体”“河南省民办教育先进单位”“河南省五四红旗团委”“博鳌亚洲论坛年会志愿者最佳合作单位”“新乡市全面深入实施‘十大战略’先进集体”“河南省5A级社会组织”等荣誉称号。\n\n根据学校发展需要，现面向海内外公开招聘高层次人才。具体事项如下：\n\n一、招聘对象\n\n（一）具有一定学术水平和科研潜力，已取得相应的高水平科研业绩的海内外高校、科研院所毕业的博士研究生；\n\n（二）具有副高及以上职称的专业技术人员，或具有相关企业（行业）丰富工作经验的高级工程师；\n\n（三）荣获省级及以上劳动模范或“五一劳动奖章”称号，或省级技能大师工作室主持人，或其他省级及以上技能大奖、技能能手的高技能人才（高级技师以上）。\n\n二、基本条件\n\n（一）具有中华人民共和国国籍，拥护党和国家的路线、方针、政策，热爱高等教育事业，遵纪守法。\n\n（二）教书育人，为人师表，敬业奉献，身心健康，具有良好的思想品质、职业道德和团队合作精神。\n\n（三）具有扎实的专业理论知识，具备较强的从事教学和应用研究的素质与能力。\n\n（四）第一学历须为统招全日制本科学历，所学专业与招聘专业一致或相近；博士研究生要求硕博士阶段所学专业与招聘专业一致或相近。\n\n（五）原则上博士研究生年龄不超过45周岁（1980年1月1日后出生），高职称和高技能人才年龄不超过50周岁（1975年1月1日后出生）。条件特别优秀人员，年龄可适当放宽。\n\n三、招聘专业\n\n序号\n\n学院\n\n需求专业\n\n招聘类别\n\n联系人\n\n及联系方式\n\n1\n\n生物工程学院\n\n新能源科学与工程、化学、化学工程与工艺、生物工程、生物技术\n\n博士研究生/高职称人员/高技能人员\n\n王老师13623735166\n\n2\n\n食品工程学院\n\n食品相关专业\n\n博士研究生/高职称人员/高技能人员\n\n孙老师13837339744\n\n3\n\n机电工程学院\n\n机械工程、电气工程、电子科学与技术\n\n博士研究生/高职称人员/高技能人员\n\n廖老师15836018283\n\n4\n\n信息工程学院\n\n计算数学，基础数学，运筹学与控制论、通信与信息系统、信号与信息处理、计算机系统结构、计算机软件与理论、计算机应用技术\n\n博士研究生/高职称人员/高技能人员\n\n焦老师15236617461\n\n5\n\n经济与管理学院\n\n管理学、经济学（除国防经济）相关专业\n\n博士研究生/高职称人员/高技能人员\n\n赵老师15903074461\n\n6\n\n外国语学院\n\n英语语言文学、外国语言学及应用语言学\n\n博士研究生/高职称人员/高技能人员\n\n张老师13938767647\n\n7\n\n文法学院\n\n汉语言文学专业各方向、新闻学、汉语国际教育\n\n博士研究生/高职称人员/高技能人员\n\n罗老师13639632294\n\n8\n\n艺术学院\n\n视觉传达设计、环境设计、服装与服饰设计、产品设计、工艺美术\n\n博士研究生/高职称人员/高技能人员\n\n邢老师15993096907\n\n9\n\n马克思主义学院\n\n马克思主义理论、思想政治教育、政治学、哲学、历史学\n\n博士研究生/高职称人员\n\n田老师13598672014\n\n10\n\n体育学院\n\n体育学\n\n博士研究生/高职称人员\n\n杨老师15993086885\n\n其他说明\n\n马克思主义学院教师须为中共党员或中共预备党员；体育学院教师本硕博阶段均为体育相关专业。\n\n四、招聘待遇\n\n（一）博士研究生入校后按内聘副教授或内聘教授（已取得副高级职称）待遇执行。聘期内享受同等职称评审及博士人才“绿色通道”有关政策，对于符合学校初定条件的博士研究生，可直接初定中级职称。其他高级职称人员入校后直接享受同等职称人员待遇；\n\n（二）符合引进博士配偶的条件（全日制本科及以上学历）以及学校工作需要，妥善安置配偶工作；\n\n（三）对于符合学校博士人才引进条件的，服务期内享受博士津贴和相关科研启动费和安家费，具体按照“一人一策、一事一议”有关政策执行；\n\n（四）专业技术人员享受河南省职称评审同等政策，已取得职称者，任职年限按上级有关政策执行；\n\n（五）办理国家规定的“五险一金”，加入工会，享受工会福利待遇；\n\n（六）可选购集团为教职工定向开发的家属区住房，以及享受集团旗下开发的其他商品房的购房优惠；\n\n（七）具有高级职称人员或高层次人才称号的人员，聘期内可享受高职称人才补贴或高层次人才称号补贴；\n\n（八）优先解决子女到附属学校（小学、初中）就读，并享受学费补贴。\n\n五、报名方式及招聘程序\n\n（一）公告发布之日起开始报名，高层次人才全年招聘。报名邮箱为：xgrsc@xxgc.edu.cn。邮件名称统一为：“高层次人才+张三+专业”。\n\n报名时需同时上传以下材料：①《高层次人才信息登记表》；②身份证照片；③本科及以上毕业证、学位证（应届毕业生须提供按期毕业证明）；④海外留学人员应提供教育部留学服务中心出具的《国外学历学位认证书》；⑤职称证/职业技能/专业技术证书；⑥科研业绩证明材料；⑦相关荣誉（获奖）证书；⑧工作项目成果或其他能证明本人能力的相关材料。对于报名材料弄虚作假者，不论何时发现，一经查实，即取消聘用资格。\n\n（二）学校人事部门会同专业所在学院对应聘人员进行资格审核和专业论证，经审核通过后，择优进入面试考核。\n\n（三）面试考核采用线下方式进行，具体面试考核安排请应聘人员及时关注个人邮箱，原则上会在报名后一周内与本人联系。\n\n（四）考核通过后，学校安排体检事宜，体检通过应届毕业生签订预约报到协议，往届毕业生可直接办理录用报到手续。\n\n六、其他说明\n\n窗体顶端\n\n本次招聘报名及面试不收取任何费用，未委托其他单位代为招聘，谨防诈骗。学校纪检监察部门负责整个招聘工作的监督检查，对于弄虚作假或提供材料不真实的应聘人员，取消其应聘资格。如有下列情形之一的不得报名应聘：\n\n（一）曾因犯罪受过刑事处罚的人员；\n\n（二）尚未解除处分或正在接受纪律审查的人员；\n\n（三）涉嫌违法正在接受司法调查尚未做出结论的人员；\n\n（四）不具备教育部、河南省关于教师师德要求的人员；\n\n（五）曾在招聘考试中被认定有舞弊等严重违反招聘纪律行为的人员；\n\n（六）其他不符合招聘有关要求的人员。\n\n七、联系方式\n\n联系人：赵老师/张老师\n\n联系电话：(0373)6330022 15660132390/15137332007\n\n监督电话：（0373）6330018\n\n邮 箱：xgrsc@xxgc.edu.cn\n\n学校官网：http://www.xxgc.edu.cn\n\n到校路线：学校位于新乡市新飞大道南段777号，距新乡市高铁站14公里左右，北临新乡市汽车客运南站，西临107国道，东临京港澳高速。市内可乘坐11路公交车到新乡工程学院下车。自驾者在手机地图搜索“新乡工程学院（南校区）”可直接到达。', metadata={'source': '/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/knowledge_base/test.txt'})]

```
📘 使用场景

	•	最推荐的默认 Loader；
	•	适合大多数通用文本文档；
	•	对编码与文件结构自动识别，几乎无需参数。



### ② PyPDFLoader —— 结构化 PDF 加载器

🔹 基于 PyPDF2 实现；
🔹 能逐页提取文本内容，并保留页码信息。

✅ 示例代码
```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

print(f"文档共 {len(docs)} 页")
print(docs[0].metadata)  # {'source': 'data/sample.pdf', 'page': 0}
print(docs[0].page_content[:200])
```
📘 使用场景

	•	原生数字 PDF（非扫描版）；
	•	希望保留页码、章节等结构信息；
	•	可结合 TextSplitter 分页拆分长文档。



### ③ CSVLoader —— 表格文本加载器

🔹 用于加载 .csv 文件，将每一行转化为 Document；
🔹 自动识别编码，可指定分隔符、字段映射。

✅ 示例代码
```
from langchain.document_loaders import CSVLoader

loader = CSVLoader(file_path="data/finance.csv", encoding="utf-8")
docs = loader.load()

print(f"共加载 {len(docs)} 行数据")
print(docs[0].page_content)

[
 Document(page_content='question: What is LangChain?\nanswer: LangChain is a framework for building LLM-powered applications.\ncategory: AI', metadata={'source': '/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/document_loaders/test.csv', 'row': 0}), 
 Document(page_content='question: What is ChatGPT?\nanswer: ChatGPT is an AI developed by OpenAI.\ncategory: AI', metadata={'source': '/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/document_loaders/test.csv', 'row': 1}), 
 Document(page_content='question: What is Python?\nanswer: Python is a popular programming language.\ncategory: Programming', metadata={'source': '/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/document_loaders/test.csv', 'row': 2})
 ]
```
📘 使用场景

	•	表格类知识文件（如财报、指标表）；
	•	结构化数据转换为自然语言语料；
	•	适合与数值分析任务结合。

⸻

### ④ JSONLoader —— JSON 文件加载器

🔹 用于解析 .json 文件，可通过 jq_schema 提取特定字段。
🔹 可处理层级结构，支持多级嵌套。

✅ 示例代码
```
from langchain.document_loaders import JSONLoader

# jq_schema="." 表示提取整个JSON
loader = JSONLoader(file_path="data/config.json", jq_schema=".", text_content=False)
docs = loader.load()

print(f"共加载 {len(docs)} 条数据")
print(docs[0].page_content)

📘 使用场景

	•	结构化文本（如知识库配置、API响应）；
	•	结合 jq_schema 实现定向提取：

jq_schema=".items[].description"

```



### ⑤ DirectoryLoader —— 文件夹批量加载器

🔹 一次性加载目录中的所有文件；
🔹 支持指定文件后缀、过滤器。

✅ 示例代码

```
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="data/docs",
    glob="**/*.txt"  # 匹配所有 txt 文件
)
docs = loader.load()

print(f"共加载 {len(docs)} 个文件")
print(docs[0].metadata)
```
📘 使用场景

	•	批量加载知识库；
	•	自动扫描整个文件夹；
	•	可配合 RecursiveCharacterTextSplitter 拆分后入库。

### 6️⃣ 自定义Imgloader


```
from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from document_loaders.ocr import get_ocr


class RapidOCRLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def img2text(filepath):
            resp = ""
            ocr = get_ocr()
            result, _ = ocr(filepath)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)
            return resp

        text = img2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRLoader(file_path="/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/knowledge_base/samples/content/llm/img/大模型技术栈-算法与原理-幕布图片-81470-404273.jpg")
    docs = loader.load()
    print(docs)

```

### 7️⃣ 自定义PDFloader

```
from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from document_loaders.ocr import get_ocr
import tqdm


class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(filepath):
            import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):

                # 更新描述
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                # 立即显示进度条更新结果
                b_unit.refresh()
                # TODO: 依据文本与图片顺序调整处理方式
                text = page.get_text("text")
                resp += text + "\n"

                img_list = page.get_images()
                for img in img_list:
                    pix = fitz.Pixmap(doc, img[0])
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRPDFLoader(file_path="/Volumes/PSSD/未命名文件夹/donwload/创建知识库数据库/langchain.pdf")
    docs = loader.load()
    print(docs)

```

### 💡 实战建议

| 场景 | 推荐 Loader |
|------|--------------|
| 通用文本（TXT / DOCX / HTML / MD） | `UnstructuredFileLoader` |
| PDF 文档（数字版） | `PyPDFLoader` |
| 表格数据 | `CSVLoader` |
| 结构化 JSON | `JSONLoader` |
| 批量加载文件夹 | `DirectoryLoader` |

