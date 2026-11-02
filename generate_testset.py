import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 1. 读取你的两份文档
loader1 = PyPDFLoader("data/观演须知.pdf")
loader2 = PyPDFLoader("data/演出介绍.pdf")
docs = loader1.load() + loader2.load()

# 把文本拼起来（如果文本太长，大模型会记不住，所以我们要切分）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
# 取前几块内容（包含购票规则、入场须知等核心信息的段落）
context_text = "\n\n".join([doc.page_content for doc in splits[:4]])

# 2. 配置大模型（作为出题老师）
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7
)

# 3. 下达出题指令
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个极其严谨的 AI 测试工程师。请根据我提供的演唱会文档，生成 30 个高质量、多样化的测试问答对。\n"
               "要求：\n"
               "1. 问题要像真实用户在提问（可以口语化），覆盖退票、限购、入场、时长、禁带物品等各个方面。\n"
               "2. ground_truth 必须是文档中原原本本的话，不能自己总结，不能编造！\n"
               "3. 必须输出严格的 JSON 数组格式，不要加任何 markdown 代码块符号（不要```json），直接输出纯文本。\n"
               "格式如下：\n"
               '[{{"question": "问题1", "ground_truth": "答案1"}}, {{"question": "问题2", "ground_truth": "答案2"}}]'),
    ("human", "文档内容：\n{text}")
])

chain = prompt | llm

print("正在呼叫 AI 批量生成 30 个测试集，请稍候...")
result = chain.invoke({"text": context_text})

try:
    # 清理可能出现的 markdown 格式标记
    clean_json_str = result.content.strip().replace("```json", "").replace("```", "")
    test_data = json.loads(clean_json_str)
    
    # 保存文件
    with open("test_set.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 成功生成 {len(test_data)} 条测试数据！请打开 test_set.json 人工过目一遍。")
except Exception as e:
    print(f"❌ 解析失败，大模型输出格式可能有误。报错：{e}")
    print("原始输出：", result.content)