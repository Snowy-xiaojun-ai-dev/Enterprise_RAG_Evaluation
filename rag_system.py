import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

def build_naive_rag():
    # 1. 读取 data 文件夹下的两份 PDF
    doc1 = PyPDFLoader("data/观演须知.pdf").load()
    doc2 = PyPDFLoader("data/演出介绍.pdf").load()
    docs = doc1 + doc2

    # 2. 朴素切分 (这就是我们要优化的 Baseline 基线)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. 向量化存入 Chroma
    # ⚠️ 请确保这个路径是你电脑上真实的 BGE 模型路径
    embeddings = HuggingFaceEmbeddings(model_name="/Users/liangjiajun/Desktop/capstone_project/bge-small-zh-v1.5") 
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    # 我们多给模型一点材料，配合上面的 Prompt 让它自己挑最核心的。

    # 4. 配置大模型
    llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=ZHIPU_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1
    )

    # 5. 基础 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的购票平台客服。请根据提供的参考资料回答用户问题。\n"
                   "要求：\n"
                   "1. 回答必须简明扼要，直击痛点。\n"
                   "2. 字数不能超过 100 字。\n"
                   "3. 如果资料里包含多条规则，请只提取与用户问题最相关的 1-2 条，严禁长篇大论。\n"
                   "参考资料：\n{context}"),
        ("human", "{question}")
    ])

    def query_rag(question: str):
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        
        return response, [doc.page_content for doc in retrieved_docs]

    return query_rag