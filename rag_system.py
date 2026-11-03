import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import jieba

load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

def build_naive_rag():
    # 1. 读取与切分
    docs = PyPDFLoader("data/观演须知.pdf").load() + PyPDFLoader("data/演出介绍.pdf").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    text_chunks = [doc.page_content for doc in splits]

    # 2. 构建双路检索器
    # 路 1：向量检索（Dense）
    embeddings = HuggingFaceEmbeddings(model_name="/Users/liangjiajun/Desktop/capstone_project/bge-small-zh-v1.5") 
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 路 2：BM25 关键词检索（Sparse）
    tokenized_corpus = [list(jieba.cut(text)) for text in text_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. 加载 Reranker（交叉编码器）
    reranker = CrossEncoder("/Users/liangjiajun/Desktop/capstone_project/bge-reranker-base")

    # 4. 大模型与极严厉的 Prompt（防复读、防废话）
    llm = ChatOpenAI(
        model="glm-4-flash", api_key=ZHIPU_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/", temperature=0.1
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个严谨的票务客服。请严格根据资料回答问题。\n"
                   "要求：\n"
                   "1. 字数不超过 100 字。\n"
                   "2. 严禁重复相同的句子或词语！\n"
                   "3. 如果资料包含多条相似信息，请用一句话合并总结。\n"
                   "资料：\n{context}"),
        ("human", "{question}")
    ])

    def query_rag(question: str):
        # 【第一层：双路召回】
        vector_docs = vector_retriever.invoke(question)
        
        tokenized_query = list(jieba.cut(question))
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        bm25_docs = [splits[i] for i in top_bm25_indices]

        # 【第二层：RRF 融合算法（核心手写代码）】
        # RRF 得分 = 1 / (k + 排名)，k 取 60 是业界黄金准则，能平滑两路召回的不同分布
        rrf_scores = {}
        for rank, doc in enumerate(vector_docs):
            rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (60 + rank + 1)
        for rank, doc in enumerate(bm25_docs):
            rrf_scores[doc.page_content] = rrf_scores.get(doc.page_content, 0) + 1 / (60 + rank + 1)
            
        # 拿到合并后的 Top 10
        combined_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # 【第三层：Reranker 精排】
        pairs = [[question, text] for text, score in combined_docs]
        rerank_scores = reranker.predict(pairs)
        
        # 选出精排后的 Top 3
        scored_docs = sorted(zip(rerank_scores, [text for text, score in combined_docs]), key=lambda x: x[0], reverse=True)
        top_3_texts = [text for score, text in scored_docs[:3]]

        # 【第四层：生成】
        context = "\n\n".join(top_3_texts)
        response = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
        
        return response, top_3_texts

    return query_rag