import json
import os
from dotenv import load_dotenv
from rag_system import build_naive_rag
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall 
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

judge_llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.01
)

local_embeddings = HuggingFaceEmbeddings(model_name="/Users/liangjiajun/Desktop/capstone_project/bge-small-zh-v1.5")
wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings=local_embeddings)

def run_evaluation():
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("正在加载基础 RAG 系统...")
    query_rag = build_naive_rag()

    questions, answers, contexts, ground_truths = [], [], [], []

    print("开始批量推理...")
    for item in test_data:
        q = item["question"]
        gt = item["ground_truth"]
        print(f"正在回答: {q}")
        ans, ctx = query_rag(q)
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("正在计算 RAGAs 评估指标...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall],
        llm=judge_llm,
        embeddings=wrapped_embeddings
    )

    print("\n" + "="*50)
    print(" Baseline 评估报告")
    print("="*50)
    print(result)

    df = result.to_pandas()
    
    # 挑选你关心的列，重命名成中文，方便看
    clean_df = df[['user_input', 'response', 'reference', 'faithfulness', 'context_recall']]
    clean_df.columns = ['用户问题', '模型回答', '标准答案', '忠实度', '召回率']
    
    clean_df.to_csv("baseline_result.csv", index=False)
    print("\n[OK] 结果已保存至 baseline_result.csv")

if __name__ == "__main__":
    run_evaluation()