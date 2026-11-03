import json
import os
from dotenv import load_dotenv
from rag_system import build_naive_rag
from datasets import Dataset
from ragas import evaluate
# 引入四大核心指标
from ragas.metrics import faithfulness, context_recall, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
import openai

load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# ================= 魔法补丁：强行修复智谱 temperature 精度报错 =================
_original_create = openai.resources.chat.completions.Completions.create
def _patched_create(*args, **kwargs):
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        kwargs["temperature"] = round(float(kwargs["temperature"]), 2)
    return _original_create(*args, **kwargs)
openai.resources.chat.completions.Completions.create = _patched_create
# ==============================================================================

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

    print(f"正在加载基础 RAG 系统，测试集共 {len(test_data)} 条...")
    query_rag = build_naive_rag()

    questions, answers, contexts, ground_truths = [], [], [], []

    print("开始批量推理...")
    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{i+1}/{len(test_data)}] 正在回答: {q}")
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

    print("\n正在计算 RAGAs 四大核心评估指标 (大概需要 3-5 分钟)...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,       # 忠实度（防幻觉）
            answer_relevancy,   # 答案相关性（防答非所问）
            context_recall,     # 上下文召回率（防漏检）
            context_precision   # 上下文精确率（防检索废话）
        ],
        llm=judge_llm,
        embeddings=wrapped_embeddings
    )

    print("\n" + "="*50)
    print(" 完整版 Baseline 评估报告 (四大天王)")
    print("="*50)
    print(result)

    # 1. 保存为方便看的 CSV
    df = result.to_pandas()
    clean_df = df[['user_input', 'response', 'reference', 'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']]
    clean_df.columns = ['用户问题', '模型回答', '标准答案', '忠实度', '答案相关性', '召回率', '精确率']
    clean_df.to_csv("baseline_result.csv", index=False)
    
# 2. 保存为大厂规范的数据格式 JSON
    # 我们用 pandas 直接算列的平均值，避开新版 ragas 的 bug
    summary_scores = {
        "faithfulness": float(clean_df['忠实度'].mean()),
        "answer_relevancy": float(clean_df['答案相关性'].mean()),
        "context_recall": float(clean_df['召回率'].mean()),
        "context_precision": float(clean_df['精确率'].mean())
    }
    
    final_json = {
        "summary_scores": summary_scores,
        "details": clean_df.to_dict(orient="records")
    }
    with open("baseline_eval.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("\n[OK] 结果已保存至 baseline_result.csv")
    print("[OK] 结果已保存至 baseline_eval.json")

if __name__ == "__main__":
    run_evaluation()