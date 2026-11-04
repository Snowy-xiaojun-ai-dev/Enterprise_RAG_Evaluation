import json
import time
import os
from dotenv import load_dotenv
from rag_system import build_naive_rag # 引入你优化好的单次RAG
# 假设你之前的 Agentic 项目里的 graph 已经封装好，且能 import 过来
# 如果没封装，我们先用假数据模拟 Agent 的耗时（后面教你接真 Agent）
# from agent_graph import agent_app

load_dotenv()

def run_ab_test():
    with open("test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 初始化单次 RAG 系统
    print("正在加载单次 RAG (Naive) 引擎...")
    query_naive_rag = build_naive_rag()

    results = []
    
    print(f"开始 A/B 测试，共 {len(test_data)} 题...")
    print("="*50)

    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        diff = item["difficulty"]

        print(f"[{i+1}/{len(test_data)}] 难度: {diff} | 问题: {q}")

        # ==========================================
        # 测试 A：单次 RAG（无 Agent 循环）
        # ==========================================
        start_time = time.time()
        ans_naive, _ = query_naive_rag(q)
        latency_naive = round(time.time() - start_time, 2)
        # 粗略估算 token (按字数算，实际工程中从 API 返回值里取)
        token_naive = len(str(ans_naive)) * 2 

        # ==========================================
        # 测试 B：Agentic 循环（这里为了演示，我们用代码模拟）
        # 真实场景下，你需要调用 LangGraph 跑一遍
        # ==========================================
        start_time = time.time()
        
        # 【模拟逻辑】：Agent 多步思考，耗时和 Token 通常是单次 RAG 的 3-5 倍
        time.sleep(1.5) # 模拟 Agent 思考和多步工具调用的延迟
        if diff == "trap":
            ans_agent = "抱歉，我无法回答该问题。" # Agent 成功反思拒答
        else:
            # 模拟 Agent 经过多步循环，给出了更精准的回答（这里用标准答案代替演示）
            ans_agent = gt 
        
        latency_agent = round(time.time() - start_time, 2)
        token_agent = len(str(ans_agent)) * 2 + 1500 # 增加了大量中间思考的 Token

        # 记录数据
        results.append({
            "difficulty": diff,
            "question": q,
            "naive_answer": ans_naive,
            "naive_latency_sec": latency_naive,
            "naive_token": token_naive,
            "agent_answer": ans_agent,
            "agent_latency_sec": latency_agent,
            "agent_token": token_agent
        })

    # 将结果保存为 JSON
    with open("ab_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(" A/B 测试完成！结果已保存至 ab_test_results.json")
    print("请打开文件，对比 Naive 和 Agent 的延迟与 Token 差距！")

if __name__ == "__main__":
    run_ab_test()