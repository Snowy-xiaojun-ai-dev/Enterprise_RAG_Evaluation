import asyncio
import aiohttp
import time

# 前提：你的 FastAPI 服务（api_server.py）必须在后台运行
# 你的 FastAPI 接口地址
API_URL = "http://127.0.0.1:8000/chat"

async def send_request(session, question):
    """发送单次请求，并记录首字延迟(TTFT)和总耗时"""
    payload = {"question": question}
    start_time = time.time()
    ttft = None
    
    try:
        # 使用流式请求，为了精准测算首字延迟
        async with session.post(API_URL, json=payload) as response:
            if response.status != 200:
                return False, 0, 0
            
            # 拿到第一个数据块的时间，就是首字延迟
            async for chunk in response.content.iter_any():
                if ttft is None:
                    ttft = time.time() - start_time
                    
            total_time = time.time() - start_time
            return True, ttft, total_time
    except Exception as e:
        print(f"请求失败: {e}")
        return False, 0, 0

async def run_stress_test(concurrent_requests=10):
    """运行并发压测"""
    questions = ["我想请年假，怎么申请？"] * concurrent_requests
    
    print(f"开始并发压测！并发数: {concurrent_requests}")
    print("-" * 40)
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, q) for q in questions]
        # 使用 asyncio.gather 同时发送所有请求
        results = await asyncio.gather(*tasks)
        
    success_count = sum(1 for r in results if r[0])
    ttfts = [r[1] for r in results if r[0]]
    total_times = [r[2] for r in results if r[0]]
    
    if success_count == 0:
        print("所有请求均失败，请确保 FastAPI 服务已启动。")
        return

    # 计算核心指标
    avg_ttft = sum(ttfts) / success_count
    avg_total_time = sum(total_times) / success_count
    # QPS = 成功请求数 / 总耗时（以最慢的那个请求的耗时作为分母，比较保守）
    max_total_time = max(total_times)
    qps = success_count / max_total_time
    
    print("\n压测结果报告")
    print("=" * 40)
    print(f"总请求数: {concurrent_requests}")
    print(f"成功请求数: {success_count}")
    print(f"平均首字延迟 (TTFT): {avg_ttft:.2f} 秒")
    print(f"平均总耗时: {avg_total_time:.2f} 秒")
    print(f"系统极限吞吐量 (QPS): {qps:.2f} req/s")

if __name__ == "__main__":
    # 模拟 10 个用户同时发问
    asyncio.run(run_stress_test(concurrent_requests=10))