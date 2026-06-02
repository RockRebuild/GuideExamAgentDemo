import asyncio
import time
async def async_search(query: str) -> str:
    await asyncio.sleep(2)
    return f"关于'{query}'的教材片段"

async def batch_search(queries: list[str]) -> list[str]:
    result_list = [async_search(a) for a in queries]
    return await asyncio.gather(*result_list)

async def main():
    start = time.time()
    print(f"开始时间'{start}'")

    query_list = ["地陪接团流程", "全陪导游职责", "旅游法第35条", "再试几个", "三个太少了"]
    result_list = await batch_search(query_list)
    end = time.time()
    print(f"结束时间'{end}'")
    during = end - start
    print(f"耗时'{during}'")
    print(result_list)

asyncio.run(main())