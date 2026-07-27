# server/core/executor.py
# ── 全局异步执行器管理 ──
#
# 原理:
#   LangGraph Agent 的 invoke/stream 是同步阻塞调用（底层调 OpenAI sync client）。
#   FastAPI 是 async 框架。在 async handler 中直接调同步 blocking 操作会冻结
#   事件循环（block the event loop），导致整个服务无法处理其他请求。
#
#   解决方案: run_in_executor() 将同步调用卸载到线程池。
#   但 agent_service.py 当前每请求创建一个 ThreadPoolExecutor(max_workers=1)，
#   存在两个问题:
#     1. 线程泄漏 — 异常路径下 shutdown() 可能被跳过
#     2. 线程膨胀 — 并发请求多时创建大量线程，上下文切换开销大
#
#   本模块提供一个全局共享线程池，并统一管理生命周期。
#
# 线程池大小公式:
#   max_workers = local_max_concurrency + 4 (预留 buffer)
#   默认: 5 + 4 = 9
#   这意味着最多同时执行 5 个 Agent 调用 + 4 个 RAGAS 评估/其他同步任务。
#
# asyncio.run() 嵌套问题:
#   tools.py 中 MCP 工具的 closure 调用了 asyncio.run(_call())。
#   当 ReAct Agent 在 async 路径中调用工具时，事件循环已存在，
#   asyncio.run() 会失败（RuntimeError: asyncio.run() cannot be called from
#   a running event loop）。
#
#   修复: 用本模块的 run_async_in_sync() 函数替代 asyncio.run()，
#   它可以安全地在事件循环内/外调用 async 函数。

import asyncio
import concurrent.futures
import logging
import os
import threading
from typing import TypeVar, Callable, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── 全局线程池 ─────────────────────────────────────────

_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_max_workers() -> int:
    """从环境变量计算线程池大小。"""
    default = int(os.environ.get("CONCURRENCY_LOCAL_MAX_CONCURRENCY", "5")) + 4
    return int(os.environ.get("EXECUTOR_MAX_WORKERS", str(default)))


def get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """获取全局共享线程池（懒初始化，线程安全）。"""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                max_workers = _get_max_workers()
                _executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="agent-worker",
                )
                logger.info(
                    "Global ThreadPoolExecutor initialized: max_workers=%d",
                    max_workers,
                )
    return _executor


def shutdown_executor(wait: bool = True, timeout: float = 30):
    """关闭全局线程池（在 FastAPI shutdown 时调用）。"""
    global _executor
    if _executor is not None:
        logger.info(
            "Shutting down ThreadPoolExecutor (wait=%s, timeout=%.0fs)",
            wait, timeout,
        )
        _executor.shutdown(wait=wait, cancel_futures=not wait)
        _executor = None


# ── Sync/Async 桥接 ────────────────────────────────────

async def run_in_executor(fn: Callable[..., T], *args, **kwargs) -> T:
    """在全局线程池中执行同步阻塞函数，不冻结事件循环。

    用法（替代 asyncio.run / 直接调用）:
        result = await run_in_executor(agent.invoke, messages, config=config)
        chunk = await run_in_executor(next, stream_generator)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()

    def _wrapper():
        return fn(*args, **kwargs)

    return await loop.run_in_executor(executor, _wrapper)


def run_async_in_sync(coro: Awaitable[T], timeout: float = 15) -> T:
    """在同步代码中安全地执行异步协程（替代 asyncio.run）。

    处理两种场景:
    1. 不在事件循环内 → 创建新循环执行（标准 asyncio.run 行为）
    2. 在事件循环内 → 在新线程中创建独立循环执行（避免嵌套冲突）

    用法（替代 tools.py 中的 asyncio.run(_call())）:
        result = run_async_in_sync(_call(), timeout=10)
    """
    try:
        # 场景 1：不在事件循环内 → 直接 asyncio.run
        asyncio.get_running_loop()
    except RuntimeError:
        # 不在事件循环内，安全
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))

    # 场景 2：在事件循环内 → 新线程 + 独立事件循环
    result_box = []
    error_box = []

    def _run_in_new_loop():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_box.append(loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=timeout)
                ))
            finally:
                loop.close()
        except Exception as e:
            error_box.append(e)

    thread = threading.Thread(target=_run_in_new_loop, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 2)

    if thread.is_alive():
        raise TimeoutError(f"Async operation timed out after {timeout}s")

    if error_box:
        raise error_box[0]

    return result_box[0] if result_box else None


# ── Queue-based 流式桥接 ────────────────────────────────

async def stream_sync_to_async(
    sync_generator_func: Callable,
    *args,
    queue_size: int = 100,
    **kwargs,
):
    """通用的同步流 → 异步流转换器。

    将同步生成器（如 agent.stream()）包装为 async generator，
    使用 asyncio.Queue 在专用线程和事件循环之间传递数据。

    这替代了 agent_service.py 中手写的 ThreadPoolExecutor +
    loop.call_soon_threadsafe 模式。

    Yields:
        与原始同步生成器相同的数据块。

    Raises:
        同步生成器中的异常会在异步侧抛出。
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    executor = get_executor()
    loop = asyncio.get_running_loop()

    def _run_sync():
        try:
            for item in sync_generator_func(*args, **kwargs):
                # put_nowait 可能因队列满而失败 → 用小超时轮询
                future = asyncio.run_coroutine_threadsafe(
                    queue.put(("item", item)), loop
                )
                future.result(timeout=30)
            asyncio.run_coroutine_threadsafe(
                queue.put(("done", None)), loop
            )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", e)), loop
            )

    executor.submit(_run_sync)

    while True:
        kind, value = await queue.get()
        if kind == "done":
            break
        if kind == "error":
            raise value
        yield value
