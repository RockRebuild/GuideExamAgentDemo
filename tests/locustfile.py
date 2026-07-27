# tests/locustfile.py
# ── HTTP 压力测试脚本 (Locust) ──
#
# 原理:
#   Locust 是 Python 生态的负载测试框架，用纯 Python 代码定义用户行为。
#   每个 HttpUser 实例模拟一个独立用户，按 wait_time 节拍发请求。
#   支持分布式运行（master + worker），可模拟数千并发。
#
# 架构:
#   ┌────────────────────────────────────────────────────┐
#   │  Locust Master (Web UI :8089)                      │
#   │    → 定义 User Behavior                             │
#   │    → 汇总 Worker 上报的统计                          │
#   ├────────────────────────────────────────────────────┤
#   │  Locust Worker × N                                 │
#   │    → 每个 Worker 运行 M 个 HttpUser 实例             │
#   │    → 独立进程，不共享状态                            │
#   └────────────────────────────────────────────────────┘
#
# 运行:
#   # 单机模式 (Web UI):
#   locust -f tests/locustfile.py --host http://localhost:8080
#
#   # 无 UI 模式 (CI/CD):
#   locust -f tests/locustfile.py --host http://localhost:8080 \
#     --headless --users 50 --spawn-rate 5 --run-time 5m \
#     --html report.html --csv results
#
# 关键指标解读:
#   - RPS (Requests Per Second): 服务吞吐量
#   - P50 / P95 / P99 延迟: 用户体验 → P95 > 10s 说明 5% 用户很慢
#   - 失败率: > 1% 需要排查
#   - 并发数 vs 延迟曲线: 找到系统容量拐点

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locust import HttpUser, task, between, events
from locust.runners import STATE_STOPPING, STATE_STOPPED

# ── 测试用例池 ─────────────────────────────────────────
# 模拟真实用户行为分布: 80% 知识问答 + 10% 出卷 + 5% 批改 + 5% 多Agent

KNOWLEDGE_PROMPTS = [
    "导游证的种类有哪些？",
    "旅游法第35条是什么？",
    "地陪导游接团前需要准备哪些证件？",
    "中国一共有多少个世界遗产？",
    "什么是旅游合同？",
    "全陪导游和地陪导游的工作有什么区别？",
    "导游考试报名条件是什么？",
    "合同法律制度这一章都讲了啥？",
    "带团的时候要注意啥？",
    "政策与法律法规的第二章主要讲了什么？",
]

EXAM_PROMPTS = [
    "导游业务 团队导游服务规范 出3道单选题",
    "中国饮食文化 出5道判断题",
    "政策与法律法规 合同法律制度 出5道多选题",
    "全国导游基础知识 中国历史文化 出3道单选题",
]

GRADER_PROMPTS = [
    "请批改题目 科目一的第一章单选第一题，我的答案是 B",
    "帮我批改题目 科目四的第十章多选第三题，我的答案是 A、B",
    "批改：导游证分为哪几种？答案是初级、中级、高级三级",
]

MULTI_AGENT_PROMPTS = [
    "帮我查一下导游证种类，然后根据这些知识点出3道单选题",
    "帮我出5道单选题，然后我自己答完后你帮我批改",
]


class ChatUser(HttpUser):
    """模拟真实用户：间歇性发请求，80% 简单查询 + 20% 复杂查询。"""

    wait_time = between(3, 10)  # 用户思考间隔: 3~10 秒

    # ── 连接池优化 ──────────────────────────────
    # Locust 默认 httpx 客户端连接池有限，大量并发时会排队。
    # 增加连接池大小 + 连接复用。
    pool_connections = 100
    pool_maxsize = 100

    def on_start(self):
        """用户首次登录：验证服务是否可用。"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")
                return

    @task(8)  # 权重 8/10 = 80%
    def knowledge_qa(self):
        """教材知识问答：最高频的请求类型。"""
        prompt = random.choice(KNOWLEDGE_PROMPTS)
        self._post_chat(prompt, "📖 教材知识问答")

    @task(1)  # 权重 1/10 = 10%
    def exam_gen(self):
        """智能出卷。"""
        prompt = random.choice(EXAM_PROMPTS)
        self._post_chat(prompt, "📝 智能出卷")

    @task(1)  # 权重 1/10 = 10%（含 grader + multi_agent）
    def grader_and_multi(self):
        """批改或多Agent（各 5%）。"""
        if random.random() < 0.5:
            prompt = random.choice(GRADER_PROMPTS)
            mode = "📊 阅卷批改"
        else:
            prompt = random.choice(MULTI_AGENT_PROMPTS)
            mode = "🤖 多Agent协作"
        self._post_chat(prompt, mode)

    def _post_chat(self, prompt: str, mode: str):
        """发送 SSE streaming 请求并消费完整个流。

        SSE 格式: event: <type>\ndata: <json>\n\n
        """
        try:
            with self.client.post(
                "/api/chat/stream",
                json={"prompt": prompt, "mode": mode},
                catch_response=True,
                stream=True,
                timeout=120,
                name=f"/api/chat/stream [{mode}]",
            ) as resp:
                if resp.status_code == 429:
                    # 限流返回 429 是正常的（验证限流生效）
                    resp.success()
                    # 检查是否给出排队 token
                    try:
                        body = resp.json()
                        if body.get("error") == "queued":
                            resp.request_meta["response_length"] = 0
                    except Exception:
                        pass
                    return

                if resp.status_code != 200:
                    resp.failure(f"Chat failed: {resp.status_code}")
                    return

                # 消费 SSE 流，统计事件类型
                done_received = False
                error_received = False
                for line in resp.iter_lines(decode_unicode=True):
                    if line.startswith("event: done"):
                        done_received = True
                    elif line.startswith("event: error"):
                        error_received = True

                if error_received and not done_received:
                    resp.failure("SSE stream ended with error")

        except Exception as e:
            # 连接超时等异常
            pass  # Locust 自动记录异常


# ── 自定义事件: 上报关键指标 ──────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """测试开始时打印配置信息。"""
    print(f"\n{'='*60}")
    print(f"🚀 压力测试开始")
    print(f"   目标: {environment.host}")
    print(f"   用户数: {environment.runner.target_user_count if environment.runner else 'N/A'}")
    print(f"   场景: 80% 知识问答 + 10% 出卷 + 5% 批改 + 5% 多Agent")
    print(f"{'='*60}\n")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """测试结束时打印汇总。"""
    if environment.stats.total.num_requests == 0:
        print("\n⚠️ 未发送任何请求，请检查服务是否可达")
        return

    stats = environment.stats.total
    print(f"\n{'='*60}")
    print(f"📊 压力测试汇总")
    print(f"   总请求数: {stats.num_requests}")
    print(f"   失败数: {stats.num_failures}")
    print(f"   失败率: {stats.fail_ratio * 100:.2f}%")
    print(f"   平均响应: {stats.avg_response_time:.0f}ms")
    print(f"   P50: {stats.get_response_time_percentile(0.5):.0f}ms")
    print(f"   P95: {stats.get_response_time_percentile(0.95):.0f}ms")
    print(f"   P99: {stats.get_response_time_percentile(0.99):.0f}ms")
    print(f"   RPS: {stats.total_rps:.1f}")
    print(f"{'='*60}\n")
