# server/core/llm_provider.py
# ── 多 LLM 供应商 + 自动故障转移 ──
#
# 原理:
#   DeepSeek 作为主供应商，Kimi K3 作为备选。
#   DeepSeek 重试耗尽或熔断打开时，自动切换到 Kimi K3。
#
# 架构:
#   ┌──────────────────────────────────────────────────┐
#   │  get_provider("deepseek") → ChatOpenAI            │
#   │    base_url: https://api.deepseek.com/v1          │
#   │    model: deepseek-v4-flash                        │
#   │    thinking: disabled                              │
#   ├──────────────────────────────────────────────────┤
#   │  get_provider("kimi") → ChatOpenAI                │
#   │    base_url: https://api.moonshot.cn/v1           │
#   │    model: kimi-k3                                  │
#   │    thinking: disabled (Kimi 不支持 thinking)       │
#   ├──────────────────────────────────────────────────┤
#   │  get_primary_llm() → DeepSeek (环境变量可切换)     │
#   │  get_fallback_llm() → Kimi K3 (环境变量可切换)    │
#   └──────────────────────────────────────────────────┘
#
# 环境变量:
#   PRIMARY_PROVIDER=deepseek     (主供应商，默认 deepseek)
#   FALLBACK_PROVIDER=kimi        (备选供应商，默认 kimi)
#   LLM_FAILOVER_ENABLED=true     (是否启用故障转移)
#   DEEPSEEK_API_KEY=sk-xxx
#   DEEPSEEK_MODEL=deepseek-v4-flash
#   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
#   KIMI_API_KEY=sk-xxx
#   KIMI_MODEL=kimi-k3
#   KIMI_BASE_URL=https://api.moonshot.cn/v1

import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ── 供应商配置 ──────────────────────────────────────────

PROVIDER_CONFIG = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_model": "deepseek-v4-flash",
        "default_base_url": "https://api.deepseek.com/v1",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "kimi": {
        # 注意: Kimi 官方 SDK 用 MOONSHOT_API_KEY，此处兼容两个环境变量名
        "api_key_env": "MOONSHOT_API_KEY",
        "api_key_env_fallback": "KIMI_API_KEY",
        "model_env": "KIMI_MODEL",
        "base_url_env": "KIMI_BASE_URL",
        "default_model": "kimi-k3",
        "default_base_url": "https://api.moonshot.cn/v1",
        # Kimi 支持 reasoning_effort（"min"|"low"|"medium"|"high"|"max"），
        # 默认 None 不走思考模式（与 DeepSeek thinking=disabled 对齐）
        # ⚠️ Kimi K3 要求 temperature 必须为 1
        "extra_body": None,
        "temperature_override": 1,
    },
}

PRIMARY_PROVIDER = os.environ.get("PRIMARY_PROVIDER", "deepseek")
FALLBACK_PROVIDER = os.environ.get("FALLBACK_PROVIDER", "kimi")
FAILOVER_ENABLED = os.environ.get("LLM_FAILOVER_ENABLED", "true").lower() == "true"

# 活跃供应商追踪（用于 metrics / 日志）
_active_provider: str = PRIMARY_PROVIDER


def get_provider() -> str:
    """返回当前活跃的供应商名称。"""
    return _active_provider


def set_provider(provider: str):
    """切换活跃供应商（failover 时调用）。"""
    global _active_provider
    old = _active_provider
    _active_provider = provider
    logger.warning(
        "LLM provider switched: %s → %s",
        old, provider,
        extra={"extra_fields": {"old_provider": old, "new_provider": provider}},
    )


def get_llm(provider: str, temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    """根据供应商名创建 ChatOpenAI 实例。

    Args:
        provider: "deepseek" 或 "kimi"
        temperature: 温度参数
        streaming: 是否启用流式输出

    Returns:
        ChatOpenAI 实例

    Raises:
        ValueError: 供应商名未知或 API Key 未配置
    """
    cfg = PROVIDER_CONFIG.get(provider)
    if cfg is None:
        raise ValueError(f"未知的 LLM 供应商: {provider}，可选: {list(PROVIDER_CONFIG.keys())}")

    api_key = os.environ.get(cfg["api_key_env"])
    # 兼容降级：如果主 key 没设置，尝试 fallback key 名
    if not api_key and cfg.get("api_key_env_fallback"):
        api_key = os.environ.get(cfg["api_key_env_fallback"])
    if not api_key:
        raise ValueError(
            f"缺少 {cfg['api_key_env']} 环境变量，"
            f"请在 .env 中配置或设置该环境变量"
        )

    model = os.environ.get(cfg["model_env"], cfg["default_model"])
    base_url = os.environ.get(cfg["base_url_env"], cfg["default_base_url"])

    # 供应商对 temperature 有特殊要求时覆盖
    if "temperature_override" in cfg:
        temperature = cfg["temperature_override"]

    kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "streaming": streaming,
    }

    if cfg["extra_body"]:
        kwargs["extra_body"] = cfg["extra_body"]

    return ChatOpenAI(**kwargs)


def get_primary_llm(temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    """获取主供应商 LLM 实例。"""
    return get_llm(PRIMARY_PROVIDER, temperature=temperature, streaming=streaming)


def get_fallback_llm(temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    """获取备选供应商 LLM 实例。"""
    return get_llm(FALLBACK_PROVIDER, temperature=temperature, streaming=streaming)


def get_active_llm(temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    """获取当前活跃供应商的 LLM 实例。"""
    return get_llm(_active_provider, temperature=temperature, streaming=streaming)


# ── 故障转移 ────────────────────────────────────────────

def failover_to_fallback() -> bool:
    """切换到备选供应商。返回 False 表示备选不可用。"""
    if not FAILOVER_ENABLED:
        logger.warning("LLM failover is disabled via LLM_FAILOVER_ENABLED=false")
        return False

    # 检查备选 API Key 是否已配置
    fallback_cfg = PROVIDER_CONFIG.get(FALLBACK_PROVIDER, {})
    if not os.environ.get(fallback_cfg.get("api_key_env", "")):
        logger.error(
            "无法切换: 备选供应商 %s 未配置 API Key (%s)",
            FALLBACK_PROVIDER, fallback_cfg.get("api_key_env", ""),
        )
        return False

    set_provider(FALLBACK_PROVIDER)
    return True


def reset_to_primary():
    """切回主供应商（熔断恢复时调用）。"""
    set_provider(PRIMARY_PROVIDER)
