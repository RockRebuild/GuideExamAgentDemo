# server/core/concurrency/redis_scripts.py
# ── Redis Lua 脚本（原子操作）──
# 在模块加载时注册，运行时通过 evalsha 调用避免每次传脚本正文。

import logging

logger = logging.getLogger(__name__)

# ── GCRA 速率限制 ──────────────────────────────────────
# 输入 KEY[1]=桶名, ARGV[1]=period(秒), ARGV[2]=burst, ARGV[3]=TTL(秒)
# 返回 {allowed(0|1), remaining, retry_after_seconds}
GCRA_LUA = """
local key = KEYS[1]
local period = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local now = redis.call('TIME')
local now_sec = now[1] + now[2] / 1000000

local tat = tonumber(redis.call('GET', key)) or now_sec
tat = math.max(tat, now_sec)

local allow_at = tat - burst * period
if now_sec >= allow_at then
    local new_tat = math.max(tat, now_sec) + period
    redis.call('SET', key, new_tat, 'PX', ttl * 1000)
    local remaining = math.floor((burst - (new_tat - now_sec) / period))
    if remaining < 0 then remaining = 0 end
    return {1, remaining}
else
    local retry_after = math.ceil(allow_at - now_sec)
    return {0, 0, retry_after}
end
"""

# ── 三层限流联合检查 ───────────────────────────────────
# 依次检查 global_rpm、user_rpm、ip_rpm 三个桶
# KEY[1]=global key, KEY[2]=user key, KEY[3]=ip key
# ARGV[1]=period, ARGV[2]=burst, ARGV[3]=ttl (global)
# ARGV[4]=period, ARGV[5]=burst, ARGV[6]=ttl (user)
# ARGV[7]=period, ARGV[8]=burst, ARGV[9]=ttl (ip)
# 返回 {allowed(0|1), retry_after, blocked_by}
MULTI_CHECK_LUA = """
local function check(key, period, burst, ttl)
    local now = redis.call('TIME')
    local now_sec = now[1] + now[2] / 1000000
    local tat = tonumber(redis.call('GET', key)) or now_sec
    tat = math.max(tat, now_sec)
    local allow_at = tat - burst * period
    if now_sec >= allow_at then
        local new_tat = math.max(tat, now_sec) + period
        redis.call('SET', key, new_tat, 'PX', ttl * 1000)
        return 1, 0
    else
        return 0, math.ceil(allow_at - now_sec)
    end
end

-- Check global
local ok, retry = check(KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]))
if ok == 0 then return {0, retry, 'global'} end

-- Check user (skip if key is empty)
if KEYS[2] ~= '' then
    ok, retry = check(KEYS[2], tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[6]))
    if ok == 0 then
        -- Refund global
        local now = redis.call('TIME')
        local now_sec = now[1] + now[2] / 1000000
        local tat = tonumber(redis.call('GET', KEYS[1])) or now_sec
        redis.call('SET', KEYS[1], math.max(tat - tonumber(ARGV[1]), now_sec), 'PX', tonumber(ARGV[3]) * 1000)
        return {0, retry, 'user'}
    end
end

-- Check IP (skip if key is empty)
if KEYS[3] ~= '' then
    ok, retry = check(KEYS[3], tonumber(ARGV[7]), tonumber(ARGV[8]), tonumber(ARGV[9]))
    if ok == 0 then
        -- Refund global + user
        local now = redis.call('TIME')
        local now_sec = now[1] + now[2] / 1000000
        local tat_g = tonumber(redis.call('GET', KEYS[1])) or now_sec
        redis.call('SET', KEYS[1], math.max(tat_g - tonumber(ARGV[1]), now_sec), 'PX', tonumber(ARGV[3]) * 1000)
        if KEYS[2] ~= '' then
            local tat_u = tonumber(redis.call('GET', KEYS[2])) or now_sec
            redis.call('SET', KEYS[2], math.max(tat_u - tonumber(ARGV[4]), now_sec), 'PX', tonumber(ARGV[6]) * 1000)
        end
        return {0, retry, 'ip'}
    end
end

return {1, 0, 'none'}
"""


def register_scripts(redis_client) -> dict:
    """向 Redis 注册 Lua 脚本，返回 {name: sha} 字典。

    如果 Redis 不可用则返回空字典。
    """
    scripts = {}
    try:
        scripts["gcra"] = redis_client.script_load(GCRA_LUA)
        scripts["multi_check"] = redis_client.script_load(MULTI_CHECK_LUA)
        logger.info("Redis Lua scripts registered: gcra, multi_check")
    except Exception as e:
        logger.warning("Redis Lua script registration failed: %s", e)
    return scripts
