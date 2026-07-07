from datetime import date, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
app = FastAPI()


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    method = body.get("method")

    if method == "tools/list":
        return JSONResponse({
            "tools": [{
                "name": "get_weather",
                "description": "查询指定城市的天气。支持当天、明天、后天及未来多日预报。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，支持中文或英文，例如 北京、上海、Tokyo"
                        },
                        "days": {
                            "type": "integer",
                            "description": "查询未来几天（含今天）。1=仅今天，3=今明后三天。默认1。"
                        }
                    },
                    "required": ["city"]
                }
            }]
        })

    elif method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name == "get_weather":
            city = arguments.get("city", "")
            days = arguments.get("days", 1)
            try:
                async with httpx.AsyncClient() as client:
                    if days > 1:
                        # JSON 格式获取多日预报
                        url = f"https://wttr.in/{city}?format=j1"
                        resp = await client.get(url, timeout=10)
                        data = resp.json()
                        result = _format_forecast(data, days)
                    else:
                        # 当天简洁格式
                        url = f"https://wttr.in/{city}?format=%C+%t"
                        resp = await client.get(url, timeout=10)
                        result = f"{city} 当前天气：{resp.text.strip()}"
            except Exception as e:
                result = f"获取天气失败：{str(e)}"
            return JSONResponse({
                "content": [{"type": "text", "text": result}]
            })

    return JSONResponse({"error": "Unknown method"}, status_code=400)


def _format_forecast(data: dict, days: int) -> str:
    """将 wttr.in JSON 格式化为可读的天气预报。"""
    lines = []
    today = date.today()

    for d in range(min(days, 3)):  # wttr.in 免费版支持 3 天
        target = today + timedelta(days=d)
        date_str = target.strftime("%Y-%m-%d")
        label = "今天" if d == 0 else "明天" if d == 1 else "后天"

        day_data = data.get("weather", [])[d] if d < len(data.get("weather", [])) else None
        if not day_data:
            continue

        # 最高/最低温度
        maxt = day_data.get("maxtempC", "?")
        mint = day_data.get("mintempC", "?")

        # 白天/夜间天气描述
        hourly = day_data.get("hourly", [])
        desc = "未知"
        for h in hourly:
            wd = h.get("weatherDesc", [{}])
            if wd and wd[0].get("value"):
                desc = wd[0]["value"]
                break

        lines.append(f"{label} {date_str}：{desc}，气温 {mint}°C ~ {maxt}°C")

    return "\n".join(lines) if lines else "未获取到天气预报数据"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)