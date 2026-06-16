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
                "description": "查询指定城市的实时天气。参数：city（城市名称，支持中文）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称，例如 北京"}
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
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://wttr.in/{city}?format=%C+%t"
                    resp = await client.get(url, timeout=10)
                    result = f"{city} 当前天气：{resp.text.strip()}"
            except Exception as e:
                result = f"获取天气失败：{str(e)}"
            return JSONResponse({
                "content": [{"type": "text", "text": result}]
            })

    return JSONResponse({"error": "Unknown method"}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)