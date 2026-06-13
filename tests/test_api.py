
import requests
import os

os.environ["DEEPSEEK_API_KEY"] = "sk-"

url = "https://api.deepseek.com/v1/embeddings"
headers = {
    "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {"model": "deepseek-embedding", "input": ["你好"]}

response = requests.post(url, headers=headers, json=payload)

print("状态码:", response.status_code)
print("响应头:", response.headers.get("Content-Type"))
print("原始响应前500字符:", response.text[:500])