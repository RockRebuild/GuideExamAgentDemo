from fastapi import FastAPI

app = FastAPI(title="导游考试 Agent API")

@app.get("/")
def root():
    return {"message": "导游考试 AI Agent 已就绪"}

@app.get("/health")
def health_check():
    return {"status": "ok"}