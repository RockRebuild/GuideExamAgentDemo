from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# 加载测试数据
data = Dataset.from_json("./ragas_test_data.json")

# 运行评估
result = evaluate(
    data,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)

print("📊 RAGAS 评估结果：")
for metric, score in result.items():
    print(f"  {metric}: {score:.3f}")