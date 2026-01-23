# RAG-Anything 评估系统使用指南

## 📋 目录

1. [快速开始](#快速开始)
2. [创建测试数据集](#创建测试数据集)
3. [运行评估](#运行评估)
4. [自定义指标](#自定义指标)
5. [扩展评估器](#扩展评估器)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 步骤 1: 安装可选依赖

```bash
# 安装基础评估依赖
pip install rouge-score nltk

# 可选：高级语义指标
pip install sentence-transformers scikit-learn

# 可选：BERTScore（计算密集）
pip install bert-score
```

### 步骤 2: 验证系统

```bash
cd /home/ik2200-2025-g2/WorkZone/RAG-Anything
conda activate rag-anything
python examples/evaluation_quick_test.py
```

### 步骤 3: 运行完整评估

```bash
python examples/evaluation_example.py
```

---

## 📝 创建测试数据集

### 方法 1: 使用代码生成

```python
from evaluation import DatasetLoader

# 创建示例数据集
DatasetLoader.create_sample_dataset(
    output_path="./my_test_dataset.json"
)
```

### 方法 2: 手动编写 JSON

创建 `my_test_dataset.json`:

```json
{
  "metadata": {
    "name": "Metasurface Paper Evaluation",
    "description": "评估 Neuro-TF 论文的 RAG 性能",
    "version": "1.0"
  },
  "queries": [
    {
      "id": "q1",
      "type": "text",
      "question": "What is the Neuro-TF approach?",
      "ground_truth": "Neuro-TF is a neural network based transfer function approach for fast metasurface design.",
      "relevant_chunks": ["chunk-abc123"],
      "difficulty": "easy"
    },
    {
      "id": "q2",
      "type": "multimodal",
      "modality_type": "image",
      "question": "Describe the absorber structure in Figure 1.",
      "ground_truth": "Figure 1 shows a three-layer metasurface absorber with metallic patterns on the top layer.",
      "relevant_chunks": ["chunk-img001"],
      "relevant_multimodal": [
        {"id": "Figure 1", "type": "image"}
      ],
      "difficulty": "medium"
    }
  ]
}
```

### 数据集字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `id` | 是 | 查询唯一标识符 |
| `type` | 是 | 查询类型 (`text` 或 `multimodal`) |
| `question` | 是 | 查询问题 |
| `ground_truth` | 推荐 | 参考答案（用于生成器评估） |
| `relevant_chunks` | 推荐 | 相关chunk列表（用于检索器评估） |
| `modality_type` | 条件 | 多模态类型（仅 `type="multimodal"` 时需要） |
| `relevant_multimodal` | 可选 | 相关多模态内容列表 |
| `difficulty` | 可选 | 难度级别（用于分析） |

---

## 🎯 运行评估

### 完整评估流程

```python
import asyncio
from evaluation import RAGEvaluator, EvaluationConfig, DatasetLoader
from raganything import RAGAnything

async def run_full_evaluation():
    # 1. 加载 RAG 实例
    rag = RAGAnything(working_dir="./rag_storage", ...)
    await rag._ensure_lightrag_initialized()
    
    # 2. 配置评估
    config = EvaluationConfig(
        working_dir="./rag_storage",
        output_dir="./evaluation_results",
        enable_retriever_eval=True,
        enable_generator_eval=True,
        enable_qa_eval=True,
        enable_multimodal_eval=True,
        use_llm_judge=True,
    )
    
    # 3. 创建评估器
    evaluator = RAGEvaluator(rag, config=config, llm_func=my_llm_func)
    
    # 4. 执行评估
    results = await evaluator.evaluate_all(
        dataset_path="./my_test_dataset.json"
    )
    
    # 5. 生成报告
    evaluator.generate_report(output_format="markdown")

asyncio.run(run_full_evaluation())
```

### 仅评估特定组件

```python
# 只评估检索器
config = EvaluationConfig(
    enable_retriever_eval=True,
    enable_generator_eval=False,
    enable_qa_eval=False,
    enable_multimodal_eval=False,
)
```

### 快速评估（无需数据集）

```python
questions = ["What is X?", "How does Y work?"]
ground_truths = ["X is...", "Y works by..."]

quick_result = await evaluator.quick_eval(
    questions=questions,
    ground_truths=ground_truths
)
```

---

## 🔧 自定义指标

### 创建新指标

```python
from evaluation.base import BaseMetric, EvaluationResult

class F1Score(BaseMetric):
    """F1 分数指标"""
    
    def __init__(self):
        super().__init__(name="F1Score")
    
    def compute(self, predictions, references, **kwargs):
        # 计算精确率
        precision = self._calculate_precision(predictions, references)
        # 计算召回率
        recall = self._calculate_recall(predictions, references)
        
        # 计算 F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return EvaluationResult(
            metric_name=self.name,
            value=f1,
            metadata={
                "precision": precision,
                "recall": recall
            }
        )
```

### 添加到评估器

```python
from evaluation.components import RetrieverEvaluator

# 方法 1: 初始化时添加
evaluator = RetrieverEvaluator(config={})
evaluator.add_metric(F1Score())

# 方法 2: 在配置中指定（需要扩展 config.py）
```

---

## 🌟 扩展评估器

### 创建新的评估器

```python
from evaluation.base import BaseEvaluator, ComponentEvalResult, EvaluationResult

class LatencyEvaluator(BaseEvaluator):
    """延迟评估器 - 评估系统响应速度"""
    
    def __init__(self, config=None):
        super().__init__(name="Latency", config=config)
    
    async def evaluate(self, rag_instance=None, test_queries=None, **kwargs):
        """测量查询延迟"""
        import time
        
        latencies = []
        
        for query in test_queries:
            start_time = time.time()
            await rag_instance.aquery(query["question"], mode="hybrid")
            latency = time.time() - start_time
            latencies.append(latency)
        
        # 计算统计
        avg_latency = sum(latencies) / len(latencies)
        
        result = EvaluationResult(
            metric_name="AverageLatency",
            value=avg_latency,
            metadata={
                "min": min(latencies),
                "max": max(latencies),
                "p50": sorted(latencies)[len(latencies)//2],
                "p95": sorted(latencies)[int(len(latencies)*0.95)]
            }
        )
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=[result],
            summary={"avg_latency_seconds": avg_latency}
        )
```

### 集成到主评估器

修改 `evaluation/evaluator.py`，在 `_init_evaluators()` 中添加:

```python
if self.config.get("enable_latency_eval", False):
    self.latency_evaluator = LatencyEvaluator(config=config_dict)
```

---

## 🎨 可视化结果

### 生成图表

```python
from evaluation.visualization import EvaluationVisualizer

# 加载结果
viz = EvaluationVisualizer("./evaluation_results/evaluation_results_20260123_123456.json")

# 绘制检索指标对比图
viz.plot_retrieval_metrics(output_path="./retrieval_chart.png")

# 绘制生成指标雷达图
viz.plot_generation_metrics_radar(output_path="./generation_radar.png")

# 生成 HTML 报告
viz.generate_html_report(output_path="./report.html")
```

---

## 📐 集成外部评估框架

### 示例: 集成 RAGAS

```python
# 安装 RAGAS
# pip install ragas

from ragas.metrics import faithfulness, answer_relevancy
from evaluation.base import BaseMetric, EvaluationResult

class RAGASFaithfulness(BaseMetric):
    """RAGAS Faithfulness 指标包装器"""
    
    def __init__(self):
        super().__init__(name="RAGAS-Faithfulness")
        self.ragas_metric = faithfulness
    
    async def compute(self, predictions, references, **kwargs):
        from ragas import evaluate
        from datasets import Dataset
        
        # 准备数据
        data = {
            "question": [references["question"]],
            "answer": [predictions],
            "contexts": [references["contexts"]],
        }
        dataset = Dataset.from_dict(data)
        
        # 评估
        result = evaluate(dataset, metrics=[self.ragas_metric])
        score = result[self.ragas_metric.name]
        
        return EvaluationResult(
            metric_name=self.name,
            value=score,
            metadata={"framework": "RAGAS"}
        )
```

---

## 🐛 常见问题

### Q1: 如何获取 chunk IDs？

查看 RAG 存储中的文件：
```bash
cat ./rag_storage/kv_store_text_chunks.json | jq 'keys'
```

### Q2: 评估速度太慢怎么办？

1. **减少测试样本**：选择代表性的查询子集
2. **禁用慢速指标**：如 BERTScore
3. **增加并发数**：调整 `max_concurrent_evals`
4. **缓存 LLM 响应**：LightRAG 自动缓存

### Q3: 如何评估没有 ground truth 的数据？

使用 **LLM-as-a-Judge**：
```python
config = EvaluationConfig(
    enable_generator_eval=True,
    use_llm_judge=True,
    llm_judge_aspects=["relevance", "coherence"],  # 不需要 ground truth
)
```

### Q4: 如何导出结果到 Excel？

```python
import json
import pandas as pd

# 加载结果
with open("evaluation_results.json") as f:
    results = json.load(f)

# 转换为 DataFrame
summary = results["component_level"]["generator"]["summary"]
df = pd.DataFrame([summary])

# 保存为 Excel
df.to_excel("evaluation_results.xlsx", index=False)
```

---

## 📊 评估指标说明

### 检索指标

| 指标 | 范围 | 说明 | 何时使用 |
|------|------|------|----------|
| **Precision@K** | 0-1 | 检索结果的精确度 | 关注准确性 |
| **Recall@K** | 0-1 | 召回了多少相关文档 | 关注完整性 |
| **MRR** | 0-1 | 第一个相关结果的位置 | 关注排序质量 |
| **NDCG@K** | 0-1 | 综合考虑排序和相关性 | 综合评估 |
| **Hit Rate@K** | 0-1 | 是否至少命中一个 | 二元判断 |

### 生成指标

| 指标 | 范围 | 说明 | 优点 | 缺点 |
|------|------|------|------|------|
| **ROUGE-L** | 0-1 | 最长公共子序列 | 快速、无需模型 | 只看表面重叠 |
| **BLEU** | 0-1 | N-gram 重叠 | 标准化 | 对同义词不敏感 |
| **BERTScore** | 0-1 | 语义相似度 | 捕捉语义 | 计算慢 |
| **LLM-Judge** | 0-1 | LLM 评判 | 全面、灵活 | 需要 LLM 调用 |

### LLM 评判维度

| 维度 | 说明 | 评估内容 |
|------|------|----------|
| **Faithfulness** | 忠实度 | 答案是否基于检索的上下文 |
| **Relevance** | 相关性 | 答案是否回答了问题 |
| **Coherence** | 连贯性 | 答案逻辑是否清晰 |
| **Completeness** | 完整性 | 答案是否全面 |

---

## 🎯 高级用法

### 按难度分析结果

```python
# 加载结果
with open("evaluation_results.json") as f:
    results = json.load(f)

# 按难度分组
details = results["component_level"]["generator"]["details"]

easy_queries = [d for d in details if d.get("difficulty") == "easy"]
hard_queries = [d for d in details if d.get("difficulty") == "hard"]

# 计算各难度的平均分
```

### 对比不同配置

```python
# 评估配置 A
config_a = EvaluationConfig(chunk_token_size=200)
results_a = await evaluator_a.evaluate_all(dataset)

# 评估配置 B
config_b = EvaluationConfig(chunk_token_size=400)
results_b = await evaluator_b.evaluate_all(dataset)

# 对比结果
```

### 增量评估

```python
# 只评估新增的查询
existing_ids = set(...)  # 已评估的查询 ID
new_queries = [q for q in dataset.queries if q["id"] not in existing_ids]

# 评估新查询
```

---

## 🔬 实验设计建议

### 1. A/B 测试不同配置

```python
# 实验组：大 chunk
config_large = {"chunk_token_size": 400}

# 对照组：小 chunk
config_small = {"chunk_token_size": 200}

# 对比评估结果
```

### 2. 消融实验

```python
# 测试多模态处理的影响
config_with_multimodal = {"enable_image_processing": True}
config_without_multimodal = {"enable_image_processing": False}

# 对比性能差异
```

### 3. 跨模型对比

```python
# 模型 A
rag_a = RAGAnything(llm_model="qwen2.5:7b", ...)
results_a = await evaluator.evaluate_all(...)

# 模型 B
rag_b = RAGAnything(llm_model="qwen2.5:14b", ...)
results_b = await evaluator.evaluate_all(...)

# 对比结果
```

---

## 📈 结果分析示例

### 使用 Pandas 分析

```python
import pandas as pd
import json

# 加载结果
with open("evaluation_results.json") as f:
    results = json.load(f)

# 提取详细结果
details = results["component_level"]["generator"]["details"]

# 转换为 DataFrame
df = pd.DataFrame(details)

# 分析
print("平均 ROUGE-L:", df["metrics"].apply(lambda x: x[0]["value"]).mean())
print("按难度统计:", df.groupby("difficulty")["metrics"].count())
```

### 趋势分析

```python
# 加载多次评估结果
results_history = []
for result_file in Path("./evaluation_results").glob("*.json"):
    with open(result_file) as f:
        results_history.append(json.load(f))

# 绘制趋势图
timestamps = [r["metadata"]["evaluation_time"] for r in results_history]
scores = [r["component_level"]["generator"]["summary"]["ROUGE-rougeL_mean"] 
          for r in results_history]

import matplotlib.pyplot as plt
plt.plot(timestamps, scores)
plt.xlabel("Time")
plt.ylabel("ROUGE-L Score")
plt.title("RAG Performance Over Time")
plt.savefig("trend.png")
```

---

## 💡 最佳实践

1. ✅ **建立基准**：首次评估后保存结果作为 baseline
2. ✅ **版本控制数据集**：将测试数据集纳入 git 管理
3. ✅ **自动化评估**：集成到 CI/CD 流程
4. ✅ **定期更新数据集**：添加新的边界案例
5. ✅ **多维度评估**：不要只依赖单一指标

## 🔗 相关资源

- [RAG-Anything 主文档](../README.md)
- [评估系统 API 文档](./README.md)
- [示例脚本](../examples/evaluation_example.py)
