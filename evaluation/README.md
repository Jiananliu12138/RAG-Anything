# RAG-Anything 评估系统

完整的、可扩展的 RAG 评估框架，支持组件级和端到端评估。

## 🏗️ 架构设计

```
evaluation/
├── config.py                   # 评估配置
├── base.py                     # 基础抽象类
├── evaluator.py                # 主评估器
├── metrics/                    # 评价指标
│   ├── retrieval_metrics.py   # 检索指标 (Precision@K, Recall@K, MRR, NDCG)
│   ├── generation_metrics.py  # 生成指标 (ROUGE, BLEU, BERTScore)
│   └── multimodal_metrics.py  # 多模态指标
├── components/                 # 组件级评估
│   ├── retriever_eval.py      # 检索器评估
│   └── generator_eval.py      # 生成器评估
├── end_to_end/                # 端到端评估
│   ├── qa_eval.py             # 问答评估
│   └── multimodal_eval.py     # 多模态评估
└── dataset/                   # 数据集管理
    └── dataset_loader.py      # 数据集加载器
```

## ✨ 特性

### 组件级评估

#### 1️⃣ 检索器评估
- **Precision@K**: 前 K 个结果的精确率
- **Recall@K**: 前 K 个结果的召回率
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
- **NDCG@K**: 归一化折损累积增益
- **Hit Rate@K**: 命中率

#### 2️⃣ 生成器评估
- **ROUGE**: 文本重叠度（ROUGE-1, ROUGE-2, ROUGE-L）
- **BLEU**: 机器翻译质量评估
- **BERTScore**: 基于 BERT 的语义相似度
- **Semantic Similarity**: 基于 Sentence-Transformers 的语义相似度
- **LLM-as-a-Judge**: 使用 LLM 评判答案质量
  - Faithfulness (忠实度)
  - Relevance (相关性)
  - Coherence (连贯性)
  - Completeness (完整性)

### 端到端评估

#### 3️⃣ 问答评估
综合评估检索和生成的整体表现

#### 4️⃣ 多模态评估
- **Multimodal Retrieval Accuracy**: 多模态检索准确率
- **Multimodal Coverage**: 多模态类型覆盖率
- **Image Description Quality**: 图像描述质量

## 🚀 快速开始

### 1. 基本使用

```python
import asyncio
from evaluation import RAGEvaluator, EvaluationConfig, DatasetLoader
from raganything import RAGAnything

async def run_evaluation():
    # 初始化 RAG 实例
    rag = RAGAnything(working_dir="./rag_storage", ...)
    await rag._ensure_lightrag_initialized()
    
    # 配置评估
    eval_config = EvaluationConfig(
        working_dir="./rag_storage",
        output_dir="./evaluation_results",
        enable_retriever_eval=True,
        enable_generator_eval=True,
        use_llm_judge=True,
    )
    
    # 创建评估器
    evaluator = RAGEvaluator(rag, config=eval_config)
    
    # 执行评估
    results = await evaluator.evaluate_all(
        dataset_path="./test_dataset.json"
    )
    
    # 生成报告
    evaluator.generate_report(output_format="markdown")

asyncio.run(run_evaluation())
```

### 2. 创建测试数据集

```python
from evaluation import DatasetLoader

# 创建示例数据集
DatasetLoader.create_sample_dataset(
    output_path="./my_test_dataset.json"
)
```

### 3. 快速评估（无需完整数据集）

```python
# 快速评估几个问题
questions = [
    "What is the main topic?",
    "What are the key findings?"
]

ground_truths = [
    "The paper discusses metasurface design.",
    "The key finding is improved absorption rates."
]

quick_result = await evaluator.quick_eval(
    questions=questions,
    ground_truths=ground_truths
)
```

## 📊 数据集格式

测试数据集应为 JSON 格式：

```json
{
  "metadata": {
    "name": "My Test Dataset",
    "version": "1.0"
  },
  "queries": [
    {
      "id": "q1",
      "type": "text",
      "question": "What is...?",
      "ground_truth": "The answer is...",
      "relevant_chunks": ["chunk-id1", "chunk-id2"],
      "difficulty": "easy"
    },
    {
      "id": "q2",
      "type": "multimodal",
      "modality_type": "image",
      "question": "Describe the figure...",
      "ground_truth": "The figure shows...",
      "relevant_multimodal": [
        {"id": "Figure 1", "type": "image"}
      ],
      "difficulty": "medium"
    }
  ]
}
```

## 🔧 扩展性设计

### 添加新的评估指标

#### 1. 创建自定义指标类

```python
from evaluation.base import BaseMetric, EvaluationResult

class MyCustomMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(name="MyMetric", **kwargs)
    
    def compute(self, predictions, references, **kwargs):
        # 你的评估逻辑
        score = self._calculate_score(predictions, references)
        
        return EvaluationResult(
            metric_name=self.name,
            value=score,
            metadata={"custom_info": "..."}
        )
```

#### 2. 将指标添加到评估器

```python
from evaluation.components import GeneratorEvaluator

evaluator = GeneratorEvaluator(config={})
evaluator.add_metric(MyCustomMetric())
```

### 添加新的评估器

```python
from evaluation.base import BaseEvaluator, ComponentEvalResult

class MyCustomEvaluator(BaseEvaluator):
    def __init__(self, config=None):
        super().__init__(name="MyEvaluator", config=config)
        # 添加指标
        self.add_metric(MyCustomMetric())
    
    async def evaluate(self, **kwargs):
        # 你的评估逻辑
        results = []
        for metric in self.metrics:
            result = metric.compute(...)
            results.append(result)
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=results,
            summary=self._aggregate_results(results)
        )
```

### 集成外部评估框架

```python
# 示例：集成 RAGAS 框架
from ragas.metrics import faithfulness, answer_relevancy

class RAGASMetric(BaseMetric):
    def __init__(self, ragas_metric):
        super().__init__(name=f"RAGAS-{ragas_metric.name}")
        self.ragas_metric = ragas_metric
    
    def compute(self, predictions, references, **kwargs):
        # 调用 RAGAS 指标
        score = self.ragas_metric.score(...)
        return EvaluationResult(self.name, score)
```

## 📦 依赖项

### 核心依赖（已包含）
- `numpy`
- `raganything`
- `lightrag-hku`

### 可选依赖

```bash
# 生成指标
pip install rouge-score nltk

# BERTScore
pip install bert-score

# 语义相似度
pip install sentence-transformers scikit-learn

# RAGAS 集成（可选）
pip install ragas

# 数据分析和可视化
pip install pandas matplotlib seaborn
```

## 📈 评估结果格式

### JSON 结果文件

```json
{
  "metadata": {
    "evaluation_time": "2026-01-23T...",
    "dataset": {...},
    "config": {...}
  },
  "component_level": {
    "retriever": {
      "metrics": [...],
      "summary": {
        "Precision@5_mean": 0.85,
        "Recall@5_mean": 0.72,
        ...
      }
    },
    "generator": {
      "metrics": [...],
      "summary": {
        "ROUGE-rougeL_mean": 0.68,
        "LLM-Judge-faithfulness_mean": 0.82,
        ...
      }
    }
  },
  "end_to_end": {
    "qa": {...},
    "multimodal": {...}
  }
}
```

## 🎯 最佳实践

1. **准备高质量的测试数据集**
   - 覆盖不同难度级别的问题
   - 包含多种模态类型的查询
   - 提供准确的 ground truth

2. **选择合适的评估指标**
   - 检索任务：Precision, Recall, NDCG
   - 生成任务：ROUGE, LLM-Judge
   - 多模态任务：Coverage, Retrieval Accuracy

3. **使用 LLM-as-a-Judge 补充自动指标**
   - 自动指标（ROUGE等）可能无法捕捉语义质量
   - LLM 评判可以评估 faithfulness, coherence 等方面

4. **定期评估和监控**
   - 在模型更新后重新评估
   - 追踪评估指标的变化趋势

## 🔍 故障排除

### 问题 1: 缺少依赖
```bash
# 安装所有可选依赖
pip install rouge-score nltk sentence-transformers scikit-learn
```

### 问题 2: 数据集格式错误
使用 `DatasetLoader.validate_dataset()` 验证数据集格式

### 问题 3: 评估速度慢
- 减少测试查询数量
- 禁用 BERTScore（计算密集型）
- 调整 `max_concurrent_evals` 参数

## 📚 参考资料

- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge)
- [BERTScore](https://github.com/Tiiiger/bert_score)
