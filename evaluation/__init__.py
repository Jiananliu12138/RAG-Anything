"""
RAG-Anything 评估系统
提供完整的、可扩展的 RAG 评估框架

主要组件:
- 组件级评估: 检索器、生成器评估
- 端到端评估: 问答、多模态评估
- 指标系统: 检索指标、生成指标、多模态指标
- 数据集管理: 加载和验证测试数据集

使用示例:
    from evaluation import RAGEvaluator, EvaluationConfig
    from raganything import RAGAnything
    
    # 初始化 RAG 实例
    rag = RAGAnything(...)
    
    # 配置评估
    eval_config = EvaluationConfig(
        working_dir="./rag_storage",
        output_dir="./evaluation_results"
    )
    
    # 执行评估
    evaluator = RAGEvaluator(rag, config=eval_config)
    results = await evaluator.evaluate_all(dataset_path="test_dataset.json")
"""

from evaluation.config import EvaluationConfig
from evaluation.evaluator import RAGEvaluator
from evaluation.dataset import DatasetLoader, EvaluationDataset
from evaluation.base import EvaluationResult, ComponentEvalResult

__version__ = "1.0.0"

__all__ = [
    "EvaluationConfig",
    "RAGEvaluator",
    "DatasetLoader",
    "EvaluationDataset",
    "EvaluationResult",
    "ComponentEvalResult",
]
