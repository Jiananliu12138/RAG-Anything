"""
端到端评估模块
"""

from evaluation.end_to_end.qa_eval import QAEvaluator
from evaluation.end_to_end.multimodal_eval import MultimodalEvaluator

# 条件导入RAGAS评估器（需要安装ragas）
try:
    from evaluation.end_to_end.ragas_eval import RAGASEvaluator
    __all__ = [
        "QAEvaluator",
        "MultimodalEvaluator",
        "RAGASEvaluator",
    ]
except ImportError:
    __all__ = [
        "QAEvaluator",
        "MultimodalEvaluator",
    ]
