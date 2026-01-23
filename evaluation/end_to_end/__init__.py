"""
端到端评估模块
"""

from evaluation.end_to_end.qa_eval import QAEvaluator
from evaluation.end_to_end.multimodal_eval import MultimodalEvaluator

__all__ = [
    "QAEvaluator",
    "MultimodalEvaluator",
]
