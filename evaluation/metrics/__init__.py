"""
评估指标模块
"""

from evaluation.metrics.retrieval_metrics import (
    PrecisionAtK,
    RecallAtK,
    MeanReciprocalRank,
    HitRate,
    NDCG,
    create_retrieval_metrics,
)

from evaluation.metrics.generation_metrics import (
    ROUGEMetric,
    BLEUMetric,
    BERTScoreMetric,
    SemanticSimilarity,
    LLMJudgeMetric,
    create_generation_metrics,
)

from evaluation.metrics.multimodal_metrics import (
    MultimodalRetrievalAccuracy,
    MultimodalCoverageRate,
    ImageDescriptionQuality,
    create_multimodal_metrics,
)

__all__ = [
    # 检索指标
    "PrecisionAtK",
    "RecallAtK",
    "MeanReciprocalRank",
    "HitRate",
    "NDCG",
    "create_retrieval_metrics",
    # 生成指标
    "ROUGEMetric",
    "BLEUMetric",
    "BERTScoreMetric",
    "SemanticSimilarity",
    "LLMJudgeMetric",
    "create_generation_metrics",
    # 多模态指标
    "MultimodalRetrievalAccuracy",
    "MultimodalCoverageRate",
    "ImageDescriptionQuality",
    "create_multimodal_metrics",
]
