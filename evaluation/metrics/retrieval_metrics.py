"""
检索指标模块
实现常用的检索评估指标：Precision@K, Recall@K, MRR, NDCG, Hit Rate 等
"""

import numpy as np
from typing import List, Set, Any
from evaluation.base import BaseMetric, EvaluationResult


class PrecisionAtK(BaseMetric):
    """Precision@K: 前 K 个结果中相关文档的比例"""
    
    def __init__(self, k: int = 10):
        super().__init__(name=f"Precision@{k}", k=k)
        self.k = k
    
    def compute(
        self, 
        predictions: List[str], 
        references: List[str], 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 Precision@K
        
        Args:
            predictions: 检索结果列表（chunk_id 或 entity_id）
            references: 相关文档列表（ground truth）
            
        Returns:
            EvaluationResult
        """
        top_k_preds = predictions[:self.k]
        relevant_set = set(references)
        
        if not top_k_preds:
            precision = 0.0
        else:
            relevant_in_top_k = len([p for p in top_k_preds if p in relevant_set])
            precision = relevant_in_top_k / len(top_k_preds)
        
        return EvaluationResult(
            metric_name=self.name,
            value=precision,
            metadata={
                "k": self.k,
                "relevant_retrieved": len([p for p in top_k_preds if p in relevant_set]),
                "total_retrieved": len(top_k_preds)
            }
        )


class RecallAtK(BaseMetric):
    """Recall@K: 前 K 个结果召回了多少相关文档"""
    
    def __init__(self, k: int = 10):
        super().__init__(name=f"Recall@{k}", k=k)
        self.k = k
    
    def compute(
        self, 
        predictions: List[str], 
        references: List[str], 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 Recall@K
        
        Args:
            predictions: 检索结果列表
            references: 相关文档列表（ground truth）
            
        Returns:
            EvaluationResult
        """
        top_k_preds = predictions[:self.k]
        relevant_set = set(references)
        
        if not relevant_set:
            recall = 0.0
        else:
            relevant_in_top_k = len([p for p in top_k_preds if p in relevant_set])
            recall = relevant_in_top_k / len(relevant_set)
        
        return EvaluationResult(
            metric_name=self.name,
            value=recall,
            metadata={
                "k": self.k,
                "relevant_retrieved": len([p for p in top_k_preds if p in relevant_set]),
                "total_relevant": len(relevant_set)
            }
        )


class MeanReciprocalRank(BaseMetric):
    """MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名"""
    
    def __init__(self):
        super().__init__(name="MRR")
    
    def compute(
        self, 
        predictions: List[str], 
        references: List[str], 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 MRR
        
        Args:
            predictions: 检索结果列表（按相关性排序）
            references: 相关文档列表
            
        Returns:
            EvaluationResult
        """
        relevant_set = set(references)
        
        for rank, pred in enumerate(predictions, 1):
            if pred in relevant_set:
                mrr = 1.0 / rank
                return EvaluationResult(
                    metric_name=self.name,
                    value=mrr,
                    metadata={"first_relevant_rank": rank}
                )
        
        # 如果没有找到相关文档
        return EvaluationResult(
            metric_name=self.name,
            value=0.0,
            metadata={"first_relevant_rank": None}
        )


class HitRate(BaseMetric):
    """Hit Rate: 前 K 个结果中是否至少包含一个相关文档"""
    
    def __init__(self, k: int = 10):
        super().__init__(name=f"HitRate@{k}", k=k)
        self.k = k
    
    def compute(
        self, 
        predictions: List[str], 
        references: List[str], 
        **kwargs
    ) -> EvaluationResult:
        """
        计算 Hit Rate@K
        
        Args:
            predictions: 检索结果列表
            references: 相关文档列表
            
        Returns:
            EvaluationResult
        """
        top_k_preds = predictions[:self.k]
        relevant_set = set(references)
        
        hit = 1.0 if any(p in relevant_set for p in top_k_preds) else 0.0
        
        return EvaluationResult(
            metric_name=self.name,
            value=hit,
            metadata={"k": self.k, "hit": hit == 1.0}
        )


class NDCG(BaseMetric):
    """NDCG (Normalized Discounted Cumulative Gain): 考虑排序的检索质量"""
    
    def __init__(self, k: int = 10):
        super().__init__(name=f"NDCG@{k}", k=k)
        self.k = k
    
    def _dcg(self, relevances: List[float]) -> float:
        """计算 DCG"""
        return sum(
            (2**rel - 1) / np.log2(rank + 2)  # rank+2 因为从 0 开始，log2(1) 无意义
            for rank, rel in enumerate(relevances)
        )
    
    def compute(
        self, 
        predictions: List[str], 
        references: List[str], 
        relevance_scores: List[float] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        计算 NDCG@K
        
        Args:
            predictions: 检索结果列表
            references: 相关文档列表
            relevance_scores: 可选的相关性分数（如果没有则用二元相关性）
            
        Returns:
            EvaluationResult
        """
        top_k_preds = predictions[:self.k]
        relevant_set = set(references)
        
        # 如果没有提供相关性分数，使用二元相关性（相关=1，不相关=0）
        if relevance_scores is None:
            relevances = [1.0 if p in relevant_set else 0.0 for p in top_k_preds]
        else:
            relevances = relevance_scores[:self.k]
        
        # 计算 DCG
        dcg = self._dcg(relevances)
        
        # 计算 IDCG (理想情况下的 DCG)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._dcg(ideal_relevances)
        
        # 计算 NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return EvaluationResult(
            metric_name=self.name,
            value=ndcg,
            metadata={
                "k": self.k,
                "dcg": dcg,
                "idcg": idcg
            }
        )


# 便捷函数：创建常用的检索指标集合
def create_retrieval_metrics(k_values: List[int] = [1, 3, 5, 10, 20]) -> List[BaseMetric]:
    """
    创建一套完整的检索指标
    
    Args:
        k_values: K 值列表
        
    Returns:
        List[BaseMetric]: 指标列表
    """
    metrics = []
    
    for k in k_values:
        metrics.extend([
            PrecisionAtK(k=k),
            RecallAtK(k=k),
            HitRate(k=k),
            NDCG(k=k),
        ])
    
    # MRR 不需要 K 值
    metrics.append(MeanReciprocalRank())
    
    return metrics
