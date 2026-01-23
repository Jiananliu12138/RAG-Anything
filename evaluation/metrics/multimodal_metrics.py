"""
多模态指标模块
评估多模态内容（图像、表格、公式）的检索和理解质量
"""

from typing import List, Dict, Any
from evaluation.base import BaseMetric, EvaluationResult


class MultimodalRetrievalAccuracy(BaseMetric):
    """多模态检索准确率：检索到的多模态内容是否正确"""
    
    def __init__(self, modality_type: str = "image"):
        super().__init__(name=f"{modality_type.title()}RetrievalAccuracy", modality_type=modality_type)
        self.modality_type = modality_type
    
    def compute(
        self, 
        predictions: List[Dict[str, Any]], 
        references: List[Dict[str, Any]], 
        **kwargs
    ) -> EvaluationResult:
        """
        计算多模态检索准确率
        
        Args:
            predictions: 检索到的多模态内容列表
            references: 应该检索到的多模态内容列表（ground truth）
            
        Returns:
            EvaluationResult
        """
        if not references:
            return EvaluationResult(self.name, 1.0, {"note": "No ground truth"})
        
        # 提取 ID 或标识符进行匹配
        pred_ids = set(p.get("id") or p.get("entity_name") for p in predictions)
        ref_ids = set(r.get("id") or r.get("entity_name") for r in references)
        
        # 计算准确率
        correct = len(pred_ids & ref_ids)
        total = len(ref_ids)
        accuracy = correct / total if total > 0 else 0.0
        
        return EvaluationResult(
            metric_name=self.name,
            value=accuracy,
            metadata={
                "modality": self.modality_type,
                "correct": correct,
                "total": total,
                "retrieved": len(pred_ids)
            }
        )


class MultimodalCoverageRate(BaseMetric):
    """多模态覆盖率：所有类型的多模态内容是否都被检索到"""
    
    def __init__(self):
        super().__init__(name="MultimodalCoverage")
    
    def compute(
        self, 
        predictions: Dict[str, List[Any]], 
        references: Dict[str, List[Any]], 
        **kwargs
    ) -> EvaluationResult:
        """
        计算多模态覆盖率
        
        Args:
            predictions: 各类型检索到的内容 {"image": [...], "table": [...], ...}
            references: 各类型应该检索到的内容
            
        Returns:
            EvaluationResult
        """
        total_types = len(references)
        covered_types = 0
        
        type_details = {}
        for mod_type, ref_items in references.items():
            if ref_items:  # 如果有 ground truth
                pred_items = predictions.get(mod_type, [])
                if pred_items:  # 如果检索到了
                    covered_types += 1
                    type_details[mod_type] = True
                else:
                    type_details[mod_type] = False
        
        coverage = covered_types / total_types if total_types > 0 else 0.0
        
        return EvaluationResult(
            metric_name=self.name,
            value=coverage,
            metadata={
                "covered_types": covered_types,
                "total_types": total_types,
                "details": type_details
            }
        )


class ImageDescriptionQuality(BaseMetric):
    """图像描述质量（使用 LLM 评判生成的图像描述是否准确）"""
    
    def __init__(self, llm_func):
        super().__init__(name="ImageDescriptionQuality")
        self.llm_func = llm_func
    
    async def compute(
        self, 
        predictions: str, 
        references: Dict[str, Any], 
        **kwargs
    ) -> EvaluationResult:
        """
        评估图像描述质量
        
        Args:
            predictions: 生成的图像描述
            references: 包含 image_path, ground_truth_caption 等的字典
            
        Returns:
            EvaluationResult
        """
        prompt = f"""
You are an expert evaluator. Given a generated image description and a reference description, 
rate how well the generated description captures the key visual elements.

Reference Description: {references.get('ground_truth', 'N/A')}

Generated Description: {predictions}

Rate the quality on a scale of 1-5:
1 = Poor (misses key elements)
2 = Fair (captures some elements)
3 = Good (captures main elements)
4 = Very Good (captures most elements accurately)
5 = Excellent (comprehensive and accurate)

Output ONLY a single number (1-5) without explanation.
"""
        
        try:
            response = await self.llm_func(prompt)
            import re
            score_match = re.search(r'[1-5]', response)
            score = int(score_match.group()) / 5.0 if score_match else 0.0
            
            return EvaluationResult(
                metric_name=self.name,
                value=score,
                metadata={"raw_score": score * 5 if score_match else None}
            )
        except Exception as e:
            return EvaluationResult(
                metric_name=self.name,
                value=0.0,
                metadata={"error": str(e)}
            )


def create_multimodal_metrics(
    modality_types: List[str] = ["image", "table", "equation"]
) -> List[BaseMetric]:
    """
    创建多模态指标集合
    
    Args:
        modality_types: 需要评估的模态类型列表
        
    Returns:
        List[BaseMetric]: 指标列表
    """
    metrics = []
    
    for mod_type in modality_types:
        metrics.append(MultimodalRetrievalAccuracy(modality_type=mod_type))
    
    metrics.append(MultimodalCoverageRate())
    
    return metrics
