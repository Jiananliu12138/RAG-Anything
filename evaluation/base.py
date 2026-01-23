"""
评估系统基础抽象类
定义了评估器和指标的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    
    metric_name: str
    """指标名称"""
    
    value: float
    """指标值"""
    
    metadata: Dict[str, Any] = None
    """额外的元数据"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "metric": self.metric_name,
            "value": self.value,
            "metadata": self.metadata or {}
        }


@dataclass
class ComponentEvalResult:
    """组件评估结果"""
    
    component_name: str
    """组件名称（如 retriever, generator）"""
    
    metrics: List[EvaluationResult]
    """评估指标列表"""
    
    summary: Dict[str, float]
    """汇总统计"""
    
    details: Optional[List[Dict[str, Any]]] = None
    """详细的逐条评估结果"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "component": self.component_name,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
            "details": self.details or []
        }
    
    def save(self, output_path: str):
        """保存到文件"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class BaseMetric(ABC):
    """基础指标抽象类"""
    
    def __init__(self, name: str, **kwargs):
        """
        初始化指标
        
        Args:
            name: 指标名称
            **kwargs: 额外配置参数
        """
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def compute(self, predictions: Any, references: Any, **kwargs) -> EvaluationResult:
        """
        计算指标
        
        Args:
            predictions: 预测结果
            references: 参考答案/真值
            **kwargs: 额外参数
            
        Returns:
            EvaluationResult: 评估结果
        """
        pass
    
    def batch_compute(
        self, 
        predictions_list: List[Any], 
        references_list: List[Any],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        批量计算指标
        
        Args:
            predictions_list: 预测结果列表
            references_list: 参考答案列表
            **kwargs: 额外参数
            
        Returns:
            List[EvaluationResult]: 评估结果列表
        """
        results = []
        for pred, ref in zip(predictions_list, references_list):
            results.append(self.compute(pred, ref, **kwargs))
        return results


class BaseEvaluator(ABC):
    """基础评估器抽象类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化评估器
        
        Args:
            name: 评估器名称
            config: 配置字典
        """
        self.name = name
        self.config = config or {}
        self.metrics = []
    
    def add_metric(self, metric: BaseMetric):
        """添加评估指标"""
        self.metrics.append(metric)
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> ComponentEvalResult:
        """
        执行评估
        
        Args:
            **kwargs: 评估所需的数据和参数
            
        Returns:
            ComponentEvalResult: 组件评估结果
        """
        pass
    
    def _aggregate_results(
        self, 
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """
        聚合评估结果
        
        Args:
            results: 评估结果列表
            
        Returns:
            Dict[str, float]: 聚合后的统计数据
        """
        if not results:
            return {}
        
        # 按指标名称分组
        metric_groups = {}
        for result in results:
            if result.metric_name not in metric_groups:
                metric_groups[result.metric_name] = []
            metric_groups[result.metric_name].append(result.value)
        
        # 计算每个指标的平均值、最小值、最大值
        summary = {}
        for metric_name, values in metric_groups.items():
            summary[f"{metric_name}_mean"] = sum(values) / len(values)
            summary[f"{metric_name}_min"] = min(values)
            summary[f"{metric_name}_max"] = max(values)
            summary[f"{metric_name}_count"] = len(values)
        
        return summary
