"""
端到端多模态评估
评估 RAG 系统处理多模态内容（图像、表格、公式）的能力
"""

from typing import List, Dict, Any
from evaluation.base import BaseEvaluator, ComponentEvalResult, EvaluationResult
from evaluation.metrics import create_multimodal_metrics


class MultimodalEvaluator(BaseEvaluator):
    """多模态评估器"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_func=None):
        super().__init__(name="Multimodal-EndToEnd", config=config)
        
        # 添加多模态指标
        modality_types = config.get("multimodal_types", ["image", "table", "equation"]) if config else ["image", "table", "equation"]
        self.metrics = create_multimodal_metrics(modality_types=modality_types)
        
        # 添加 LLM 评判（针对多模态理解质量）
        if llm_func:
            from evaluation.metrics import LLMJudgeMetric
            self.metrics.append(
                LLMJudgeMetric(llm_func=llm_func, aspect="multimodal_understanding")
            )
    
    async def evaluate(
        self,
        rag_instance=None,
        test_queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> ComponentEvalResult:
        """
        评估多模态处理能力
        
        Args:
            rag_instance: RAGAnything 实例
            test_queries: 测试查询列表，每个包含:
                - question: 涉及多模态内容的查询
                - modality_type: 涉及的模态类型 (image/table/equation)
                - ground_truth: 参考答案
                - relevant_multimodal: 相关多模态内容列表 (ground truth)
                
        Returns:
            ComponentEvalResult: 评估结果
        """
        if not test_queries:
            raise ValueError("需要提供多模态测试查询数据")
        
        all_results = []
        detailed_results = []
        
        # 按模态类型分组
        queries_by_modality = {}
        for query in test_queries:
            mod_type = query.get("modality_type", "unknown")
            if mod_type not in queries_by_modality:
                queries_by_modality[mod_type] = []
            queries_by_modality[mod_type].append(query)
        
        # 对每种模态类型进行评估
        for mod_type, queries in queries_by_modality.items():
            print(f"\n评估 {mod_type} 类型的多模态查询...")
            
            for query_item in queries:
                question = query_item["question"]
                ground_truth_items = query_item.get("relevant_multimodal", [])
                
                try:
                    # 执行查询
                    answer = await rag_instance.aquery(question, mode="hybrid")
                    
                    # 检查答案中是否包含多模态内容的引用
                    retrieved_multimodal = self._extract_multimodal_refs(answer, mod_type)
                    
                    # 计算多模态检索准确率
                    for metric in self.metrics:
                        if "RetrievalAccuracy" in metric.name and mod_type in metric.name.lower():
                            result = metric.compute(
                                predictions=retrieved_multimodal,
                                references=ground_truth_items
                            )
                            all_results.append(result)
                    
                    # 保存详细结果
                    detailed_results.append({
                        "question": question,
                        "modality_type": mod_type,
                        "answer": answer,
                        "retrieved_multimodal_count": len(retrieved_multimodal),
                        "expected_multimodal_count": len(ground_truth_items)
                    })
                    
                except Exception as e:
                    print(f"评估失败: {question} - {e}")
                    continue
        
        # 聚合结果
        summary = self._aggregate_results(all_results)
        
        # 添加整体覆盖率统计
        coverage_result = self._compute_overall_coverage(queries_by_modality, detailed_results)
        all_results.append(coverage_result)
        summary["overall_coverage"] = coverage_result.value
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=all_results,
            summary=summary,
            details=detailed_results if self.config.get("save_detailed_results", True) else None
        )
    
    def _extract_multimodal_refs(self, answer: str, modality_type: str) -> List[Dict[str, Any]]:
        """
        从答案中提取多模态内容引用
        
        Args:
            answer: 生成的答案
            modality_type: 模态类型
            
        Returns:
            List[Dict[str, Any]]: 提取到的多模态引用列表
        """
        import re
        
        # 根据模态类型提取不同的模式
        patterns = {
            "image": r'(Figure|Fig\.|Image|图)\s*\d+',
            "table": r'(Table|表)\s*\d+',
            "equation": r'(Equation|Eq\.|公式)\s*\d+'
        }
        
        pattern = patterns.get(modality_type, r'\w+')
        matches = re.findall(pattern, answer, re.IGNORECASE)
        
        # 转换为字典格式
        return [{"id": match, "type": modality_type} for match in set(matches)]
    
    def _compute_overall_coverage(
        self, 
        queries_by_modality: Dict[str, List[Any]],
        detailed_results: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        计算整体的多模态覆盖率
        
        Args:
            queries_by_modality: 按模态类型分组的查询
            detailed_results: 详细结果列表
            
        Returns:
            EvaluationResult: 覆盖率评估结果
        """
        total_modalities = len(queries_by_modality)
        covered_modalities = 0
        
        for mod_type in queries_by_modality.keys():
            # 检查是否至少有一个该类型的查询成功检索到内容
            type_results = [r for r in detailed_results if r.get("modality_type") == mod_type]
            if any(r.get("retrieved_multimodal_count", 0) > 0 for r in type_results):
                covered_modalities += 1
        
        coverage = covered_modalities / total_modalities if total_modalities > 0 else 0.0
        
        return EvaluationResult(
            metric_name="OverallMultimodalCoverage",
            value=coverage,
            metadata={
                "covered_modalities": covered_modalities,
                "total_modalities": total_modalities
            }
        )
