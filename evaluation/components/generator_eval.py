"""
生成器组件评估
评估 RAG 系统中答案生成组件的性能
"""

from typing import List, Dict, Any
from evaluation.base import BaseEvaluator, ComponentEvalResult
from evaluation.metrics import create_generation_metrics, LLMJudgeMetric


class GeneratorEvaluator(BaseEvaluator):
    """生成器评估器"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_func=None):
        super().__init__(name="Generator", config=config)
        
        # 添加生成指标
        if config:
            self.metrics = create_generation_metrics(
                use_rouge=config.get("use_rouge", True),
                use_bleu=config.get("use_bleu", False),
                use_bertscore=config.get("use_bertscore", False),
                use_semantic_sim=config.get("use_semantic_sim", True)
            )
            
            # 如果启用 LLM 评判且提供了 LLM 函数
            if config.get("use_llm_judge", True) and llm_func:
                aspects = config.get("llm_judge_aspects", ["faithfulness", "relevance", "coherence"])
                for aspect in aspects:
                    self.metrics.append(LLMJudgeMetric(llm_func=llm_func, aspect=aspect))
        else:
            self.metrics = create_generation_metrics()
    
    async def evaluate(
        self,
        rag_instance=None,
        test_queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> ComponentEvalResult:
        """
        评估生成器性能
        
        Args:
            rag_instance: RAGAnything 实例
            test_queries: 测试查询列表，每个包含:
                - question: 查询问题
                - ground_truth: 参考答案
                - context: 上下文（可选，用于 faithfulness 评估）
                
        Returns:
            ComponentEvalResult: 评估结果
        """
        if not test_queries:
            raise ValueError("需要提供测试查询数据")
        
        all_results = []
        detailed_results = []
        
        for query_item in test_queries:
            question = query_item["question"]
            ground_truth = query_item.get("ground_truth", "")
            context = query_item.get("context", "")
            
            # 生成答案
            try:
                answer = await rag_instance.aquery(question, mode="hybrid")
                
                # 对每个指标进行评估
                query_results = []
                for metric in self.metrics:
                    # 构造 references 字典（用于 LLM Judge）
                    references = {
                        "question": question,
                        "context": context,
                        "ground_truth": ground_truth
                    }
                    
                    # 异步指标（如 LLM Judge）
                    if hasattr(metric.compute, '__self__') and hasattr(metric, 'llm_func'):
                        result = await metric.compute(
                            predictions=answer,
                            references=references
                        )
                    else:
                        # 同步指标（如 ROUGE, BLEU）
                        result = metric.compute(
                            predictions=answer,
                            references=ground_truth
                        )
                    
                    query_results.append(result)
                    all_results.append(result)
                
                # 保存详细结果
                detailed_results.append({
                    "question": question,
                    "generated_answer": answer,
                    "ground_truth": ground_truth,
                    "metrics": [r.to_dict() for r in query_results]
                })
                
            except Exception as e:
                print(f"评估查询失败: {question} - {e}")
                continue
        
        # 聚合结果
        summary = self._aggregate_results(all_results)
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=all_results,
            summary=summary,
            details=detailed_results if self.config.get("save_detailed_results", True) else None
        )
