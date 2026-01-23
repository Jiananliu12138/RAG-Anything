"""
端到端问答评估
综合评估整个 RAG 流程（检索 + 生成）的性能
"""

from typing import List, Dict, Any
from evaluation.base import BaseEvaluator, ComponentEvalResult
from evaluation.components import RetrieverEvaluator, GeneratorEvaluator


class QAEvaluator(BaseEvaluator):
    """端到端问答评估器"""
    
    def __init__(self, config: Dict[str, Any] = None, llm_func=None):
        super().__init__(name="QA-EndToEnd", config=config)
        
        # 创建子评估器
        self.retriever_eval = RetrieverEvaluator(config=config)
        self.generator_eval = GeneratorEvaluator(config=config, llm_func=llm_func)
    
    async def evaluate(
        self,
        rag_instance=None,
        test_queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, ComponentEvalResult]:
        """
        执行端到端评估
        
        Args:
            rag_instance: RAGAnything 实例
            test_queries: 测试查询列表，每个包含:
                - question: 查询问题
                - ground_truth: 参考答案
                - relevant_chunks: 相关chunk列表 (用于检索评估)
                - context: 上下文 (可选)
                
        Returns:
            Dict[str, ComponentEvalResult]: 包含检索和生成两部分的评估结果
        """
        if not test_queries:
            raise ValueError("需要提供测试查询数据")
        
        results = {}
        
        # 1. 评估检索组件
        if self.config.get("enable_retriever_eval", True):
            print("\n" + "="*60)
            print("评估检索组件...")
            print("="*60)
            retriever_result = await self.retriever_eval.evaluate(
                rag_instance=rag_instance,
                test_queries=test_queries
            )
            results["retriever"] = retriever_result
            print(f"✅ 检索评估完成: {len(retriever_result.metrics)} 个指标")
        
        # 2. 评估生成组件
        if self.config.get("enable_generator_eval", True):
            print("\n" + "="*60)
            print("评估生成组件...")
            print("="*60)
            generator_result = await self.generator_eval.evaluate(
                rag_instance=rag_instance,
                test_queries=test_queries
            )
            results["generator"] = generator_result
            print(f"✅ 生成评估完成: {len(generator_result.metrics)} 个指标")
        
        return results
    
    def print_summary(self, results: Dict[str, ComponentEvalResult]):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("端到端评估摘要")
        print("="*60)
        
        for component_name, result in results.items():
            print(f"\n📊 {component_name.upper()} 组件:")
            print("-" * 60)
            for metric_name, value in result.summary.items():
                print(f"  {metric_name}: {value:.4f}")
