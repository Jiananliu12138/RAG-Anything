"""
检索器组件评估
评估 RAG 系统中检索组件的性能
"""

from typing import List, Dict, Any
from evaluation.base import BaseEvaluator, ComponentEvalResult, EvaluationResult
from evaluation.metrics import create_retrieval_metrics


class RetrieverEvaluator(BaseEvaluator):
    """检索器评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="Retriever", config=config)
        
        # 添加检索指标
        k_values = config.get("retrieval_top_k", [1, 3, 5, 10, 20]) if config else [1, 3, 5, 10, 20]
        self.metrics = create_retrieval_metrics(k_values=k_values)
    
    async def evaluate(
        self,
        rag_instance=None,
        test_queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> ComponentEvalResult:
        """
        评估检索器性能
        
        Args:
            rag_instance: RAGAnything 实例
            test_queries: 测试查询列表，每个包含:
                - question: 查询问题
                - relevant_chunks: 相关chunk ID列表 (ground truth)
                - relevant_entities: 相关实体列表 (可选)
                
        Returns:
            ComponentEvalResult: 评估结果
        """
        if not test_queries:
            raise ValueError("需要提供测试查询数据")
        
        all_results = []
        detailed_results = []
        
        for query_item in test_queries:
            question = query_item["question"]
            ground_truth_chunks = query_item.get("relevant_chunks", [])
            
            # 执行检索（只获取上下文，不生成答案）
            try:
                # 使用 LightRAG 的底层检索功能
                from lightrag import QueryParam
                
                # 混合检索
                retrieved_context = await rag_instance.lightrag.aquery(
                    question,
                    param=QueryParam(
                        mode="hybrid",
                        only_need_context=True,  # 只返回检索结果
                    )
                )
                
                # 提取检索到的 chunk IDs
                retrieved_chunks = self._extract_chunk_ids(retrieved_context)
                
                # 对每个指标进行评估
                query_results = []
                for metric in self.metrics:
                    result = metric.compute(
                        predictions=retrieved_chunks,
                        references=ground_truth_chunks
                    )
                    query_results.append(result)
                    all_results.append(result)
                
                # 保存详细结果
                detailed_results.append({
                    "question": question,
                    "retrieved_count": len(retrieved_chunks),
                    "relevant_count": len(ground_truth_chunks),
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
    
    def _extract_chunk_ids(self, context: Any) -> List[str]:
        """
        从检索上下文中提取 chunk IDs
        
        Args:
            context: 检索返回的上下文
            
        Returns:
            List[str]: chunk ID 列表
        """
        # 根据实际返回格式提取
        if isinstance(context, dict):
            # 如果返回的是字典，尝试提取 chunks
            return context.get("chunks", [])
        elif isinstance(context, str):
            # 如果返回的是字符串，尝试解析
            import re
            chunk_pattern = r'chunk-[a-f0-9]+'
            return re.findall(chunk_pattern, context)
        else:
            return []
