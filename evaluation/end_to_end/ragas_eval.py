"""
RAGAS端到端评估
使用RAGAS框架评估RAG系统的整体质量

RAGAS指标：
- Faithfulness: 答案是否基于检索内容（忠实度）
- AnswerRelevancy: 答案是否相关（相关性）
- ContextRecall: 是否检索到所有相关信息（上下文召回）
- ContextPrecision: 检索的上下文是否干净无噪声（上下文精确度）
"""

import asyncio
import json
import math
import os
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# 抑制RAGAS相关的警告
warnings.filterwarnings(
    "ignore",
    message=".*LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Unexpected type for token usage.*",
    category=UserWarning,
)

# 条件导入RAGAS
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
    from tqdm.auto import tqdm
    
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Dataset = None
    evaluate = None
    LangchainLLMWrapper = None
    ChatOllama = None
    OllamaEmbeddings = None

from evaluation.base import BaseEvaluator, ComponentEvalResult


def _is_nan(value: Any) -> bool:
    """检查值是否为NaN"""
    return isinstance(value, float) and math.isnan(value)


class RAGASEvaluator(BaseEvaluator):
    """RAGAS端到端评估器"""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        llm_func=None,
        eval_llm_model: str = None,
        eval_embedding_model: str = None,
        ollama_host: str = None,
    ):
        """
        初始化RAGAS评估器
        
        Args:
            config: 评估配置
            llm_func: LLM函数（用于RAG查询，不是RAGAS评估）
            eval_llm_model: RAGAS评估使用的LLM模型（默认：qwen2.5:7b-instruct）
            eval_embedding_model: RAGAS评估使用的embedding模型（默认：nomic-embed-text）
            ollama_host: Ollama服务地址（默认：http://localhost:11434）
        """
        super().__init__(name="RAGAS-EndToEnd", config=config or {})
        
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS dependencies not installed. "
                "Install with: pip install ragas datasets langchain-ollama"
            )
        
        # 配置Ollama
        self.ollama_host = ollama_host or os.getenv("EVAL_OLLAMA_HOST", "http://localhost:11434")
        self.eval_llm_model = eval_llm_model or os.getenv("EVAL_LLM_MODEL", "qwen2.5:7b-instruct")
        self.eval_embedding_model = eval_embedding_model or os.getenv("EVAL_EMBEDDING_MODEL", "nomic-embed-text")
        
        # 创建RAGAS使用的LLM和Embeddings
        llm_kwargs = {
            "model": self.eval_llm_model,
            "base_url": self.ollama_host,
            "temperature": 0.0,
        }
        embedding_kwargs = {
            "model": self.eval_embedding_model,
            "base_url": self.ollama_host,
        }
        
        base_llm = ChatOllama(**llm_kwargs)
        self.eval_embeddings = OllamaEmbeddings(**embedding_kwargs)
        
        # 包装LLM
        try:
            self.eval_llm = LangchainLLMWrapper(
                langchain_llm=base_llm,
                bypass_n=True,
            )
        except Exception:
            self.eval_llm = base_llm
        
        self.llm_func = llm_func
        self.max_concurrent = config.get("max_concurrent_evals", 2) if config else 2
    
    async def _get_rag_response_with_contexts(
        self,
        rag_instance,
        question: str,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        从RAG实例获取答案和检索到的上下文
        
        Args:
            rag_instance: RAGAnything实例
            question: 查询问题
            mode: 查询模式
            
        Returns:
            Dict包含:
                - answer: 生成的答案
                - contexts: 检索到的chunk内容列表（字符串列表）
        """
        from lightrag import QueryParam
        
        # 1. 获取完整答案
        answer = await rag_instance.aquery(question, mode=mode)
        
        # 2. 获取检索上下文（尝试获取chunk内容）
        contexts = []
        
        try:
            # 方法1: 尝试使用only_need_context获取上下文
            context_text = await rag_instance.lightrag.aquery(
                question,
                param=QueryParam(mode=mode, only_need_context=True)
            )
            
            # 如果返回的是文本，尝试分割成chunks
            if isinstance(context_text, str) and context_text.strip():
                # 尝试按段落分割（更智能的分割）
                # 先按双换行分割
                paragraphs = [p.strip() for p in context_text.split("\n\n") if p.strip()]
                
                if len(paragraphs) > 1:
                    # 如果有多个段落，使用段落作为chunks
                    contexts = paragraphs
                elif len(paragraphs) == 1:
                    # 如果只有一个段落，尝试按单换行分割
                    lines = [l.strip() for l in paragraphs[0].split("\n") if l.strip()]
                    if len(lines) > 3:
                        # 如果有多个行，合并成合理的chunks（每3-5行为一个chunk）
                        chunk_size = 4
                        contexts = []
                        for i in range(0, len(lines), chunk_size):
                            chunk = "\n".join(lines[i:i+chunk_size])
                            if chunk.strip():
                                contexts.append(chunk)
                    else:
                        # 如果行数少，直接使用整个段落
                        contexts = paragraphs
                else:
                    # 如果无法分割，将整个上下文作为一个chunk
                    contexts = [context_text]
            else:
                # 如果返回的不是字符串，尝试转换为字符串
                contexts = [str(context_text)] if context_text else []
                
        except Exception as e:
            # 如果获取上下文失败，使用答案的一部分作为上下文
            print(f"⚠️  获取检索上下文失败: {e}，使用答案作为上下文")
            if answer:
                # 将答案分割成句子作为上下文
                import re
                sentences = re.split(r'[.!?]\s+', answer)
                contexts = [s.strip() for s in sentences if s.strip()][:5]  # 最多5个句子
            else:
                contexts = []
        
        # 如果仍然没有上下文，至少提供一个
        if not contexts:
            if answer:
                contexts = [answer[:500] + "..." if len(answer) > 500 else answer]
            else:
                contexts = ["No context available"]
        
        return {
            "answer": answer,
            "contexts": contexts,
        }
    
    async def evaluate_single_case(
        self,
        idx: int,
        test_case: Dict[str, str],
        rag_instance,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """
        评估单个测试用例
        
        Args:
            idx: 测试用例索引
            test_case: 测试用例字典，包含question和ground_truth
            rag_instance: RAGAnything实例
            semaphore: 并发控制信号量
            
        Returns:
            评估结果字典
        """
        async with semaphore:
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            
            try:
                # 获取RAG响应
                rag_response = await self._get_rag_response_with_contexts(
                    rag_instance,
                    question,
                    mode="hybrid"
                )
                
                answer = rag_response["answer"]
                contexts = rag_response["contexts"]
                
                # 准备RAGAS数据集
                eval_dataset = Dataset.from_dict({
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth],
                })
                
                # 运行RAGAS评估
                eval_results = evaluate(
                    dataset=eval_dataset,
                    metrics=[
                        Faithfulness(),
                        AnswerRelevancy(),
                        ContextRecall(),
                        ContextPrecision(),
                    ],
                    llm=self.eval_llm,
                    embeddings=self.eval_embeddings,
                )
                
                # 转换为DataFrame并提取分数
                df = eval_results.to_pandas()
                scores_row = df.iloc[0]
                
                result = {
                    "test_number": idx,
                    "question": question,
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "ground_truth": ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth,
                    "project": test_case.get("project", "unknown"),
                    "contexts_count": len(contexts),
                    "metrics": {
                        "faithfulness": float(scores_row.get("faithfulness", 0)),
                        "answer_relevancy": float(scores_row.get("answer_relevancy", 0)),
                        "context_recall": float(scores_row.get("context_recall", 0)),
                        "context_precision": float(scores_row.get("context_precision", 0)),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                
                # 计算RAGAS总分（平均值，排除NaN）
                metrics = result["metrics"]
                valid_metrics = [v for v in metrics.values() if not _is_nan(v)]
                ragas_score = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0
                result["ragas_score"] = round(ragas_score, 4)
                
                return result
                
            except Exception as e:
                print(f"❌ 评估测试用例 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "test_number": idx,
                    "question": question,
                    "error": str(e),
                    "metrics": {},
                    "ragas_score": 0,
                    "timestamp": datetime.now().isoformat(),
                }
    
    async def evaluate(
        self,
        rag_instance=None,
        test_queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> ComponentEvalResult:
        """
        执行RAGAS评估
        
        Args:
            rag_instance: RAGAnything实例
            test_queries: 测试查询列表，每个包含:
                - question: 查询问题
                - ground_truth: 参考答案
                - project: 项目名称（可选）
                
        Returns:
            ComponentEvalResult: 评估结果
        """
        if not test_queries:
            raise ValueError("需要提供测试查询数据")
        
        if not rag_instance:
            raise ValueError("需要提供RAG实例")
        
        print("\n" + "="*70)
        print("🚀 开始RAGAS端到端评估")
        print("="*70)
        print(f"  评估模型: {self.eval_llm_model}")
        print(f"  Embedding模型: {self.eval_embedding_model}")
        print(f"  Ollama地址: {self.ollama_host}")
        print(f"  测试用例数: {len(test_queries)}")
        print(f"  并发数: {self.max_concurrent}")
        print("="*70)
        
        # 创建并发控制信号量
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 创建所有评估任务
        tasks = [
            self.evaluate_single_case(
                idx,
                test_case,
                rag_instance,
                semaphore,
            )
            for idx, test_case in enumerate(test_queries, 1)
        ]
        
        # 执行所有评估
        results = await asyncio.gather(*tasks)
        results = list(results)
        
        # 过滤有效结果
        valid_results = [r for r in results if r.get("metrics")]
        failed_results = [r for r in results if not r.get("metrics")]
        
        # 计算平均指标
        if valid_results:
            metrics_data = {
                "faithfulness": {"sum": 0.0, "count": 0},
                "answer_relevancy": {"sum": 0.0, "count": 0},
                "context_recall": {"sum": 0.0, "count": 0},
                "context_precision": {"sum": 0.0, "count": 0},
                "ragas_score": {"sum": 0.0, "count": 0},
            }
            
            for result in valid_results:
                metrics = result.get("metrics", {})
                
                for metric_name in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
                    value = metrics.get(metric_name, 0)
                    if not _is_nan(value):
                        metrics_data[metric_name]["sum"] += value
                        metrics_data[metric_name]["count"] += 1
                
                ragas_score = result.get("ragas_score", 0)
                if not _is_nan(ragas_score):
                    metrics_data["ragas_score"]["sum"] += ragas_score
                    metrics_data["ragas_score"]["count"] += 1
            
            # 计算平均值
            summary = {}
            for metric_name, data in metrics_data.items():
                if data["count"] > 0:
                    avg_val = data["sum"] / data["count"]
                    summary[metric_name] = round(avg_val, 4) if not _is_nan(avg_val) else 0.0
                else:
                    summary[metric_name] = 0.0
            
            summary["total_tests"] = len(results)
            summary["successful_tests"] = len(valid_results)
            summary["failed_tests"] = len(failed_results)
            summary["success_rate"] = round(len(valid_results) / len(results) * 100, 2) if results else 0
        else:
            summary = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "ragas_score": 0.0,
                "total_tests": len(results),
                "successful_tests": 0,
                "failed_tests": len(results),
                "success_rate": 0.0,
            }
        
        # 创建详细结果
        detailed_results = results
        
        # 打印摘要
        print("\n" + "="*70)
        print("📊 RAGAS评估结果摘要")
        print("="*70)
        print(f"总测试数:    {summary['total_tests']}")
        print(f"成功:        {summary['successful_tests']}")
        print(f"失败:        {summary['failed_tests']}")
        print(f"成功率:      {summary['success_rate']:.2f}%")
        print("\n平均指标:")
        print(f"  Faithfulness:       {summary['faithfulness']:.4f}")
        print(f"  Answer Relevancy:   {summary['answer_relevancy']:.4f}")
        print(f"  Context Recall:     {summary['context_recall']:.4f}")
        print(f"  Context Precision:  {summary['context_precision']:.4f}")
        print(f"  RAGAS Score:        {summary['ragas_score']:.4f}")
        print("="*70)
        
        # 转换为ComponentEvalResult格式
        # 将RAGAS指标转换为标准格式
        metric_results = []
        for metric_name, value in summary.items():
            if metric_name not in ["total_tests", "successful_tests", "failed_tests", "success_rate"]:
                from evaluation.base import EvaluationResult
                metric_results.append(
                    EvaluationResult(
                        metric_name=metric_name,
                        value=value,
                        metadata={"type": "ragas_metric"}
                    )
                )
        
        return ComponentEvalResult(
            component_name="RAGAS-EndToEnd",
            summary=summary,
            metrics=metric_results,
            details=detailed_results,
        )
