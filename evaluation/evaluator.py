"""
RAG 评估系统主评估器
整合组件级和端到端评估，提供统一的评估接口
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from evaluation.config import EvaluationConfig
from evaluation.components import RetrieverEvaluator, GeneratorEvaluator
from evaluation.end_to_end import QAEvaluator, MultimodalEvaluator
from evaluation.dataset import DatasetLoader, EvaluationDataset


class RAGEvaluator:
    """
    RAG 评估系统主类
    提供完整的评估流程管理
    """
    
    def __init__(
        self,
        rag_instance,
        config: Optional[EvaluationConfig] = None,
        llm_func = None
    ):
        """
        初始化评估器
        
        Args:
            rag_instance: RAGAnything 实例
            config: 评估配置
            llm_func: LLM 函数（用于 LLM Judge）
        """
        self.rag = rag_instance
        self.config = config or EvaluationConfig()
        self.llm_func = llm_func
        
        # 初始化各个评估器
        self._init_evaluators()
        
        # 评估结果存储
        self.results = {}
    
    def _init_evaluators(self):
        """初始化所有评估器"""
        config_dict = {
            "retrieval_top_k": self.config.retrieval_top_k,
            "use_rouge": self.config.use_rouge,
            "use_bleu": self.config.use_bleu,
            "use_bertscore": self.config.use_bertscore,
            "use_llm_judge": self.config.use_llm_judge,
            "llm_judge_aspects": self.config.llm_judge_aspects,
            "save_detailed_results": self.config.save_detailed_results,
            "multimodal_types": self.config.multimodal_types,
            "enable_retriever_eval": self.config.enable_retriever_eval,
            "enable_generator_eval": self.config.enable_generator_eval,
        }
        
        # 组件级评估器
        if self.config.enable_retriever_eval:
            self.retriever_evaluator = RetrieverEvaluator(config=config_dict)
        
        if self.config.enable_generator_eval:
            self.generator_evaluator = GeneratorEvaluator(
                config=config_dict, 
                llm_func=self.llm_func
            )
        
        # 端到端评估器
        if self.config.enable_qa_eval:
            self.qa_evaluator = QAEvaluator(
                config=config_dict, 
                llm_func=self.llm_func
            )
        
        if self.config.enable_multimodal_eval:
            self.multimodal_evaluator = MultimodalEvaluator(
                config=config_dict,
                llm_func=self.llm_func
            )
    
    async def evaluate_all(
        self,
        dataset: Optional[EvaluationDataset] = None,
        dataset_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行完整评估流程
        
        Args:
            dataset: 评估数据集对象
            dataset_path: 数据集文件路径（如果 dataset 未提供）
            
        Returns:
            Dict[str, Any]: 完整的评估结果
        """
        print("\n" + "="*70)
        print("🚀 RAG-Anything 评估系统")
        print("="*70)
        
        # 加载数据集
        if dataset is None:
            if dataset_path:
                dataset = DatasetLoader.load_from_json(dataset_path)
            elif self.config.dataset_path:
                dataset = DatasetLoader.load_from_json(self.config.dataset_path)
            else:
                raise ValueError("需要提供数据集或数据集路径")
        
        print(f"\n📊 数据集信息:")
        print(f"  - 名称: {dataset.metadata.get('name', 'N/A')}")
        print(f"  - 查询数量: {len(dataset)}")
        
        results = {
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),
                "dataset": dataset.metadata,
                "config": self.config.__dict__
            },
            "component_level": {},
            "end_to_end": {}
        }
        
        # 组件级评估
        if self.config.enable_retriever_eval or self.config.enable_generator_eval:
            print("\n" + "="*70)
            print("📈 组件级评估")
            print("="*70)
            
            if self.config.enable_retriever_eval:
                retriever_result = await self.retriever_evaluator.evaluate(
                    rag_instance=self.rag,
                    test_queries=dataset.get_all_queries()
                )
                results["component_level"]["retriever"] = retriever_result.to_dict()
                print(f"\n✅ 检索器评估完成")
                self._print_metrics_summary(retriever_result.summary)
            
            if self.config.enable_generator_eval:
                generator_result = await self.generator_evaluator.evaluate(
                    rag_instance=self.rag,
                    test_queries=dataset.get_all_queries()
                )
                results["component_level"]["generator"] = generator_result.to_dict()
                print(f"\n✅ 生成器评估完成")
                self._print_metrics_summary(generator_result.summary)
        
        # 端到端评估
        print("\n" + "="*70)
        print("🎯 端到端评估")
        print("="*70)
        
        if self.config.enable_qa_eval:
            qa_results = await self.qa_evaluator.evaluate(
                rag_instance=self.rag,
                test_queries=dataset.get_all_queries()
            )
            results["end_to_end"]["qa"] = {
                k: v.to_dict() for k, v in qa_results.items()
            }
            print(f"\n✅ 问答评估完成")
            self.qa_evaluator.print_summary(qa_results)
        
        if self.config.enable_multimodal_eval:
            multimodal_queries = dataset.filter_by_type("multimodal")
            if multimodal_queries:
                multimodal_result = await self.multimodal_evaluator.evaluate(
                    rag_instance=self.rag,
                    test_queries=multimodal_queries
                )
                results["end_to_end"]["multimodal"] = multimodal_result.to_dict()
                print(f"\n✅ 多模态评估完成")
                self._print_metrics_summary(multimodal_result.summary)
        
        # 保存结果
        self.results = results
        self.save_results()
        
        return results
    
    def _print_metrics_summary(self, summary: Dict[str, float]):
        """打印指标摘要"""
        for metric_name, value in summary.items():
            print(f"  {metric_name}: {value:.4f}")
    
    def save_results(self, output_path: Optional[str] = None):
        """
        保存评估结果到文件
        
        Args:
            output_path: 输出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.output_dir) / f"evaluation_results_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 评估结果已保存: {output_path}")
        return output_path
    
    async def quick_eval(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        快速评估（无需完整数据集）
        
        Args:
            questions: 问题列表
            ground_truths: 参考答案列表（可选）
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 构造临时数据集
        queries = []
        for i, question in enumerate(questions):
            query_item = {
                "id": f"quick_q{i+1}",
                "type": "text",
                "question": question
            }
            if ground_truths and i < len(ground_truths):
                query_item["ground_truth"] = ground_truths[i]
            queries.append(query_item)
        
        temp_dataset = EvaluationDataset({"queries": queries})
        
        # 执行评估（仅生成器）
        if self.config.enable_generator_eval and ground_truths:
            result = await self.generator_evaluator.evaluate(
                rag_instance=self.rag,
                test_queries=queries
            )
            return result.to_dict()
        else:
            print("⚠️  快速评估需要 ground_truths 且启用生成器评估")
            return {}
    
    def generate_report(self, output_format: str = "markdown") -> str:
        """
        生成评估报告
        
        Args:
            output_format: 报告格式 (markdown, html, txt)
            
        Returns:
            str: 报告文件路径
        """
        if not self.results:
            raise ValueError("没有评估结果，请先运行评估")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "markdown":
            report_path = Path(self.config.output_dir) / f"evaluation_report_{timestamp}.md"
            content = self._generate_markdown_report()
        else:
            report_path = Path(self.config.output_dir) / f"evaluation_report_{timestamp}.txt"
            content = self._generate_text_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n📄 评估报告已生成: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self) -> str:
        """生成 Markdown 格式报告"""
        lines = []
        lines.append("# RAG-Anything 评估报告\n")
        lines.append(f"**评估时间**: {self.results['metadata']['evaluation_time']}\n")
        lines.append(f"**数据集**: {self.results['metadata']['dataset'].get('name', 'N/A')}\n")
        lines.append("\n---\n")
        
        # 组件级评估
        if "component_level" in self.results:
            lines.append("\n## 组件级评估\n")
            for component, data in self.results["component_level"].items():
                lines.append(f"\n### {component.upper()}\n")
                lines.append("\n| 指标 | 值 |")
                lines.append("\n|------|-----|")
                for metric_name, value in data.get("summary", {}).items():
                    lines.append(f"\n| {metric_name} | {value:.4f} |")
        
        # 端到端评估
        if "end_to_end" in self.results:
            lines.append("\n\n## 端到端评估\n")
            for eval_type, data in self.results["end_to_end"].items():
                lines.append(f"\n### {eval_type.upper()}\n")
                if isinstance(data, dict) and "summary" in data:
                    lines.append("\n| 指标 | 值 |")
                    lines.append("\n|------|-----|")
                    for metric_name, value in data.get("summary", {}).items():
                        lines.append(f"\n| {metric_name} | {value:.4f} |")
        
        return "".join(lines)
    
    def _generate_text_report(self) -> str:
        """生成纯文本格式报告"""
        lines = []
        lines.append("="*70 + "\n")
        lines.append("RAG-Anything 评估报告\n")
        lines.append("="*70 + "\n")
        lines.append(f"\n评估时间: {self.results['metadata']['evaluation_time']}\n")
        
        # 组件级评估
        if "component_level" in self.results:
            lines.append("\n" + "-"*70 + "\n")
            lines.append("组件级评估\n")
            lines.append("-"*70 + "\n")
            for component, data in self.results["component_level"].items():
                lines.append(f"\n{component.upper()}:\n")
                for metric_name, value in data.get("summary", {}).items():
                    lines.append(f"  {metric_name}: {value:.4f}\n")
        
        # 端到端评估
        if "end_to_end" in self.results:
            lines.append("\n" + "-"*70 + "\n")
            lines.append("端到端评估\n")
            lines.append("-"*70 + "\n")
            for eval_type, data in self.results["end_to_end"].items():
                lines.append(f"\n{eval_type.upper()}:\n")
                if isinstance(data, dict) and "summary" in data:
                    for metric_name, value in data.get("summary", {}).items():
                        lines.append(f"  {metric_name}: {value:.4f}\n")
        
        return "".join(lines)
