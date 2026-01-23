#!/usr/bin/env python3
"""
RAGAS评估示例
使用RAGAS框架进行端到端评估

数据集格式（test_cases）:
{
  "test_cases": [
    {
      "question": "问题文本",
      "ground_truth": "参考答案",
      "project": "项目名称（可选）"
    }
  ]
}

使用方法:
    python evaluation/example/ragas_evaluation_example.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from evaluation import RAGEvaluator, EvaluationConfig
from evaluation.dataset import DatasetLoader


async def main():
    """主函数"""
    
    # 1. 配置RAG实例
    print("="*70)
    print("🔧 初始化RAG实例")
    print("="*70)
    
    # 使用Ollama模型
    from examples.raganything_example import (
        ollama_model_complete,
        vision_model_func,
        embedding_func,
    )
    
    config = RAGAnythingConfig()
    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_model_complete,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": "qwen2.5:7b-instruct",
            "chunk_token_size": 200,
            "chunk_overlap_token_size": 30,
        }
    )
    
    # 设置工作目录（使用已处理的数据）
    working_dir = "./rag_storage1"
    if hasattr(rag, 'lightrag') and rag.lightrag:
        print(f"✅ RAG实例已初始化，工作目录: {working_dir}")
    else:
        print("⚠️  警告: LightRAG未初始化，请确保数据已处理")
    
    # 2. 加载数据集（支持test_cases格式）
    print("\n" + "="*70)
    print("📊 加载评估数据集")
    print("="*70)
    
    dataset_path = "./evaluation/dataset/evaluation_dataset_from_storage.json"
    
    # 检查数据集格式
    import json
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_data = json.load(f)
    
    # 如果数据集是queries格式，转换为test_cases格式
    if "queries" in dataset_data and "test_cases" not in dataset_data:
        print("📝 转换数据集格式: queries -> test_cases")
        test_cases = []
        for query in dataset_data.get("queries", []):
            test_case = {
                "question": query.get("question", ""),
                "ground_truth": query.get("ground_truth", query.get("answer", "")),
            }
            if "project" in query:
                test_case["project"] = query["project"]
            test_cases.append(test_case)
        
        dataset_data["test_cases"] = test_cases
        # 保存转换后的数据集
        converted_path = dataset_path.replace(".json", "_ragas.json")
        with open(converted_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, indent=2, ensure_ascii=False)
        dataset_path = converted_path
        print(f"✅ 转换后的数据集已保存: {converted_path}")
    
    dataset = DatasetLoader.load_from_json(dataset_path)
    print(f"✅ 数据集加载完成: {len(dataset)} 个测试用例")
    
    # 3. 配置评估器
    print("\n" + "="*70)
    print("⚙️  配置评估器")
    print("="*70)
    
    eval_config = EvaluationConfig(
        working_dir=working_dir,
        output_dir="./evaluation_results",
        # 禁用其他评估，只启用RAGAS
        enable_retriever_eval=False,
        enable_generator_eval=False,
        enable_entity_relation_eval=False,
        enable_knowledge_graph_eval=False,
        enable_chunk_embedding_eval=False,
        enable_qa_eval=False,
        enable_multimodal_eval=False,
        enable_ragas_eval=True,  # 启用RAGAS评估
        # RAGAS配置
        ragas_llm_model="qwen2.5:7b-instruct",
        ragas_embedding_model="nomic-embed-text",
        ragas_ollama_host="http://localhost:11434",
        max_concurrent_evals=2,  # RAGAS评估并发数
    )
    
    # 4. 创建评估器
    evaluator = RAGEvaluator(
        rag_instance=rag,
        config=eval_config,
        llm_func=ollama_model_complete,
    )
    
    # 5. 运行评估
    print("\n" + "="*70)
    print("🚀 开始RAGAS评估")
    print("="*70)
    
    results = await evaluator.evaluate_all(dataset=dataset)
    
    # 6. 显示结果摘要
    print("\n" + "="*70)
    print("📊 RAGAS评估结果摘要")
    print("="*70)
    
    if "end_to_end" in results and "ragas" in results["end_to_end"]:
        ragas_result = results["end_to_end"]["ragas"]
        summary = ragas_result.get("summary", {})
        
        print(f"\n总测试数:    {summary.get('total_tests', 0)}")
        print(f"成功:        {summary.get('successful_tests', 0)}")
        print(f"失败:        {summary.get('failed_tests', 0)}")
        print(f"成功率:      {summary.get('success_rate', 0):.2f}%")
        
        print("\n平均指标:")
        print(f"  Faithfulness:       {summary.get('faithfulness', 0):.4f}")
        print(f"  Answer Relevancy:   {summary.get('answer_relevancy', 0):.4f}")
        print(f"  Context Recall:     {summary.get('context_recall', 0):.4f}")
        print(f"  Context Precision:  {summary.get('context_precision', 0):.4f}")
        print(f"  RAGAS Score:        {summary.get('ragas_score', 0):.4f}")
    
    print("\n" + "="*70)
    print("✅ 评估完成！")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
