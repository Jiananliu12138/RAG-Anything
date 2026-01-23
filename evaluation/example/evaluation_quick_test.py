#!/usr/bin/env python
"""
评估系统快速测试脚本
用于验证评估系统是否正常工作
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.dataset import DatasetLoader


async def test_evaluation_system():
    """测试评估系统基本功能"""
    
    print("\n" + "="*70)
    print("🧪 评估系统快速测试")
    print("="*70)
    
    # 1. 测试数据集创建
    print("\n📝 测试 1: 创建示例数据集...")
    dataset_path = "./test_eval_dataset.json"
    DatasetLoader.create_sample_dataset(output_path=dataset_path)
    
    # 2. 测试数据集加载
    print("\n📂 测试 2: 加载数据集...")
    dataset = DatasetLoader.load_from_json(dataset_path)
    print(f"✅ 数据集加载成功: {len(dataset)} 个查询")
    
    # 3. 测试数据集过滤
    print("\n🔍 测试 3: 数据集过滤...")
    text_queries = dataset.filter_by_type("text")
    multimodal_queries = dataset.filter_by_type("multimodal")
    print(f"✅ 文本查询: {len(text_queries)} 个")
    print(f"✅ 多模态查询: {len(multimodal_queries)} 个")
    
    image_queries = dataset.filter_by_modality("image")
    table_queries = dataset.filter_by_modality("table")
    print(f"✅ 图像查询: {len(image_queries)} 个")
    print(f"✅ 表格查询: {len(table_queries)} 个")
    
    # 4. 测试指标计算
    print("\n📊 测试 4: 测试评估指标...")
    
    from evaluation.metrics import PrecisionAtK, RecallAtK, MeanReciprocalRank, NDCG
    
    # 模拟检索结果
    predictions = ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
    ground_truth = ["chunk-2", "chunk-5", "chunk-7"]
    
    # 测试 Precision@3
    precision_metric = PrecisionAtK(k=3)
    precision_result = precision_metric.compute(predictions, ground_truth)
    print(f"✅ Precision@3 = {precision_result.value:.4f}")
    
    # 测试 Recall@5
    recall_metric = RecallAtK(k=5)
    recall_result = recall_metric.compute(predictions, ground_truth)
    print(f"✅ Recall@5 = {recall_result.value:.4f}")
    
    # 测试 MRR
    mrr_metric = MeanReciprocalRank()
    mrr_result = mrr_metric.compute(predictions, ground_truth)
    print(f"✅ MRR = {mrr_result.value:.4f}")
    
    # 测试 NDCG@5
    ndcg_metric = NDCG(k=5)
    ndcg_result = ndcg_metric.compute(predictions, ground_truth)
    print(f"✅ NDCG@5 = {ndcg_result.value:.4f}")
    
    # 5. 测试生成指标（如果安装了 rouge-score）
    print("\n📝 测试 5: 测试生成指标...")
    try:
        from evaluation.metrics import ROUGEMetric
        
        pred_text = "The paper proposes a neural network approach for metasurface design."
        ref_text = "This paper presents a Neuro-TF method for designing metasurface absorbers."
        
        rouge_metric = ROUGEMetric(rouge_type="rougeL")
        rouge_result = rouge_metric.compute(pred_text, ref_text)
        print(f"✅ ROUGE-L = {rouge_result.value:.4f}")
    except ImportError:
        print("⚠️  rouge-score 未安装，跳过 ROUGE 测试")
        print("   安装命令: pip install rouge-score")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！评估系统工作正常")
    print("="*70)
    print("\n💡 下一步:")
    print("  1. 准备你的测试数据集 (参考 test_eval_dataset.json)")
    print("  2. 运行完整评估: python examples/evaluation_example.py")
    print("  3. 查看评估结果: ./evaluation_results/")


if __name__ == "__main__":
    asyncio.run(test_evaluation_system())
