#!/usr/bin/env python
"""
RAG-Anything 评估系统使用示例
演示如何使用评估系统对 RAG 进行全面评估
"""

import os
import asyncio
import sys
from pathlib import Path
from functools import partial

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv, dotenv_values
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnything, RAGAnythingConfig

# 导入评估模块
from evaluation import (
    RAGEvaluator,
    EvaluationConfig,
    DatasetLoader,
)

load_dotenv(dotenv_path=".env", override=False)


async def main():
    """主函数"""
    
    print("\n" + "="*70)
    print("RAG-Anything 评估系统示例")
    print("="*70)
    
    # 1. 初始化 RAG 实例（加载已有的知识图谱）
    print("\n📦 步骤 1: 初始化 RAG 实例...")
    
    working_dir = "./rag_storage"
    
    # 配置 Ollama
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    llm_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    if llm_host.endswith("/v1"):
        llm_host = llm_host[:-3]
    
    vision_model = os.getenv("VISION_MODEL", "qwen3-vl:8b")
    
    # 清理环境变量冲突
    for key in ["EMBEDDING_DIM", "EMBEDDING_MODEL", "LLM_MODEL"]:
        if key in os.environ:
            os.environ.pop(key)
    
    env_values = dotenv_values(".env")
    embedding_dim = int(env_values.get("EMBEDDING_DIM", 768))
    embedding_model = env_values.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
    embedding_host = env_values.get("EMBEDDING_BINDING_HOST", llm_host)
    if embedding_host.endswith("/v1"):
        embedding_host = embedding_host[:-3]
    
    # 创建 Embedding 函数
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=512,
        func=partial(
            ollama_embed.func,
            embed_model=embedding_model,
            host=embedding_host,
            timeout=1200,
        ),
    )
    
    # 创建 RAG 配置
    rag_config = RAGAnythingConfig(
        working_dir=working_dir,
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # 初始化 RAGAnything
    rag = RAGAnything(
        config=rag_config,
        llm_model_func=ollama_model_complete,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": llm_model,
            "summary_max_tokens": 2048,
            "chunk_token_size": 200,
            "chunk_overlap_token_size": 30,
            "llm_model_kwargs": {
                "host": llm_host,
                "options": {"num_ctx": 4096},
                "timeout": 1200,
            },
            "llm_model_max_async": 1,
            "default_llm_timeout": 1200,
        }
    )
    
    # 加载知识图谱
    await rag._ensure_lightrag_initialized()
    print(f"✅ RAG 实例已加载: {working_dir}")
    
    # 2. 创建或加载评估数据集
    print("\n📊 步骤 2: 准备评估数据集...")
    
    dataset_path = "./evaluation_dataset_sample.json"
    
    # 如果数据集不存在，创建示例数据集
    if not os.path.exists(dataset_path):
        print(f"📝 创建示例数据集: {dataset_path}")
        DatasetLoader.create_sample_dataset(output_path=dataset_path)
    
    # 加载数据集
    dataset = DatasetLoader.load_from_json(dataset_path)
    print(f"✅ 数据集已加载: {len(dataset)} 个查询")
    
    # 3. 配置评估系统
    print("\n⚙️  步骤 3: 配置评估系统...")
    
    eval_config = EvaluationConfig(
        working_dir=working_dir,
        output_dir="./evaluation_results",
        # 组件级评估
        enable_retriever_eval=True,
        enable_generator_eval=True,
        # 端到端评估
        enable_qa_eval=True,
        enable_multimodal_eval=True,
        # 指标配置
        retrieval_top_k=[1, 3, 5, 10],
        use_rouge=True,
        use_bleu=False,
        use_bertscore=False,  # 需要额外安装 bert-score
        use_llm_judge=True,
        llm_judge_aspects=["faithfulness", "relevance", "coherence"],
        # 多模态配置
        multimodal_types=["image", "table", "equation"],
        # 其他配置
        save_detailed_results=True,
    )
    
    print(f"✅ 评估配置完成")
    print(f"  - 检索评估: {eval_config.enable_retriever_eval}")
    print(f"  - 生成评估: {eval_config.enable_generator_eval}")
    print(f"  - 问答评估: {eval_config.enable_qa_eval}")
    print(f"  - 多模态评估: {eval_config.enable_multimodal_eval}")
    
    # 4. 创建 LLM 评判函数（用于 LLM-as-a-Judge）
    async def llm_judge_func(prompt):
        """LLM 评判函数"""
        result = await ollama_model_complete(
            prompt,
            hashing_kv=rag.lightrag.llm_response_cache,
            host=llm_host,
            timeout=600,
            options={"num_ctx": 4096},
        )
        return result
    
    # 5. 执行评估
    print("\n🚀 步骤 4: 执行评估...")
    
    evaluator = RAGEvaluator(
        rag_instance=rag,
        config=eval_config,
        llm_func=llm_judge_func
    )
    
    # 执行完整评估
    results = await evaluator.evaluate_all(dataset=dataset)
    
    # 6. 生成报告
    print("\n📝 步骤 5: 生成评估报告...")
    report_path = evaluator.generate_report(output_format="markdown")
    
    print("\n" + "="*70)
    print("✅ 评估完成！")
    print("="*70)
    print(f"\n结果文件:")
    print(f"  - JSON: {eval_config.output_dir}/evaluation_results_*.json")
    print(f"  - 报告: {report_path}")
    
    # 7. 演示快速评估（可选）
    print("\n" + "="*70)
    print("💡 快速评估示例")
    print("="*70)
    
    quick_questions = [
        "What is the main contribution of this paper?",
        "What method is proposed in the paper?"
    ]
    
    quick_ground_truths = [
        "The paper proposes a Neuro-TF approach for metasurface design.",
        "The Neuro-TF approach combines neural networks with transfer functions."
    ]
    
    quick_result = await evaluator.quick_eval(
        questions=quick_questions,
        ground_truths=quick_ground_truths
    )
    
    if quick_result:
        print(f"\n✅ 快速评估完成:")
        for metric_name, value in quick_result.get("summary", {}).items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
