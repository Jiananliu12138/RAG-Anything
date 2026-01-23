"""
Chunks 和 Embeddings 质量评估器
评估文本分块和向量嵌入的质量
"""

import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from evaluation.base import BaseEvaluator, ComponentEvalResult, EvaluationResult


class ChunkEmbeddingEvaluator(BaseEvaluator):
    """Chunks 和 Embeddings 质量评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="ChunkEmbedding", config=config)
        self.storage_dir = config.get("working_dir", "./rag_storage") if config else "./rag_storage"
    
    async def evaluate(self, rag_instance=None, **kwargs) -> ComponentEvalResult:
        """
        评估 chunks 和 embeddings 质量
        
        Returns:
            ComponentEvalResult: 评估结果
        """
        print(f"\n📦 Chunks & Embeddings 评估:")
        
        # 加载数据
        text_chunks = self._load_text_chunks()
        vdb_chunks = self._load_vdb_chunks()
        vdb_entities = self._load_vdb_entities()
        
        print(f"  - 文本Chunks数: {len(text_chunks)}")
        print(f"  - 向量Chunks数: {len(vdb_chunks)}")
        print(f"  - 向量实体数: {len(vdb_entities)}")
        
        all_results = []
        
        # 1. Chunk 大小分布
        chunk_size_metric = self._evaluate_chunk_sizes(text_chunks)
        all_results.append(chunk_size_metric)
        
        # 2. Chunk Token 数量分布
        chunk_token_metric = self._evaluate_chunk_tokens(text_chunks)
        all_results.append(chunk_token_metric)
        
        # 3. Embedding 覆盖率
        embedding_coverage = self._evaluate_embedding_coverage(text_chunks, vdb_chunks)
        all_results.append(embedding_coverage)
        
        # 4. Embedding 向量质量
        if vdb_chunks:
            embedding_quality = self._evaluate_embedding_quality(vdb_chunks)
            all_results.append(embedding_quality)
        
        # 5. 实体 Embedding 覆盖率
        entity_embedding_coverage = self._evaluate_entity_embeddings(vdb_entities)
        all_results.append(entity_embedding_coverage)
        
        # 聚合结果
        summary = self._aggregate_results(all_results)
        summary["total_chunks"] = len(text_chunks)
        summary["total_chunk_embeddings"] = len(vdb_chunks)
        summary["total_entity_embeddings"] = len(vdb_entities)
        
        # 详细统计
        details = {
            "chunk_statistics": chunk_size_metric.metadata,
            "token_statistics": chunk_token_metric.metadata,
            "embedding_coverage": embedding_coverage.metadata,
            "embedding_quality": embedding_quality.metadata if vdb_chunks else {},
            "entity_embeddings": entity_embedding_coverage.metadata
        }
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=all_results,
            summary=summary,
            details=[details] if self.config.get("save_detailed_results", True) else None
        )
    
    def _load_text_chunks(self) -> Dict:
        """加载文本chunks"""
        filepath = Path(self.storage_dir) / "kv_store_text_chunks.json"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载文本chunks失败: {e}")
            return {}
    
    def _load_vdb_chunks(self) -> Dict:
        """加载向量数据库chunks"""
        filepath = Path(self.storage_dir) / "vdb_chunks.json"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载向量chunks失败: {e}")
            return {}
    
    def _load_vdb_entities(self) -> Dict:
        """加载向量数据库实体"""
        filepath = Path(self.storage_dir) / "vdb_entities.json"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载向量实体失败: {e}")
            return {}
    
    def _evaluate_chunk_sizes(self, text_chunks: Dict) -> EvaluationResult:
        """评估chunk大小分布（字符数）"""
        if not text_chunks:
            return EvaluationResult("ChunkSizeDistribution", 0.0)
        
        sizes = [len(chunk.get("content", "")) for chunk in text_chunks.values()]
        
        avg_size = np.mean(sizes) if sizes else 0
        std_size = np.std(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        
        # 理想chunk大小：500-1000字符
        optimal_range = (300, 1500)
        in_range = sum(1 for s in sizes if optimal_range[0] <= s <= optimal_range[1])
        quality_score = in_range / len(sizes) if sizes else 0.0
        
        return EvaluationResult(
            metric_name="ChunkSizeDistribution",
            value=quality_score,
            metadata={
                "avg_size": float(avg_size),
                "std_size": float(std_size),
                "min_size": min_size,
                "max_size": max_size,
                "in_optimal_range": in_range,
                "total_chunks": len(sizes)
            }
        )
    
    def _evaluate_chunk_tokens(self, text_chunks: Dict) -> EvaluationResult:
        """评估chunk token数量分布"""
        if not text_chunks:
            return EvaluationResult("ChunkTokenDistribution", 0.0)
        
        tokens = [chunk.get("tokens", 0) for chunk in text_chunks.values()]
        
        avg_tokens = np.mean(tokens) if tokens else 0
        std_tokens = np.std(tokens) if tokens else 0
        
        # 理想token数：150-250 (配置的chunk_token_size=200)
        target_tokens = 200
        deviations = [abs(t - target_tokens) / target_tokens for t in tokens if t > 0]
        avg_deviation = np.mean(deviations) if deviations else 0
        
        # 偏差越小越好
        quality_score = max(0.0, 1.0 - avg_deviation)
        
        return EvaluationResult(
            metric_name="ChunkTokenDistribution",
            value=quality_score,
            metadata={
                "avg_tokens": float(avg_tokens),
                "std_tokens": float(std_tokens),
                "min_tokens": min(tokens) if tokens else 0,
                "max_tokens": max(tokens) if tokens else 0,
                "target_tokens": target_tokens,
                "avg_deviation": float(avg_deviation)
            }
        )
    
    def _evaluate_embedding_coverage(self, text_chunks: Dict, vdb_chunks: Dict) -> EvaluationResult:
        """评估embedding覆盖率：有多少chunks有对应的embedding"""
        if not text_chunks:
            return EvaluationResult("EmbeddingCoverage", 0.0)
        
        text_chunk_ids = set(text_chunks.keys())
        vdb_chunk_ids = set(vdb_chunks.keys())
        
        covered = len(text_chunk_ids & vdb_chunk_ids)
        coverage = covered / len(text_chunk_ids) if text_chunk_ids else 0.0
        
        return EvaluationResult(
            metric_name="EmbeddingCoverage",
            value=coverage,
            metadata={
                "total_text_chunks": len(text_chunk_ids),
                "chunks_with_embeddings": covered,
                "coverage_percentage": coverage * 100
            }
        )
    
    def _evaluate_embedding_quality(self, vdb_chunks: Dict) -> EvaluationResult:
        """评估embedding向量质量"""
        if not vdb_chunks:
            return EvaluationResult("EmbeddingQuality", 0.0)
        
        # 提取embedding向量
        embeddings = []
        for chunk_data in vdb_chunks.values():
            if isinstance(chunk_data, dict):
                # 尝试不同的可能字段
                emb = chunk_data.get("embedding") or chunk_data.get("vector") or chunk_data.get("__vector__")
                if emb and isinstance(emb, (list, np.ndarray)):
                    embeddings.append(np.array(emb))
        
        if not embeddings:
            return EvaluationResult(
                metric_name="EmbeddingQuality",
                value=0.0,
                metadata={"note": "No embedding vectors found"}
            )
        
        # 计算向量统计
        embeddings = np.array(embeddings)
        
        # 1. 向量范数分布
        norms = np.linalg.norm(embeddings, axis=1)
        avg_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        
        # 2. 向量维度
        emb_dim = embeddings.shape[1]
        
        # 3. 零值比例（好的embedding应该很少有零）
        zero_ratio = float(np.mean(embeddings == 0))
        
        # 4. 向量相似度分布（抽样计算）
        sample_size = min(100, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embs = embeddings[sample_indices]
        
        # 计算余弦相似度
        norms_sample = np.linalg.norm(sample_embs, axis=1, keepdims=True)
        normalized = sample_embs / (norms_sample + 1e-8)
        similarities = np.dot(normalized, normalized.T)
        
        # 排除对角线（自己和自己）
        mask = ~np.eye(sample_size, dtype=bool)
        avg_similarity = float(np.mean(similarities[mask]))
        
        # 质量分数：基于多个因素
        # 1. 范数稳定性（标准差小）
        norm_stability = 1.0 / (1.0 + std_norm / avg_norm) if avg_norm > 0 else 0.0
        # 2. 非零率（应该>0.8）
        non_zero_score = 1.0 - zero_ratio
        # 3. 相似度合理性（不应该太高，表示多样性）
        diversity_score = 1.0 - min(1.0, avg_similarity)
        
        quality_score = (norm_stability + non_zero_score + diversity_score) / 3.0
        
        return EvaluationResult(
            metric_name="EmbeddingQuality",
            value=quality_score,
            metadata={
                "embedding_dimension": emb_dim,
                "avg_norm": avg_norm,
                "std_norm": std_norm,
                "zero_ratio": zero_ratio,
                "avg_similarity": avg_similarity,
                "total_embeddings": len(embeddings)
            }
        )
    
    def _evaluate_entity_embeddings(self, vdb_entities: Dict) -> EvaluationResult:
        """评估实体embedding覆盖率"""
        if not vdb_entities:
            return EvaluationResult(
                metric_name="EntityEmbeddingCoverage",
                value=0.0,
                metadata={"total_entity_embeddings": 0}
            )
        
        # 统计有embedding的实体数
        entity_count = len(vdb_entities)
        
        return EvaluationResult(
            metric_name="EntityEmbeddingCoverage",
            value=1.0 if entity_count > 0 else 0.0,
            metadata={
                "total_entity_embeddings": entity_count,
                "note": "All entities in vdb have embeddings"
            }
        )
