"""
实体和关系提取评估器
评估知识图谱构建过程中的实体和关系提取质量
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from evaluation.base import BaseEvaluator, ComponentEvalResult, EvaluationResult


class EntityRelationEvaluator(BaseEvaluator):
    """实体和关系提取评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="EntityRelation", config=config)
        self.storage_dir = config.get("working_dir", "./rag_storage") if config else "./rag_storage"
    
    async def evaluate(
        self,
        rag_instance=None,
        test_queries: List[Dict[str, Any]] = None,
        **kwargs
    ) -> ComponentEvalResult:
        """
        评估实体和关系提取质量
        
        Args:
            rag_instance: RAGAnything 实例
            test_queries: 测试查询列表（应包含 relevant_entities 和 relevant_relations）
            
        Returns:
            ComponentEvalResult: 评估结果
        """
        if not test_queries:
            raise ValueError("需要提供测试查询数据")
        
        # 加载知识图谱数据
        entity_chunks = self._load_entity_chunks()
        relation_chunks = self._load_relation_chunks()
        
        print(f"\n📊 实体关系评估:")
        print(f"  - 知识图谱中的实体数: {len(entity_chunks)}")
        print(f"  - 知识图谱中的关系数: {len(relation_chunks)}")
        
        all_results = []
        detailed_results = []
        
        # 1. 实体覆盖率评估
        entity_coverage = self._evaluate_entity_coverage(test_queries, entity_chunks)
        all_results.append(entity_coverage)
        
        # 2. 关系覆盖率评估
        relation_coverage = self._evaluate_relation_coverage(test_queries, relation_chunks)
        all_results.append(relation_coverage)
        
        # 3. 实体准确性评估
        entity_accuracy = self._evaluate_entity_accuracy(test_queries, entity_chunks)
        all_results.append(entity_accuracy)
        
        # 4. 关系准确性评估
        relation_accuracy = self._evaluate_relation_accuracy(test_queries, relation_chunks)
        all_results.append(relation_accuracy)
        
        # 5. 实体-Chunk 关联质量
        entity_chunk_quality = self._evaluate_entity_chunk_quality(entity_chunks)
        all_results.append(entity_chunk_quality)
        
        # 详细结果
        for query in test_queries:
            if "relevant_entities" in query or "relevant_relations" in query:
                query_result = {
                    "question": query.get("question", ""),
                    "expected_entities": query.get("relevant_entities", []),
                    "expected_relations": query.get("relevant_relations", []),
                    "entity_found": self._check_entities_exist(
                        query.get("relevant_entities", []), 
                        entity_chunks
                    ),
                    "relation_found": self._check_relations_exist(
                        query.get("relevant_relations", []), 
                        relation_chunks
                    )
                }
                detailed_results.append(query_result)
        
        # 聚合结果
        summary = self._aggregate_results(all_results)
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=all_results,
            summary=summary,
            details=detailed_results if self.config.get("save_detailed_results", True) else None
        )
    
    def _load_entity_chunks(self) -> Dict:
        """加载实体chunks"""
        filepath = Path(self.storage_dir) / "kv_store_entity_chunks.json"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载实体数据失败: {e}")
            return {}
    
    def _load_relation_chunks(self) -> Dict:
        """加载关系chunks"""
        filepath = Path(self.storage_dir) / "kv_store_relation_chunks.json"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载关系数据失败: {e}")
            return {}
    
    def _evaluate_entity_coverage(
        self, 
        test_queries: List[Dict], 
        entity_chunks: Dict
    ) -> EvaluationResult:
        """评估实体覆盖率：测试集中期望的实体有多少在知识图谱中"""
        total_expected = 0
        found = 0
        
        for query in test_queries:
            expected_entities = query.get("relevant_entities", [])
            total_expected += len(expected_entities)
            for entity in expected_entities:
                if entity in entity_chunks:
                    found += 1
        
        coverage = found / total_expected if total_expected > 0 else 0.0
        
        return EvaluationResult(
            metric_name="EntityCoverage",
            value=coverage,
            metadata={
                "found_entities": found,
                "total_expected": total_expected,
                "description": "测试集中期望实体在知识图谱中的覆盖率"
            }
        )
    
    def _evaluate_relation_coverage(
        self, 
        test_queries: List[Dict], 
        relation_chunks: Dict
    ) -> EvaluationResult:
        """评估关系覆盖率"""
        total_expected = 0
        found = 0
        
        for query in test_queries:
            expected_relations = query.get("relevant_relations", [])
            total_expected += len(expected_relations)
            for relation in expected_relations:
                if relation in relation_chunks:
                    found += 1
        
        coverage = found / total_expected if total_expected > 0 else 0.0
        
        return EvaluationResult(
            metric_name="RelationCoverage",
            value=coverage,
            metadata={
                "found_relations": found,
                "total_expected": total_expected,
                "description": "测试集中期望关系在知识图谱中的覆盖率"
            }
        )
    
    def _evaluate_entity_accuracy(
        self, 
        test_queries: List[Dict], 
        entity_chunks: Dict
    ) -> EvaluationResult:
        """评估实体提取准确性：检查实体是否被正确关联到相关chunks"""
        total_checks = 0
        correct = 0
        
        for query in test_queries:
            expected_entities = query.get("relevant_entities", [])
            expected_chunks = query.get("relevant_chunks", [])
            
            for entity in expected_entities:
                if entity in entity_chunks:
                    entity_data = entity_chunks[entity]
                    entity_chunk_ids = set(entity_data.get("chunk_ids", []))
                    expected_chunk_set = set(expected_chunks)
                    
                    # 检查是否有重叠
                    if entity_chunk_ids & expected_chunk_set:
                        correct += 1
                    total_checks += 1
        
        accuracy = correct / total_checks if total_checks > 0 else 0.0
        
        return EvaluationResult(
            metric_name="EntityAccuracy",
            value=accuracy,
            metadata={
                "correct_associations": correct,
                "total_checks": total_checks,
                "description": "实体与chunks的正确关联比例"
            }
        )
    
    def _evaluate_relation_accuracy(
        self, 
        test_queries: List[Dict], 
        relation_chunks: Dict
    ) -> EvaluationResult:
        """评估关系提取准确性"""
        total_checks = 0
        correct = 0
        
        for query in test_queries:
            expected_relations = query.get("relevant_relations", [])
            expected_entities = set(query.get("relevant_entities", []))
            
            for relation in expected_relations:
                if relation in relation_chunks:
                    rel_data = relation_chunks[relation]
                    src_id = rel_data.get("src_id", "")
                    tgt_id = rel_data.get("tgt_id", "")
                    
                    # 检查关系的实体是否在期望实体集中
                    if src_id in expected_entities or tgt_id in expected_entities:
                        correct += 1
                    total_checks += 1
        
        accuracy = correct / total_checks if total_checks > 0 else 0.0
        
        return EvaluationResult(
            metric_name="RelationAccuracy",
            value=accuracy,
            metadata={
                "correct_relations": correct,
                "total_checks": total_checks,
                "description": "关系涉及正确实体的比例"
            }
        )
    
    def _evaluate_entity_chunk_quality(self, entity_chunks: Dict) -> EvaluationResult:
        """评估实体-Chunk关联质量：平均每个实体关联的chunks数量"""
        if not entity_chunks:
            return EvaluationResult("EntityChunkQuality", 0.0)
        
        chunk_counts = [len(data.get("chunk_ids", [])) for data in entity_chunks.values()]
        avg_chunks_per_entity = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0.0
        
        # 归一化分数（假设3-5个chunks为最优）
        optimal_range = (2, 6)
        if optimal_range[0] <= avg_chunks_per_entity <= optimal_range[1]:
            quality_score = 1.0
        elif avg_chunks_per_entity < optimal_range[0]:
            quality_score = avg_chunks_per_entity / optimal_range[0]
        else:
            quality_score = optimal_range[1] / avg_chunks_per_entity
        
        return EvaluationResult(
            metric_name="EntityChunkQuality",
            value=quality_score,
            metadata={
                "avg_chunks_per_entity": avg_chunks_per_entity,
                "total_entities": len(entity_chunks),
                "description": "实体-Chunk关联质量（平衡性）"
            }
        )
    
    def _check_entities_exist(self, entities: List[str], entity_chunks: Dict) -> List[bool]:
        """检查实体是否存在"""
        return [entity in entity_chunks for entity in entities]
    
    def _check_relations_exist(self, relations: List[str], relation_chunks: Dict) -> List[bool]:
        """检查关系是否存在"""
        return [relation in relation_chunks for relation in relations]
