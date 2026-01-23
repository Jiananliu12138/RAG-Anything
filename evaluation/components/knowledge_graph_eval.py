"""
知识图谱评估器
评估知识图谱的结构质量、连通性、密度等特性
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple
from pathlib import Path
from evaluation.base import BaseEvaluator, ComponentEvalResult, EvaluationResult


class KnowledgeGraphEvaluator(BaseEvaluator):
    """知识图谱质量评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(name="KnowledgeGraph", config=config)
        self.storage_dir = config.get("working_dir", "./rag_storage") if config else "./rag_storage"
    
    async def evaluate(self, rag_instance=None, **kwargs) -> ComponentEvalResult:
        """
        评估知识图谱质量
        
        Returns:
            ComponentEvalResult: 评估结果
        """
        print(f"\n🕸️  知识图谱评估:")
        
        # 加载图数据
        graph_data = self._load_graph()
        entity_chunks = self._load_entity_chunks()
        relation_chunks = self._load_relation_chunks()
        
        if not graph_data:
            print("⚠️  无法加载知识图谱数据")
            return ComponentEvalResult(
                component_name=self.name,
                metrics=[],
                summary={},
                details=None
            )
        
        nodes, edges = graph_data
        print(f"  - 节点数: {len(nodes)}")
        print(f"  - 边数: {len(edges)}")
        
        all_results = []
        
        # 1. 图规模指标
        scale_metric = self._evaluate_graph_scale(nodes, edges)
        all_results.append(scale_metric)
        
        # 2. 图密度
        density_metric = self._evaluate_graph_density(nodes, edges)
        all_results.append(density_metric)
        
        # 3. 平均度数
        degree_metric = self._evaluate_average_degree(nodes, edges)
        all_results.append(degree_metric)
        
        # 4. 连通性
        connectivity_metric = self._evaluate_connectivity(nodes, edges)
        all_results.append(connectivity_metric)
        
        # 5. 实体分布质量
        entity_distribution = self._evaluate_entity_distribution(entity_chunks)
        all_results.append(entity_distribution)
        
        # 6. 关系多样性
        relation_diversity = self._evaluate_relation_diversity(relation_chunks)
        all_results.append(relation_diversity)
        
        # 7. 孤立节点比例
        isolated_nodes_metric = self._evaluate_isolated_nodes(nodes, edges)
        all_results.append(isolated_nodes_metric)
        
        # 聚合结果
        summary = self._aggregate_results(all_results)
        
        # 额外统计
        summary["total_nodes"] = len(nodes)
        summary["total_edges"] = len(edges)
        summary["total_entities"] = len(entity_chunks)
        summary["total_relations"] = len(relation_chunks)
        
        # 详细分析
        details = {
            "graph_structure": {
                "nodes": len(nodes),
                "edges": len(edges),
                "density": density_metric.value,
                "avg_degree": degree_metric.value
            },
            "entity_analysis": {
                "total_entities": len(entity_chunks),
                "entity_distribution": entity_distribution.metadata
            },
            "relation_analysis": {
                "total_relations": len(relation_chunks),
                "relation_diversity": relation_diversity.metadata
            },
            "quality_issues": {
                "isolated_nodes_ratio": isolated_nodes_metric.value
            }
        }
        
        return ComponentEvalResult(
            component_name=self.name,
            metrics=all_results,
            summary=summary,
            details=[details] if self.config.get("save_detailed_results", True) else None
        )
    
    def _load_graph(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """加载图结构（从 GraphML 文件）"""
        filepath = Path(self.storage_dir) / "graph_chunk_entity_relation.graphml"
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # GraphML 命名空间
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            
            # 提取节点
            nodes = []
            for node in root.findall('.//graphml:node', ns):
                node_id = node.get('id')
                if node_id:
                    nodes.append(node_id)
            
            # 提取边
            edges = []
            for edge in root.findall('.//graphml:edge', ns):
                source = edge.get('source')
                target = edge.get('target')
                if source and target:
                    edges.append((source, target))
            
            return nodes, edges
        except Exception as e:
            print(f"⚠️  加载图文件失败: {e}")
            return [], []
    
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
    
    def _evaluate_graph_scale(self, nodes: List[str], edges: List[Tuple]) -> EvaluationResult:
        """评估图规模（节点数和边数的比例）"""
        if not nodes:
            return EvaluationResult("GraphScale", 0.0)
        
        edge_node_ratio = len(edges) / len(nodes) if nodes else 0.0
        
        # 理想的边节点比（1-3之间较好）
        optimal_ratio = 2.0
        if 1.0 <= edge_node_ratio <= 3.0:
            scale_score = 1.0
        elif edge_node_ratio < 1.0:
            scale_score = edge_node_ratio / 1.0
        else:
            scale_score = optimal_ratio / edge_node_ratio
        
        return EvaluationResult(
            metric_name="GraphScale",
            value=scale_score,
            metadata={
                "nodes": len(nodes),
                "edges": len(edges),
                "edge_node_ratio": edge_node_ratio
            }
        )
    
    def _evaluate_graph_density(self, nodes: List[str], edges: List[Tuple]) -> EvaluationResult:
        """评估图密度 = 2*E / (N*(N-1))"""
        n = len(nodes)
        e = len(edges)
        
        if n <= 1:
            return EvaluationResult("GraphDensity", 0.0)
        
        max_edges = n * (n - 1)  # 有向图
        density = (2 * e) / max_edges if max_edges > 0 else 0.0
        
        return EvaluationResult(
            metric_name="GraphDensity",
            value=density,
            metadata={
                "actual_edges": e,
                "max_possible_edges": max_edges // 2,  # 无向图
                "density_percentage": density * 100
            }
        )
    
    def _evaluate_average_degree(self, nodes: List[str], edges: List[Tuple]) -> EvaluationResult:
        """评估平均度数"""
        if not nodes:
            return EvaluationResult("AverageDegree", 0.0)
        
        # 计算每个节点的度数
        degree_count = {node: 0 for node in nodes}
        for src, tgt in edges:
            if src in degree_count:
                degree_count[src] += 1
            if tgt in degree_count:
                degree_count[tgt] += 1
        
        avg_degree = sum(degree_count.values()) / len(degree_count) if degree_count else 0.0
        
        # 归一化分数（平均度3-10为较好）
        if 3 <= avg_degree <= 10:
            degree_score = 1.0
        elif avg_degree < 3:
            degree_score = avg_degree / 3.0
        else:
            degree_score = 10.0 / avg_degree
        
        return EvaluationResult(
            metric_name="AverageDegree",
            value=degree_score,
            metadata={
                "avg_degree": avg_degree,
                "max_degree": max(degree_count.values()) if degree_count else 0,
                "min_degree": min(degree_count.values()) if degree_count else 0
            }
        )
    
    def _evaluate_connectivity(self, nodes: List[str], edges: List[Tuple]) -> EvaluationResult:
        """评估连通性（使用 BFS 计算最大连通分量）"""
        if not nodes:
            return EvaluationResult("GraphConnectivity", 0.0)
        
        # 构建邻接表
        adj_list = {node: [] for node in nodes}
        for src, tgt in edges:
            if src in adj_list and tgt in adj_list:
                adj_list[src].append(tgt)
                adj_list[tgt].append(src)  # 无向图
        
        # BFS 查找连通分量
        visited = set()
        max_component_size = 0
        
        def bfs(start):
            queue = [start]
            visited.add(start)
            size = 1
            
            while queue:
                node = queue.pop(0)
                for neighbor in adj_list.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        size += 1
            return size
        
        for node in nodes:
            if node not in visited:
                component_size = bfs(node)
                max_component_size = max(max_component_size, component_size)
        
        connectivity = max_component_size / len(nodes) if nodes else 0.0
        
        return EvaluationResult(
            metric_name="GraphConnectivity",
            value=connectivity,
            metadata={
                "largest_component_size": max_component_size,
                "total_nodes": len(nodes),
                "connectivity_ratio": connectivity
            }
        )
    
    def _evaluate_entity_distribution(self, entity_chunks: Dict) -> EvaluationResult:
        """评估实体分布质量（实体关联的chunks数量分布）"""
        if not entity_chunks:
            return EvaluationResult("EntityDistribution", 0.0)
        
        chunk_counts = [len(data.get("chunk_ids", [])) for data in entity_chunks.values()]
        
        avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0.0
        max_chunks = max(chunk_counts) if chunk_counts else 0
        min_chunks = min(chunk_counts) if chunk_counts else 0
        
        # 计算标准差（评估分布均匀性）
        if len(chunk_counts) > 1:
            variance = sum((x - avg_chunks) ** 2 for x in chunk_counts) / len(chunk_counts)
            std_dev = variance ** 0.5
            # 归一化：标准差越小，分布越均匀
            distribution_score = 1.0 / (1.0 + std_dev / avg_chunks) if avg_chunks > 0 else 0.0
        else:
            distribution_score = 1.0
        
        return EvaluationResult(
            metric_name="EntityDistribution",
            value=distribution_score,
            metadata={
                "avg_chunks_per_entity": avg_chunks,
                "max_chunks": max_chunks,
                "min_chunks": min_chunks,
                "std_dev": std_dev if len(chunk_counts) > 1 else 0.0
            }
        )
    
    def _evaluate_relation_diversity(self, relation_chunks: Dict) -> EvaluationResult:
        """评估关系多样性（不同类型关系的数量）"""
        if not relation_chunks:
            return EvaluationResult("RelationDiversity", 0.0)
        
        # 提取关系类型（基于 keywords）
        relation_types = set()
        for rel_data in relation_chunks.values():
            keywords = rel_data.get("keywords", "unknown")
            relation_types.add(keywords)
        
        # 多样性分数：unique types / total relations
        diversity_score = len(relation_types) / len(relation_chunks) if relation_chunks else 0.0
        
        return EvaluationResult(
            metric_name="RelationDiversity",
            value=diversity_score,
            metadata={
                "unique_relation_types": len(relation_types),
                "total_relations": len(relation_chunks),
                "diversity_ratio": diversity_score
            }
        )
    
    def _evaluate_isolated_nodes(self, nodes: List[str], edges: List[Tuple]) -> EvaluationResult:
        """评估孤立节点比例（没有连接的节点）"""
        if not nodes:
            return EvaluationResult("IsolatedNodesRatio", 0.0)
        
        # 找出所有有连接的节点
        connected_nodes = set()
        for src, tgt in edges:
            connected_nodes.add(src)
            connected_nodes.add(tgt)
        
        isolated_count = len([n for n in nodes if n not in connected_nodes])
        isolated_ratio = isolated_count / len(nodes) if nodes else 0.0
        
        # 孤立节点越少越好
        quality_score = 1.0 - isolated_ratio
        
        return EvaluationResult(
            metric_name="IsolatedNodesRatio",
            value=quality_score,
            metadata={
                "isolated_nodes": isolated_count,
                "total_nodes": len(nodes),
                "isolated_ratio": isolated_ratio
            }
        )
