"""
测试数据集加载器
支持从 JSON 文件加载测试数据集
"""

import json
from typing import List, Dict, Any
from pathlib import Path


class EvaluationDataset:
    """评估数据集类"""
    
    def __init__(self, data: Dict[str, Any]):
        """
        初始化数据集
        
        Args:
            data: 数据集字典，应包含:
                - metadata: 数据集元信息
                - queries: 查询列表
        """
        self.metadata = data.get("metadata", {})
        self.queries = data.get("queries", [])
    
    def filter_by_type(self, query_type: str) -> List[Dict[str, Any]]:
        """
        按查询类型过滤
        
        Args:
            query_type: 查询类型 (text, multimodal, etc.)
            
        Returns:
            List[Dict[str, Any]]: 过滤后的查询列表
        """
        return [q for q in self.queries if q.get("type") == query_type]
    
    def filter_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """
        按模态类型过滤
        
        Args:
            modality: 模态类型 (image, table, equation)
            
        Returns:
            List[Dict[str, Any]]: 过滤后的查询列表
        """
        return [q for q in self.queries if q.get("modality_type") == modality]
    
    def get_all_queries(self) -> List[Dict[str, Any]]:
        """获取所有查询"""
        return self.queries
    
    def __len__(self):
        return len(self.queries)


class DatasetLoader:
    """数据集加载器"""
    
    @staticmethod
    def load_from_json(file_path: str) -> EvaluationDataset:
        """
        从 JSON 文件加载数据集
        
        Args:
            file_path: JSON 文件路径
            
        Returns:
            EvaluationDataset: 数据集对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return EvaluationDataset(data)
    
    @staticmethod
    def create_sample_dataset(output_path: str = "./evaluation_dataset_sample.json"):
        """
        创建示例数据集
        
        Args:
            output_path: 输出文件路径
        """
        sample_data = {
            "metadata": {
                "name": "Sample RAG Evaluation Dataset",
                "description": "示例评估数据集",
                "version": "1.0",
                "created_at": "2026-01-23"
            },
            "queries": [
                {
                    "id": "q1",
                    "type": "text",
                    "question": "What is the main contribution of this paper?",
                    "ground_truth": "The paper proposes a Neuro-TF approach for fast design of metasurface-based microwave absorbers.",
                    "relevant_chunks": ["chunk-abc123", "chunk-def456"],
                    "difficulty": "easy"
                },
                {
                    "id": "q2",
                    "type": "multimodal",
                    "modality_type": "image",
                    "question": "Describe the structure shown in Figure 1.",
                    "ground_truth": "Figure 1 shows a three-layer metasurface absorber structure with metallic patterns.",
                    "relevant_chunks": ["chunk-img001"],
                    "relevant_multimodal": [
                        {"id": "Figure 1", "type": "image"}
                    ],
                    "difficulty": "medium"
                },
                {
                    "id": "q3",
                    "type": "multimodal",
                    "modality_type": "table",
                    "question": "What are the absorption rates reported in the results table?",
                    "ground_truth": "The results table shows absorption rates ranging from 92% to 98% across different frequencies.",
                    "relevant_chunks": ["chunk-table01"],
                    "relevant_multimodal": [
                        {"id": "Table 1", "type": "table"}
                    ],
                    "difficulty": "medium"
                },
                {
                    "id": "q4",
                    "type": "multimodal",
                    "modality_type": "equation",
                    "question": "Explain the reflection coefficient equation mentioned in the paper.",
                    "ground_truth": "The reflection coefficient is calculated using the formula Γ = (Zs - Z0) / (Zs + Z0).",
                    "relevant_chunks": ["chunk-eq001"],
                    "relevant_multimodal": [
                        {"id": "Equation 1", "type": "equation"}
                    ],
                    "difficulty": "hard"
                }
            ]
        }
        
        # 保存到文件
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 示例数据集已创建: {output_path}")
        return output_path
    
    @staticmethod
    def validate_dataset(data: Dict[str, Any]) -> bool:
        """
        验证数据集格式是否正确
        
        Args:
            data: 数据集字典
            
        Returns:
            bool: 是否有效
        """
        required_fields = ["queries"]
        
        for field in required_fields:
            if field not in data:
                print(f"❌ 缺少必需字段: {field}")
                return False
        
        # 验证每个查询的格式
        for i, query in enumerate(data["queries"]):
            if "question" not in query:
                print(f"❌ 查询 {i} 缺少 'question' 字段")
                return False
        
        return True
