"""
从 RAG 存储数据自动生成评估数据集
基于实际的知识图谱、实体、关系和chunks数据
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class RAGStorageDatasetGenerator:
    """从 RAG 存储生成评估数据集"""
    
    def __init__(self, storage_dir: str = "./rag_storage1"):
        self.storage_dir = Path(storage_dir)
        
        # 加载数据
        print(f"📂 加载 RAG 存储数据: {storage_dir}")
        self.entity_chunks = self._load_json("kv_store_entity_chunks.json")
        self.relation_chunks = self._load_json("kv_store_relation_chunks.json")
        self.text_chunks = self._load_json("kv_store_text_chunks.json")
        
        print(f"✅ 加载完成:")
        print(f"  - 实体: {len(self.entity_chunks)}")
        print(f"  - 关系: {len(self.relation_chunks)}")
        print(f"  - 文本Chunks: {len(self.text_chunks)}")
    
    def _load_json(self, filename: str) -> Dict:
        """加载 JSON 文件"""
        filepath = self.storage_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载 {filename} 失败: {e}")
            return {}
    
    def generate_dataset(
        self, 
        num_text_queries: int = 10,
        num_entity_queries: int = 5,
        num_relation_queries: int = 5,
        num_multimodal_queries: int = 3,
        output_path: str = "./evaluation_dataset_from_storage.json"
    ) -> str:
        """
        生成完整的评估数据集
        
        Args:
            num_text_queries: 生成的文本查询数量
            num_entity_queries: 实体相关查询数量
            num_relation_queries: 关系相关查询数量
            num_multimodal_queries: 多模态查询数量
            output_path: 输出文件路径
            
        Returns:
            str: 输出文件路径
        """
        print(f"\n🔨 开始生成评估数据集...")
        
        dataset = {
            "metadata": {
                "name": "RAG-Anything Evaluation Dataset (Auto-Generated)",
                "description": "基于 rag_storage1 真实数据自动生成的评估数据集",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "source_storage": str(self.storage_dir),
                "statistics": {
                    "total_entities": len(self.entity_chunks),
                    "total_relations": len(self.relation_chunks),
                    "total_text_chunks": len(self.text_chunks)
                }
            },
            "queries": []
        }
        
        # 1. 生成文本查询
        text_queries = self._generate_text_queries(num_text_queries)
        dataset["queries"].extend(text_queries)
        
        # 2. 生成实体相关查询
        entity_queries = self._generate_entity_queries(num_entity_queries)
        dataset["queries"].extend(entity_queries)
        
        # 3. 生成关系相关查询
        relation_queries = self._generate_relation_queries(num_relation_queries)
        dataset["queries"].extend(relation_queries)
        
        # 4. 生成多模态查询
        multimodal_queries = self._generate_multimodal_queries(num_multimodal_queries)
        dataset["queries"].extend(multimodal_queries)
        
        # 保存数据集
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 数据集生成完成!")
        print(f"  - 总查询数: {len(dataset['queries'])}")
        print(f"  - 文本查询: {len(text_queries)}")
        print(f"  - 实体查询: {len(entity_queries)}")
        print(f"  - 关系查询: {len(relation_queries)}")
        print(f"  - 多模态查询: {len(multimodal_queries)}")
        print(f"  - 输出文件: {output_path}")
        
        return str(output_path)
    
    def _generate_text_queries(self, num: int) -> List[Dict[str, Any]]:
        """生成文本查询"""
        queries = []
        
        # 从文本chunks中采样
        sample_chunks = random.sample(
            list(self.text_chunks.items()), 
            min(num, len(self.text_chunks))
        )
        
        for i, (chunk_id, chunk_data) in enumerate(sample_chunks, 1):
            content = chunk_data.get("content", "")
            # 从内容中提取关键句子作为查询
            sentences = content.split('. ')
            if len(sentences) > 1:
                question = self._generate_question_from_sentence(sentences[0])
                ground_truth = sentences[1] if len(sentences) > 1 else content[:200]
            else:
                question = f"What information is provided in the document about {content[:50]}?"
                ground_truth = content[:200]
            
            queries.append({
                "id": f"text_q{i}",
                "type": "text",
                "question": question,
                "ground_truth": ground_truth,
                "relevant_chunks": [chunk_id],
                "difficulty": "medium",
                "evaluation_focus": ["retrieval", "generation"]
            })
        
        return queries
    
    def _generate_entity_queries(self, num: int) -> List[Dict[str, Any]]:
        """生成实体相关查询"""
        queries = []
        
        # 采样实体
        sample_entities = random.sample(
            list(self.entity_chunks.items()), 
            min(num, len(self.entity_chunks))
        )
        
        for i, (entity_name, entity_data) in enumerate(sample_entities, 1):
            chunk_ids = entity_data.get("chunk_ids", [])
            
            queries.append({
                "id": f"entity_q{i}",
                "type": "entity",
                "question": f"What is {entity_name} and what role does it play in the paper?",
                "ground_truth": f"{entity_name} is mentioned in the context of metasurface design and electromagnetic applications.",
                "relevant_chunks": chunk_ids,
                "relevant_entities": [entity_name],
                "difficulty": "medium",
                "evaluation_focus": ["entity_extraction", "entity_linking"]
            })
        
        return queries
    
    def _generate_relation_queries(self, num: int) -> List[Dict[str, Any]]:
        """生成关系相关查询"""
        queries = []
        
        # 从relation_chunks中采样
        sample_relations = random.sample(
            list(self.relation_chunks.items()), 
            min(num, len(self.relation_chunks))
        )
        
        for i, (rel_key, rel_data) in enumerate(sample_relations, 1):
            src_id = rel_data.get("src_id", "Unknown")
            tgt_id = rel_data.get("tgt_id", "Unknown")
            keywords = rel_data.get("keywords", "related to")
            
            queries.append({
                "id": f"relation_q{i}",
                "type": "relation",
                "question": f"How is {src_id} related to {tgt_id}?",
                "ground_truth": f"{src_id} is {keywords} {tgt_id}.",
                "relevant_chunks": rel_data.get("chunk_ids", []),
                "relevant_entities": [src_id, tgt_id],
                "relevant_relations": [rel_key],
                "difficulty": "hard",
                "evaluation_focus": ["relation_extraction", "knowledge_graph"]
            })
        
        return queries
    
    def _generate_multimodal_queries(self, num: int) -> List[Dict[str, Any]]:
        """生成多模态查询"""
        queries = []
        
        # 基于关键词识别可能的多模态内容
        multimodal_keywords = {
            "image": ["Figure", "Fig.", "图", "image", "diagram"],
            "table": ["Table", "表", "数据表"],
            "equation": ["Equation", "公式", "formula", "Eq."]
        }
        
        # 查找包含多模态关键词的chunks
        multimodal_chunks = []
        for chunk_id, chunk_data in self.text_chunks.items():
            content = chunk_data.get("content", "")
            for mod_type, keywords in multimodal_keywords.items():
                if any(kw in content for kw in keywords):
                    multimodal_chunks.append((chunk_id, chunk_data, mod_type))
                    break
        
        # 采样
        sample_chunks = random.sample(
            multimodal_chunks, 
            min(num, len(multimodal_chunks))
        )
        
        for i, (chunk_id, chunk_data, mod_type) in enumerate(sample_chunks, 1):
            queries.append({
                "id": f"multimodal_q{i}",
                "type": "multimodal",
                "modality_type": mod_type,
                "question": f"Describe the {mod_type} information presented in the paper.",
                "ground_truth": f"The paper contains {mod_type} that illustrates key concepts about metasurface design.",
                "relevant_chunks": [chunk_id],
                "relevant_multimodal": [{"type": mod_type, "description": "content from paper"}],
                "difficulty": "hard",
                "evaluation_focus": ["multimodal_processing", "multimodal_retrieval"]
            })
        
        return queries
    
    def _generate_question_from_sentence(self, sentence: str) -> str:
        """从句子生成问题"""
        # 简单的问题生成逻辑
        if "method" in sentence.lower() or "approach" in sentence.lower():
            return f"What method or approach is described regarding {sentence[:50]}?"
        elif "design" in sentence.lower():
            return f"How is the design process described in terms of {sentence[:50]}?"
        else:
            return f"What information is provided about {sentence[:50]}?"


def main():
    """主函数：生成数据集"""
    import argparse
    
    parser = argparse.ArgumentParser(description="从 RAG 存储生成评估数据集")
    parser.add_argument("--storage_dir", default="./rag_storage1", help="RAG 存储目录")
    parser.add_argument("--output", default="./evaluation_dataset_from_storage.json", help="输出文件路径")
    parser.add_argument("--num_text", type=int, default=10, help="文本查询数量")
    parser.add_argument("--num_entity", type=int, default=5, help="实体查询数量")
    parser.add_argument("--num_relation", type=int, default=5, help="关系查询数量")
    parser.add_argument("--num_multimodal", type=int, default=3, help="多模态查询数量")
    
    args = parser.parse_args()
    
    generator = RAGStorageDatasetGenerator(storage_dir=args.storage_dir)
    output_path = generator.generate_dataset(
        num_text_queries=args.num_text,
        num_entity_queries=args.num_entity,
        num_relation_queries=args.num_relation,
        num_multimodal_queries=args.num_multimodal,
        output_path=args.output
    )
    
    print(f"\n🎉 完成！数据集已保存到: {output_path}")


if __name__ == "__main__":
    main()
