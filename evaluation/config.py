"""
评估系统配置
支持通过环境变量或代码灵活配置评估参数
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class EvaluationConfig:
    """评估系统配置类"""
    
    # 基础路径配置
    working_dir: str = "./rag_storage"
    """RAG 存储目录路径"""
    
    output_dir: str = "./evaluation_results"
    """评估结果输出目录"""
    
    dataset_path: Optional[str] = None
    """测试数据集路径（JSON 格式）"""
    
    # 组件级评估配置
    enable_retriever_eval: bool = True
    """是否启用检索器评估"""
    
    enable_generator_eval: bool = True
    """是否启用生成器评估"""
    
    enable_parser_eval: bool = False
    """是否启用解析器评估（需要 ground truth）"""
    
    enable_entity_relation_eval: bool = True
    """是否启用实体关系评估"""
    
    enable_knowledge_graph_eval: bool = True
    """是否启用知识图谱评估"""
    
    enable_chunk_embedding_eval: bool = True
    """是否启用chunks和embeddings评估"""
    
    # 端到端评估配置
    enable_qa_eval: bool = True
    """是否启用问答评估"""
    
    enable_multimodal_eval: bool = True
    """是否启用多模态评估"""
    
    enable_ragas_eval: bool = False
    """是否启用RAGAS评估（需要安装ragas）"""
    
    # 检索指标配置
    retrieval_top_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    """检索指标的 K 值列表"""
    
    # 生成指标配置
    use_bertscore: bool = True
    """是否使用 BERTScore（需要额外依赖）"""
    
    use_rouge: bool = True
    """是否使用 ROUGE 指标"""
    
    use_bleu: bool = False
    """是否使用 BLEU 指标"""
    
    # 多模态评估配置
    multimodal_types: List[str] = field(default_factory=lambda: ["image", "table", "equation"])
    """需要评估的多模态类型"""
    
    # LLM 评估配置（使用 LLM 作为评判器）
    use_llm_judge: bool = True
    """是否使用 LLM 作为评判器"""
    
    llm_judge_model: str = "qwen2.5:7b-instruct"
    """LLM 评判器使用的模型"""
    
    llm_judge_aspects: List[str] = field(default_factory=lambda: [
        "faithfulness",    # 忠实度（是否基于检索内容）
        "relevance",       # 相关性（是否回答了问题）
        "coherence",       # 连贯性（逻辑是否清晰）
        "completeness",    # 完整性（是否全面）
    ])
    """LLM 评判的评价维度"""
    
    # 并发控制
    max_concurrent_evals: int = 4
    """最大并发评估数"""
    
    # RAGAS评估配置
    ragas_llm_model: str = "qwen2.5:7b-instruct"
    """RAGAS评估使用的LLM模型"""
    
    ragas_embedding_model: str = "nomic-embed-text"
    """RAGAS评估使用的embedding模型"""
    
    ragas_ollama_host: str = "http://localhost:11434"
    """RAGAS评估使用的Ollama服务地址"""
    
    # 其他配置
    save_detailed_results: bool = True
    """是否保存详细的评估结果"""
    
    random_seed: int = 42
    """随机种子，用于结果可复现"""
    
    def __post_init__(self):
        """后处理：创建输出目录"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
