#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process documents with RAGAnything using MinerU parser
2. Perform pure text queries using aquery() method
3. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
4. Handle different types of multimodal content (tables, equations) in queries
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from functools import partial
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # Parser selection: mineru or docling
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function - using Ollama
        # 使用纯文本模型进行实体提取，视觉模型输出格式不符合 LightRAG 预期
        llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b")
        llm_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
        llm_timeout = int(os.getenv("TIMEOUT", "600"))
        
        logger.info(f"🤖 LLM 模型配置: {llm_model}")
        logger.info(f"🌐 LLM Host: {llm_host}")

        # Define vision model function for Ollama multimodal models
        # 视觉模型用于图像描述，使用 qwen3-vl 或回退到 LLM 模型
        vision_model = os.getenv("VISION_MODEL", "qwen3-vl:8b")
        vision_host = os.getenv("VISION_BINDING_HOST", llm_host)
        vision_timeout = int(os.getenv("VISION_TIMEOUT", str(llm_timeout)))
        
        logger.info(f"👁️  Vision 模型配置: {vision_model}")
        logger.info(f"🌐 Vision Host: {vision_host}")
        
        async def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                # Ollama expects messages parameter in kwargs
                return await ollama_model_complete(
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    hashing_kv=kwargs.get("hashing_kv"),
                    host=vision_host,
                    timeout=1200,  # 增加超时到 20 分钟
                    options={"num_ctx": 4096},
                    **{k: v for k, v in kwargs.items() if k != "hashing_kv"},
                )
            # Traditional single image format
            elif image_data:
                # Build Ollama-compatible messages with image
                vision_messages = []
                if system_prompt:
                    vision_messages.append({"role": "system", "content": system_prompt})
                vision_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                })
                return await ollama_model_complete(
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=vision_messages,
                    hashing_kv=kwargs.get("hashing_kv"),
                    host=vision_host,
                    timeout=1200,  # 增加超时到 20 分钟
                    options={"num_ctx": 4096},
                    **{k: v for k, v in kwargs.items() if k != "hashing_kv"},
                )
            # Pure text format
            else:
                return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Configure embedding function using Ollama
        from dotenv import dotenv_values
        
        for key in ["EMBEDDING_DIM", "EMBEDDING_MODEL", "LLM_MODEL"]:
            if key in os.environ:
                del os.environ[key]
                
        env_values = dotenv_values(".env")
        
        embedding_dim = int(env_values.get("EMBEDDING_DIM", 768))
        embedding_model = env_values.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        embedding_host = env_values.get("EMBEDDING_BINDING_HOST", llm_host)
        # 降低 max_token_size 避免超过 nomic-embed-text 的上下文限制（2048）
        embedding_max_tokens = int(env_values.get("MAX_EMBED_TOKENS", "512"))
        
        logger.info(f"📊 [DEBUG] Ollama Embedding配置:")
        logger.info(f"   - 模型: {embedding_model}")
        logger.info(f"   - 维度: {embedding_dim}")
        logger.info(f"   - Host: {embedding_host}")
        logger.info(f"   - 最大Tokens: {embedding_max_tokens}")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=embedding_max_tokens,
            func=partial(
                ollama_embed.func,
                embed_model=embedding_model,
                host=embedding_host,
                timeout=1200,  # 增加 embedding 超时到 20 分钟
            ),
        )
        
        logger.info(f"✅ EmbeddingFunc 创建完成，维度: {embedding_func.embedding_dim}")

        # Initialize RAGAnything with Ollama-optimized configuration
        rag = RAGAnything(
            config=config,
            llm_model_func=ollama_model_complete,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs={
                "llm_model_name": llm_model,
                "summary_max_tokens": 2048,
                "chunk_token_size": 200,
                "chunk_overlap_token_size": 30,
                "llm_model_kwargs": {
                    "host": llm_host,
                    "options": {"num_ctx": 4096},
                    "timeout": 1200,  # 增加超时到 20 分钟
                },
                "llm_model_max_async": 1,  # 单并发处理
                "default_llm_timeout": 1200,  # LightRAG 内部超时也设置为 20 分钟
            }
        )

        # Process document
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")

        # 1. 纯文本查询 - 针对元表面微波吸收器论文
        text_queries = [
            "What is the Neuro-TF approach proposed in this paper?",
            "What are the advantages of using neural networks for metasurface design?",
            "How does the proposed method compare with traditional design methods in terms of speed?",
        ]

        for query in text_queries:
            logger.info(f"\n[文本查询]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"回答: {result}")

        # 2. 多模态查询 - 分析吸收率性能数据表格
        logger.info(
            "\n[多模态查询]: 分析超材料吸收器的性能参数表格"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this absorption performance data with the results reported in the paper. Are these values reasonable for metasurface-based microwave absorbers?",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Frequency(GHz),Absorption_Rate(%),Thickness(mm),Design_Method
                                10.5,98.5,3.2,Neuro-TF
                                10.5,95.2,3.2,Traditional_Optimization
                                10.5,92.1,3.2,Empirical_Design""",
                    "table_caption": "Metasurface absorber performance comparison at 10.5 GHz",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"回答: {multimodal_result}")

        # 3. 多模态查询 - 电磁场公式分析
        logger.info("\n[多模态查询]: 分析电磁波反射系数公式")
        equation_result = await rag.aquery_with_multimodal(
            "Explain this reflection coefficient formula and how it relates to the absorption rate calculation mentioned in the paper",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "\\Gamma = \\frac{Z_s - Z_0}{Z_s + Z_0}",
                    "equation_caption": "Reflection coefficient formula where Z_s is surface impedance and Z_0 is free space impedance",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"回答: {equation_result}")

        # 4. 多模态查询 - 图像分析（针对论文中的图表）
        logger.info("\n[多模态查询]: 查询论文中的图像信息")
        image_query_result = await rag.aquery(
            "Describe the metasurface unit cell structure shown in the figures. What are the key geometric parameters?",
            mode="hybrid",
        )
        logger.info(f"回答: {image_query_result}")

        # 5. 另一个图像查询 - 吸收率曲线
        logger.info("\n[多模态查询]: 查询吸收率频谱曲线")
        absorption_curve_result = await rag.aquery(
            "What is the absorption spectrum shown in the paper? At which frequency does the absorber achieve peak performance?",
            mode="hybrid",
        )
        logger.info(f"回答: {absorption_curve_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
