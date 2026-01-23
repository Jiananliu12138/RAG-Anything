#!/usr/bin/env python
"""
RAG-Anything 查询脚本 (Ollama 版)
只进行查询，不重新构建知识图谱。
完全复用 raganything_example.py 中的 Ollama 配置逻辑。
"""

import os
import asyncio
import logging
import logging.config
from pathlib import Path
import sys
from functools import partial
from dotenv import load_dotenv, dotenv_values

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

# 加载 .env
load_dotenv(dotenv_path=".env", override=True)

def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "query_only.log"))

    print(f"\n查询日志文件: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

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

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")

async def query_only(working_dir: str = "./rag_storage"):
    """
    只进行查询，复用 raganything_example.py 的 Ollama 配置逻辑
    """
    try:
        if not os.path.exists(working_dir):
            logger.error(f"❌ 存储目录不存在: {working_dir}")
            return

        # 1. 获取 Ollama 配置 (复用 raganything_example.py 逻辑)
        llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
        llm_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
        # 确保 host 不包含 /v1，Ollama SDK 不需要
        if llm_host.endswith("/v1"):
            llm_host = llm_host[:-3]
        
        vision_model = os.getenv("VISION_MODEL", "qwen3-vl:8b")
        vision_host = os.getenv("VISION_BINDING_HOST", llm_host)
        if vision_host == "ollama": # 处理 .env 中的特殊值
            vision_host = llm_host
        if vision_host.endswith("/v1"):
            vision_host = vision_host[:-3]
        
        logger.info(f"🤖 LLM 模型: {llm_model} | Host: {llm_host}")
        logger.info(f"👁️  Vision 模型: {vision_model} | Host: {vision_host}")

        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 2. 定义 Vision 函数 (复用 raganything_example.py 逻辑并修复 Ollama 消息格式)
        async def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # 修复 Ollama SDK 的消息格式：将 list 类型的 content 转换为 string + images
            def format_ollama_messages(msgs):
                formatted = []
                for m in msgs:
                    content = m.get("content")
                    role = m.get("role")
                    if isinstance(content, list):
                        text_parts = []
                        images = []
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                # 提取 base64 数据
                                url = item.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    base64_data = url.split(",")[1]
                                    images.append(base64_data)
                        
                        msg = {"role": role, "content": " ".join(text_parts)}
                        if images:
                            msg["images"] = images
                        formatted.append(msg)
                    else:
                        formatted.append(m)
                return formatted

            if messages:
                ollama_msgs = format_ollama_messages(messages)
                return await ollama_model_complete(
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=ollama_msgs,
                    hashing_kv=kwargs.get("hashing_kv"),
                    host=vision_host,
                    timeout=1200,
                    options={"num_ctx": 4096},
                    **{k: v for k, v in kwargs.items() if k != "hashing_kv"},
                )
            elif image_data:
                # 单图片模式
                msg = {"role": "user", "content": prompt, "images": [image_data]}
                ollama_msgs = []
                if system_prompt:
                    ollama_msgs.append({"role": "system", "content": system_prompt})
                ollama_msgs.append(msg)
                
                return await ollama_model_complete(
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=ollama_msgs,
                    hashing_kv=kwargs.get("hashing_kv"),
                    host=vision_host,
                    timeout=1200,
                    options={"num_ctx": 4096},
                    **{k: v for k, v in kwargs.items() if k != "hashing_kv"},
                )
            else:
                return await ollama_model_complete(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    hashing_kv=kwargs.get("hashing_kv"),
                    host=llm_host,
                    timeout=1200,
                    options={"num_ctx": 4096},
                    **{k: v for k, v in kwargs.items() if k != "hashing_kv"},
                )

        # 3. 配置 Embedding (复用 raganything_example.py 逻辑)
        for key in ["EMBEDDING_DIM", "EMBEDDING_MODEL", "LLM_MODEL"]:
            if key in os.environ:
                os.environ.pop(key)
        
        env_values = dotenv_values(".env")
        embedding_dim = int(env_values.get("EMBEDDING_DIM", 768))
        embedding_model = env_values.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        embedding_host = env_values.get("EMBEDDING_BINDING_HOST", llm_host)
        if embedding_host == "ollama":
            embedding_host = llm_host
        if embedding_host.endswith("/v1"):
            embedding_host = embedding_host[:-3]
        embedding_max_tokens = int(env_values.get("MAX_EMBED_TOKENS", "512"))

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=embedding_max_tokens,
            func=partial(
                ollama_embed.func,
                embed_model=embedding_model,
                host=embedding_host,
                timeout=1200,
            ),
        )

        # 4. 初始化 (不要传 embedding_model_name 和 embedding_model_kwargs，LightRAG 不接受)
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
                    "timeout": 1200,
                },
                "llm_model_max_async": 1,
                "default_llm_timeout": 1200,
            }
        )
        
        # 显式初始化 LightRAG 实例并加载存储
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result.get("success"):
            logger.error(f"❌ LightRAG 初始化失败: {init_result.get('error')}")
            return
        
        logger.info("✅ 知识图谱加载成功！")
        logger.info(f"   LightRAG 实例: {rag.lightrag is not None}")
        logger.info(f"   Parse Cache: {rag.parse_cache is not None}")

        # 5. 执行查询 (复用 example 中的问题)
        text_queries = [
            "What is the Neuro-TF approach proposed in this paper?",
            "What are the advantages of using neural networks for metasurface design?",
            "How does the proposed method compare with traditional design methods in terms of speed?",
        ]

        for query in text_queries:
            logger.info(f"\n[文本查询]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"回答: {result}")

        # 表格查询
        logger.info("\n[多模态查询]: 分析吸收率性能数据表格")
        multimodal_result = await rag.aquery_with_multimodal(
            "Based on the tables in the document, what is the maximum absorption achieved by the Neuro-TF approach?",
            mode="hybrid",
        )
        logger.info(f"回答: {multimodal_result}")

        # 图片查询
        logger.info("\n[多模态查询]: 查找并分析图片内容")
        image_result = await rag.aquery_with_multimodal(
            "Find the image showing the structure of the metasurface absorber and describe its layers.",
            mode="hybrid",
        )
        logger.info(f"回答: {image_result}")

    except Exception as e:
        logger.error(f"❌ 查询过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default="./rag_storage")
    args = parser.parse_args()
    
    configure_logging()
    asyncio.run(query_only(args.working_dir))
