# AI Engineering RAG Assistant
# Copyright 2025

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()
load_dotenv("/Users/jade/job/ai-rag-demo/.env")

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# 模型配置
# 使用免费的本地embedding，不需要API key
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 阿里云DashScope配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_MODEL = "qwen-turbo"  # 免费版选 qwen-turbo，有额度用 qwen-plus

# RAG配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 3

# 知识库文档列表
DOCUMENT_FILES = [
    "rag_fundamentals.md",
    "agent_basics.md",
    "prompt_engineering.md",
]