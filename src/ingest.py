#!/usr/bin/env python3
"""
知识库构建脚本
将data目录下的文档向量化并存储到Chroma
"""

import sys
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()
load_dotenv("/Users/jade/job/ai-rag-demo/.env")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config

def load_documents():
    """加载知识库文档"""
    print("📚 Loading documents...")
    
    docs = []
    for file in config.DATA_DIR.glob("*.md"):
        print(f"   Loading: {file.name}")
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())
    
    print(f"   Loaded {len(docs)} documents")
    return docs

def split_documents(docs):
    """分块"""
    print("✂️ Splitting documents...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    chunks = splitter.split_documents(docs)
    print(f"   Created {len(chunks)} chunks")
    return chunks

def build_vectorstore(chunks):
    """构建向量数据库"""
    print("💾 Building vector store...")
    
    # 使用免费的本地embedding模型
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # 清理旧的向量库
    if config.CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(config.CHROMA_DIR)
    
    # 创建新的向量库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR)
    )
    
    print(f"   Vector store created at {config.CHROMA_DIR}")
    return vectorstore

def main():
    print("=" * 50)
    print("🚀 AI Engineering RAG - Knowledge Base Builder")
    print("=" * 50)
    
    # 创建目录
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 构建知识库
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    
    print("\n" + "=" * 50)
    print("✅ Knowledge base built successfully!")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Storage: {config.CHROMA_DIR}")
    print("=" * 50)
    
    print("\n🎯 Try asking:")
    print("   - 什么是RAG？")
    print("   - Agent和Chatbot有什么区别？")
    print("   - 如何优化Prompt？")
    print("\nRun: python src/chat.py")

if __name__ == "__main__":
    main()