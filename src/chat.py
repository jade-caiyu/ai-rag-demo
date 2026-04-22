#!/usr/bin/env python3
"""
问答脚本
基于RAG的知识库问答系统
"""

import sys
import os
from dotenv import load_dotenv

# 先加载.env文件
load_dotenv()
load_dotenv("/Users/jade/job/ai-rag-demo/.env")

import config

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

def load_vectorstore():
    """加载向量数据库"""
    if not config.CHROMA_DIR.exists():
        print("❌ Vector store not found!")
        print("Please run: python src/ingest.py")
        sys.exit(1)
    
    print("📚 Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embeddings
    )
    
    return vectorstore

def create_qa_chain(vectorstore):
    """创建问答chain"""
    print("🤖 Initializing LLM...")
    
    # 检查API Key
    if not config.DASHSCOPE_API_KEY:
        print("❌ Please set DASHSCOPE_API_KEY")
        print("   export DASHSCOPE_API_KEY='your-key'")
        print("\n   Get key from: https://dashscope.console.aliyun.com/")
        sys.exit(1)
    
    # 阿里云DashScope配置
    os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY
    
    # 使用OpenAI兼容接口（DashScope支持）
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.7,
        openai_api_key=config.DASHSCOPE_API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 创建QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVAL_TOP_K}
        ),
        return_source_documents=True
    )
    
    return qa

def chat(qa):
    """问答交互"""
    print("\n" + "=" * 50)
    print("💬 AI Engineering RAG Assistant")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'q', 'exit']:
            print("👋 Bye!")
            break
        
        try:
            result = qa.invoke(query)
            print(f"\nBot: {result['result']}\n")
            
            # 显示来源
            if result.get('source_documents'):
                print("📖 Sources:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"   {i}. {doc.metadata.get('source', 'Unknown')}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def main():
    # 加载向量库
    vectorstore = load_vectorstore()
    
    # 创建QA chain
    qa = create_qa_chain(vectorstore)
    
    # 开始聊天
    chat(qa)

if __name__ == "__main__":
    main()