#!/usr/bin/env python3
"""
问答脚本
基于RAG的知识库问答系统
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv("/Users/jade/job/ai-rag-demo/.env")

import config

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


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


def hybrid_search(query, vectorstore, top_k=5):
    """
    简化版混合检索
    1. 向量检索
    2. 关键词匹配
    3. RRF融合
    """
    # 向量检索
    docs_vector = vectorstore.similarity_search(query, k=top_k)
    
    # 简单关键词匹配
    try:
        query_terms = query.lower().split()
        all_docs = vectorstore.get()['documents']
        metadatas = vectorstore.get()['metadatas']
        
        docs_keyword = []
        for i, doc_text in enumerate(all_docs):
            score = sum(1 for term in query_terms if term in doc_text.lower())
            if score > 0:
                docs_keyword.append((i, score, doc_text))
        
        docs_keyword.sort(key=lambda x: x[1], reverse=True)
        
        # 取top结果
        keyword_results = []
        for i, _, text in docs_keyword[:top_k]:
            keyword_results.append(Document(page_content=text, metadata=metadatas[i] if i < len(metadatas) else {}))
        
        # RRF融合
        return rrf_merge(docs_vector, keyword_results, top_k)
    except:
        return docs_vector[:top_k]


def rrf_merge(docs_a, docs_b, k=3):
    """RRF融合"""
    scores = {}
    
    for i, doc in enumerate(docs_a):
        key = doc.page_content[:30]
        scores[key] = scores.get(key, 0) + 1 / (i + k)
    
    for i, doc in enumerate(docs_b):
        key = doc.page_content[:30]
        scores[key] = scores.get(key, 0) + 1 / (i + k)
    
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    all_docs = docs_a + docs_b
    result = []
    for key in sorted_keys:
        for doc in all_docs:
            if doc.page_content[:30] == key and doc not in result:
                result.append(doc)
                break
    
    return result[:k]


def chat():
    """问答"""
    vectorstore = load_vectorstore()
    
    print("🤖 Initializing LLM...")
    os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY
    
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.7,
        openai_api_key=config.DASHSCOPE_API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    prompt = PromptTemplate.from_template(
        "根据以下参考文档回答问题。\n\n参考文档：\n{context}\n\n问题：{question}\n\n回答："
    )
    
    chain = prompt | llm
    
    print("\n" + "=" * 50)
    print("💬 AI Engineering RAG")
    print("   Hybrid Search: Vector + Keyword (RRF)")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if not query or query.lower() in ['quit', 'q', 'exit']:
            print("👋 Bye!")
            break
        
        try:
            # 混合检索
            docs = hybrid_search(query, vectorstore, config.RETRIEVAL_TOP_K)
            context = "\n\n".join([d.page_content for d in docs])
            
            # 生成回答
            result = chain.invoke({"question": query, "context": context})
            # 只取内容
            answer = result.content if hasattr(result, 'content') else str(result)
            print(f"\nBot: {answer}\n")
            
            # 来源
            print("📖 Sources:")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get('source', 'Unknown')
                print(f"   {i}. {src}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    chat()