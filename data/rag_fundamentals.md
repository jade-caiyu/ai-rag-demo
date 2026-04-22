# RAG Fundamentals

## 什么是RAG？

RAG (Retrieval-Augmented Generation) 即检索增强生成，是一种将外部知识库与大语言模型结合的技术。

核心流程：
1. 文档加载与预处理
2. 文本分块（Chunking）
3. 向量化（Embedding）
4. 存储到向量数据库
5. 用户查询时，检索相关文档
6. 将检索内容与问题一起发送给LLM
7. 生成回答

## 为什么要用RAG？

1. **解决知识过时问题**：LLM的训练数据有截止日期，RAG可以接入最新知识
2. **减少幻觉**：回答基于真实文档，而非模型编造
3. **降低成本**：无需finetune就能注入新知识
4. **可解释性**：可以追溯答案来源

## RAG的核心组件

### 1. 文档加载器 (Document Loader)

LangChain支持多种格式：
- PDF: PyPDFLoader, PDFPlumberLoader
- Web: WebBaseLoader
- Notion: NotionDBLoader
- Twitter: TwitterTweetLoader

### 2. 文本分块 (Text Splitting)

常用策略：
- RecursiveCharacterTextSplitter（推荐）
- CharacterTextSplitter
- MarkdownHeaderTextSplitter

分块大小通常在256-1024 tokens之间。

### 3. 向量存储 (Vector Store)

常用选择：
- Chroma: 本地开发首选，简单
- Pinecone: 云服务，适合生产
- Weaviate: 开源分布式
- FAISS: Facebook开源，高效

### 4. 检索器 (Retriever)

检索策略：
- Similarity Search（相似度搜索）
- MMR (Maximum Marginal Relevance)：多样性检索
- Compression：压缩检索结果

## RAG的评估指标

### 检索质量
- Precision@K：前K个结果中相关文档的比例
- Recall@K：相关文档被召回的比例

### 生成质量
- 答案准确性
- 答案完整性
- 来源引用准确性

### 系统质量
- 延迟
- 成本
- 可扩展性

## RAG的常见问题与解决方案

### 1. 召回率低

解决方案：
- 调整分块大小
- 使用混合检索（向量+关键词）
- 添加重排（Reranking）

### 2. 答案不完整

解决方案：
- 增加上下文窗口
- 优化prompt
- 使用多轮对话

### 3. 幻觉

解决方案：
- 要求模型引用来源
- 设置置信度阈值
- 添加"不知道"的回复策略