# AI Engineering RAG Assistant

基于LangChain构建的AI应用开发知识问答系统。

## 项目简介

本项目实现了一个基于RAG（Retrieval-Augmented Generation）技术的知识问答系统，将AI开发相关的文档知识向量化存储，支持通过自然语言进行智能问答。

## 技术栈

| 组件 | 技术选择 |
|-----|---------|
| 框架 | LangChain |
| 向量数据库 | Chroma |
| LLM | 阿里云 Qwen (qwen-turbo) |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) |

## 项目架构

```
.
├── src/
│   ├── config.py       # 配置文件
│   ├── ingest.py     # 知识库构建脚本
│   └── chat.py      # 问答交互脚本
├── data/             # 知识库文档（Markdown格式）
├── chroma_db/       # 向量数据库（本地存储）
└── README.md
```

## 核心流程

```
用户输入 → 向量化 → 向量检索 → 内容注入Prompt → LLM生成 → 返回回答
```

### 1. 知识库构建
- 加载Markdown文档
- RecursiveCharacterTextSplitter文本分块
- sentence-transformers向量化
- Chroma向量库存储

### 2. 问答流程
- 用户问题向量化
- 相似度检索（Top-K）
- 召回内容 + 问题注入Prompt
- Qwen模型生成回答

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

在 `.env` 文件中配置API Key：

```
DASHSCOPE_API_KEY=your-api-key
```

获取API Key: [DashScope控制台](https://dashscope.console.aliyun.com/)

### 运行

```bash
# 构建知识库（首次运行）
python src/ingest.py

# 启动问答
python src/chat.py
```

## 可扩展方向

- [ ] 混合检索（BM25 + 向量）
- [ ] Cross-Encoder重排
- [ ] 评估数据集
- [ ] Streamlit Web界面
- [ ] Agent能力扩展（工具调用）

## 参考资源

- [LangChain Documentation](https://python.langchain.com/)
- [Qwen API文档](https://help.aliyun.com/document_detail/271957.html)

## 许可证

MIT