# Agent基础

## 什么是Agent？

Agent（智能体）是一个能够自主推理、规划并执行行动来完成任务的人工智能系统。

与简单的Chatbot不同，Agent可以：
- 使用工具（Tools）
- 进行多步推理（Reasoning）
- 自主规划（Planning）
- 记忆上下文（Memory）

## Agent的核心组件

### 1. LLM (大脑)

负责：
- 理解用户意图
- 生成回复
- 规划行动步骤

### 2. Tools (工具)

常用工具：
- 搜索：Google Search, Bing Search
- 数据库查询：SQL
- API调用：HTTP请求
- 代码执行：Python, REPL
- 文件操作：读取、写入

### 3. Action Loop (执行循环)

Agent的核心循环：
1. 接收用户输入
2. LLM决定是否需要使用工具
3. 如果需要，选择合适的工具并生成参数
4. 执行工具
5. 获取结果
6. 判断是否继续或结束

### 4. Memory (记忆)

- Short-term Memory：当前对话上下文
- Long-term Memory：长期知识存储

## Agent的实现框架

### LangChain Agent

```python
from langchain.agents import load_tools, initialize_agent, AgentType

# 加载工具
tools = load_tools(["serpapi", "llm-math"], llm=chatgpt)

# 初始化Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# 执行
result = agent.run("What was the high temperature in Tokyo yesterday?")
```

### LangGraph (新一代)

支持更复杂的Agent架构：
- 多Agent协作
- 条件分支
- 循环处理

## Agent vs Chatbot

| 维度 | Chatbot | Agent |
|-----|--------|-------|
| 能力 | 回答问题 | 执行任务 |
| 工具 | 无 | 有 |
| 交互 | 单轮 | 多轮 |
| 自主性 | 低 | 高 |

## Agent的评估指标

### 执行效率
- 工具选择准确率
- 任务完成率
- 平均步数

### 质量
- 正确使用工具
- 正确理解结果
- 错误恢复能力

### 安全性
- 工具调用安全
- 权限控制
- 无限循环防护