# AI 法律文档多 Agent 分析系统

这是一个面向法律文档分析场景的多 Agent 项目，核心目标是让大模型不只是“直接回答问题”，而是先做文档检索、角色分工、风险归纳，再输出更接近真实法律分析流程的结果。

项目支持上传法律文档进行审查，也支持基于本地法律知识库进行问答，适合作为大模型应用、RAG、多 Agent 协作、法律科技方向的项目展示。

## 项目亮点

- 多 Agent 分工明确：将法律分析拆分为条款分析、法律研究、风险策略、本地知识库问答四类角色，避免单个模型回答过于发散。
- RAG 检索链路完整：上传文档和本地法律知识库都会进入 Qdrant，回答前先检索证据，再生成结论。
- 支持真实业务场景：覆盖合同审查、法律研究、风险评估、合规检查、自定义问题、本地法律问答六类任务。
- 工程结构清晰：将 UI、Agent 编排、知识库构建、日志模块解耦，便于扩展和维护。
- 有真实评测闭环：不仅有离线逻辑评测脚本，还补充了项目级真实评测脚本，可实际调用模型、向量库并落盘报告。
- 可解释性更强：相比直接调用大模型，本项目的回答建立在文档片段或本地知识库检索结果之上，更适合展示“有依据的分析”。

## 核心功能

### 1. 合同审查

针对上传合同，提取关键条款、付款条件、违约责任、终止条件、争议条款，并指出模糊表述、缺失保护和 drafting weakness。

### 2. 法律研究

围绕上传文档涉及的法律问题，补充法律原则、判例、合规背景和监管风险，并区分“文档原文事实”和“外部法律研究”。

### 3. 风险评估

对合同中的问题条款和缺失条款做优先级排序，输出高、中、低风险及对应缓释建议。

### 4. 合规检查

从合规视角识别文档中的风险点，如数据处理、通知义务、监管义务、流程缺失，并给出整改建议。

### 5. 自定义法律问题

允许用户围绕上传文档自由提问，例如：

- 这份合同里最值得谈判的条款是什么？
- 责任上限条款有哪些风险？
- 如果我是甲方，应该优先补哪些保护条款？

### 6. 本地法律知识库问答

支持仅基于本地法律知识库回答，不依赖上传文档，适合做基础法律概念问答。

## 技术方案

### 技术栈

- 前端：`Streamlit`
- 大模型：`DeepSeek`
- Agent 框架：`Agno`
- 向量数据库：`Qdrant`
- 检索与嵌入：`Sentence Transformers`
- 配置管理：`python-dotenv`
- 日志：自定义 `logging_config.py`

### 系统架构

```text
用户
  |
  v
Streamlit 页面（app.py）
  |
  +--> 上传 PDF / 文本
  |        |
  |        v
  |   kb_service.py
  |        |
  |        +--> 文本切块
  |        +--> 向量化
  |        +--> 写入 Qdrant
  |
  +--> 分析请求
           |
           v
      agent_service.py
           |
           +--> Clause Analyst
           +--> Legal Researcher
           +--> Risk Strategist
           +--> Local Legal KB Agent
           |
           v
        结构化分析结果
```

## Qdrant 在项目中的作用

Qdrant 是本项目的核心检索层。

它负责存储两类知识：

- 用户上传的法律文档
- 本地法律知识库 `data/legal_kb.txt`

整个流程是：

1. 文档切块
2. 文本向量化
3. 向量写入 Qdrant
4. Agent 回答前先检索相关片段
5. 模型基于检索结果生成最终分析

如果没有 Qdrant，这个项目就会退化成“直接把问题丢给大模型”，会明显损失：

- 对合同具体条款的定位能力
- 对上传文档的证据支撑能力
- 本地知识库问答的约束能力
- 多 Agent 结果的可信度

## 目录结构

```text
Ai_legal_agent_team/
├── app.py
├── agent_service.py
├── kb_service.py
├── logging_config.py
├── requirements.txt
├── .env.example
├── README.md
├── eval_agent_service.py
├── eval_cases.json
├── evaluation/
│   ├── cases/
│   ├── fixtures/
│   ├── results/
│   └── run_project_eval.py
└── data/
    └── legal_kb.txt
```

## 项目效果与评测结果

项目包含两类评测：

- 离线逻辑评测：验证路由、prompt 构造、分析模式配置
- 项目级真实评测：实际调用 `DeepSeek + Qdrant`，输出完整报告

代表性真实评测结果：

- 总 case 数：`5`
- 通过数：`4`
- 总分：`14/15`
- 得分率：`93.33%`

对应报告文件：

- [主评测报告](/hard_data1/user/yangguobin/LLM/Ai_legal_agent_team/evaluation/results/20260331_132840/report.json)

通过的能力项：

- `Contract Review`
- `Risk Assessment`
- `Custom Query`
- `Local Query`

当时未满分的项：

- `Compliance Check`

原因不是回答完全错误，而是规则评测里少命中了 `privacy` 关键词。后续单独复跑后，该 case 也达到了 `3/3`：

- [Compliance Check 单独复跑报告](/hard_data1/user/yangguobin/LLM/Ai_legal_agent_team/evaluation/results_case3/20260331_134347/report.json)

这说明项目整体是有效的，问题主要集中在回答措辞稳定性，而不是系统链路失效。

## 我在项目中解决的关键问题

### 1. 多 Agent 职责边界

如果不做角色拆分，模型容易把“条款抽取、法律研究、风险建议”混在一起，输出既长又散。这个项目通过角色定义和团队 prompt，把任务分配清楚，提升了输出结构性。

### 2. 文档 grounding

法律场景对“有依据”要求很高，所以我没有只做一个聊天机器人，而是接入 Qdrant 做检索增强，让分析建立在上传文档和本地知识库之上。

### 3. 可评测性

很多大模型项目只能演示，不能验证。我补了项目级真实评测脚本，让系统可以批量跑 case、自动打分、输出结果文件夹。

### 4. 工程可维护性

项目不是把所有逻辑都塞进一个文件，而是拆成：

- `app.py`：页面与交互
- `agent_service.py`：角色定义与多 Agent 编排
- `kb_service.py`：知识库、向量库、切块、去重、检索
- `logging_config.py`：日志配置

这让项目更像一个可扩展原型，而不是一次性脚本。

## 当前不足与可优化点

- 输出控制还可以更严格：当前部分结果会暴露类似 “I’ll delegate...” 的协作语句，说明 Team 输出格式约束还可以继续收紧。
- 合规分析的稳定性还可提升：`Compliance Check` 在主批评测里出现过一次关键词覆盖不完整的问题。
- 推理时延偏长：多 Agent + 检索 + query expansion 会让单次分析耗时比较高，后续可以考虑缓存、裁剪上下文或减少不必要角色参与。
- 评测维度还可以更细：目前以规则命中和长度为主，未来可以补充人工标注评分或更细粒度的法律任务指标。

## 本地运行

### 环境变量

先根据 `.env.example` 配置：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
FILE_PATH=./data/legal_kb.txt
LOG_DIR=logs
```

### 安装依赖

```bash
pip install -r requirements.txt
```

当前依赖里包含一些兼容性约束，主要是为了让评测链路稳定运行，例如：

- `ddgs`
- `transformers<5`
- `huggingface-hub<1.0`

### 启动项目

```bash
streamlit run app.py
```

## 评测脚本使用

### 1. 离线逻辑评测

```bash
python eval_agent_service.py
```

用于验证：

- 分析模式配置是否正确
- 路由是否正确
- prompt 构造是否符合预期

### 2. 项目级真实评测

```bash
python evaluation/run_project_eval.py
```

它会：

- 构建主文档知识库
- 构建本地法律知识库
- 真实调用 `DeepSeek`
- 生成评测报告和每个 case 的输出文件

默认输出目录结构：

```text
evaluation/results/<timestamp>/
├── report.json
├── run_config.json
└── responses/
    ├── 01_contract_review_sample_contract.md
    ├── 02_risk_assessment_sample_contract.md
    └── ...
```

## 致谢

本项目参考了 `awesome-llm-apps` 中的 `AI Legal Agent Team` 示例，并在此基础上做了简化、重构和评测增强：

- 原始仓库：https://github.com/Shubhamsaboo/awesome-llm-apps
