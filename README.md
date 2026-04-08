# AI Legal Agent Team

面向法律文档分析的多 Agent + RAG 工程项目。系统支持上传合同进行审查，也支持基于本地法律知识库做问答；在交互层提供 `Analysis / Key Points / Recommendations` 三段式输出，在工程层提供离线、检索、端到端三类评测闭环。

## 核心亮点

- 多 Agent 分工：将条款分析、法律研究、风险判断、本地知识库问答拆分为不同角色，减少单模型回答过散的问题。
- 双知识源检索：上传文档与本地法律知识库均进入 Qdrant，回答建立在检索证据之上。
- 三段式输出：主分析之外，补充 `Key Points` 与 `Recommendations`，更贴近实际法律分析交付形态。
- 评测链路完整：同时提供离线逻辑评测、检索评测、端到端评测，便于做回归和结果对比。
- 工程结构清晰：主应用、评测入口、评测辅助、数据生成脚本已经分层整理，便于继续扩展。

## 功能范围

- `Contract Review`
- `Legal Research`
- `Risk Assessment`
- `Compliance Check`
- `Custom Query`
- `Local Query`

其中前五类任务面向上传文档，`Local Query` 面向本地法律知识库。

## 技术栈

- 前端：`Streamlit`
- Agent 框架：`Agno`
- 模型：`DeepSeek` 推理，`OpenAI-compatible API` 用于 case 生成与评测辅助
- 向量数据库：`Qdrant`
- 检索与嵌入：`Sentence Transformers`
- 配置管理：`python-dotenv`

## 系统结构

```text
User
  |
  v
app.py
  |
  +--> src/kb_service.py
  |      +--> 上传文档处理
  |      +--> 本地知识库构建
  |      +--> Qdrant 检索
  |
  +--> src/agent_service.py
         +--> Clause Analyst
         +--> Legal Researcher
         +--> Risk Strategist
         +--> Local Legal KB Agent
         |
         +--> Analysis
         +--> Key Points
         +--> Recommendations
```

## 目录结构

```text
Ai_legal_agent_team/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── legal_kb.txt
├── src/
│   ├── agent_service.py
│   ├── kb_service.py
│   └── logging_config.py
├── eval/
│   ├── cases/
│   │   ├── eval_data/
│   │   ├── contract_review/
│   │   ├── legal_research/
│   │   ├── risk_assessment/
│   │   ├── compliance_check/
│   │   ├── custom_query/
│   │   └── local_query/
│   ├── runners/
│   │   ├── run_unified_eval.py
│   │   ├── run_retrieval_eval.py
│   │   └── build_eval_collections.py
│   ├── support/
│   │   ├── assertions.py
│   │   ├── provider_agent_service.py
│   │   └── tests_from_cases.py
│   ├── offline/
│   │   ├── eval_agent_service.py
│   │   ├── eval_cases.json
│   │   └── eval_report.json
│   ├── reports/
│   ├── retrieval_reports/
│   └── sim_reports/
├── scripts/
│   ├── gen_case.py
│   └── gen_local_case.py
└── logs/
```

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 并填写：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_CASE_GEN_MODEL=gpt-4.1-mini
PROMPTFOO_TARGET_MODEL=deepseek-chat
EVAL_OPENAI_API_KEY=
EVAL_OPENAI_BASE_URL=
EVAL_OPENAI_MODEL=
DEEPSEEK_API_KEY=
QDRANT_API_KEY=
QDRANT_URL=
FILE_PATH=./data/legal_kb.txt
LOG_DIR=logs
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
streamlit run app.py
```

## 评测体系

当前项目有三类评测：

- 离线逻辑评测：检查 prompt、路由、Agent 配置是否符合设计
- 检索评测：检查文档检索和本地知识库检索是否命中预期目标
- 端到端评测：检查最终回答在 `Analysis / Key Points / Recommendations` 三阶段的表现

### 离线逻辑评测

用于快速回归 `src/agent_service.py` 的配置和 prompt 构造逻辑，不调用真实模型，也不调用向量库。

```bash
python eval/offline/eval_agent_service.py
```

输入：
- `eval/offline/eval_cases.json`

输出：
- `eval/offline/eval_report.json`

当前结果：

| 指标 | 数值 |
|---|---:|
| Case 数 | 6 |
| 通过 Case 数 | 6 |
| 总体通过率 | 100% |
| 总检查项得分 | 30 / 30 |
| 检查项命中率 | 100% |

这里的 `100%` 是结构命中率，不是回答正确率。它衡量的是：
- Agent 配置是否匹配
- 路由是否正确
- prompt 是否包含必需关键词
- prompt 是否规避禁止关键词
- 是否能生成非空响应

### 端到端评测

端到端评测支持：
- `team`：多智能体
- `single`：单智能体

```bash
PROMPTFOO_EXECUTION_MODE=team \
python eval/runners/run_unified_eval.py \
  --analysis-type "Custom Query" \
  --doc-index 1 \
  --mode e2e \
  --write-stage-reports
```

说明：
- `--mode single`：只评主分析
- `--mode e2e`：评 `analysis + key_points + recommendations`
- `--write-stage-reports`：额外输出三个阶段的扁平化报告

在线任务总体结果如下：

| 设置 | Case 数 | 平均主分析分 | 平均关键点总结分 | 平均后续建议分 | 平均端到端总分 | 通过 Case 数 | 总体通过率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 单智能体 | 50 | 0.56 | 0.80 | 0.62 | 0.66 | 30 | 60% |
| 多智能体 | 50 | 0.69 | 0.88 | 0.77 | 0.78 | 39 | 78% |

本地任务总体结果如下：

| 设置 | Case 数 | 平均主分析分 | 平均关键点总结分 | 平均后续建议分 | 平均端到端总分 | 通过 Case 数 | 总体通过率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 不带检索器本地 Agent | 10 | 0.72 | 0.84 | 0.74 | 0.77 | 5 | 50% |
| 带检索器本地 Agent | 10 | 0.88 | 0.95 | 0.89 | 0.91 | 8 | 80% |

### 检索评测

检索评测直接评估底层检索结果，不走完整回答生成链路。

```bash
PROMPTFOO_EXECUTION_MODE=team \
python eval/runners/run_retrieval_eval.py \
  --analysis-type "Custom Query" \
  --doc-index 1 \
  --top-k 3
```

说明：
- 在线任务主要看文档向量检索
- `Local Query` 主要看本地知识库检索
- 当前评分方式以目标命中为主，不是标准 IR 指标

当前综合结果如下：

| 任务类型 | Case 数 | 文档检索平均分 | 本地知识库检索平均分 | 综合检索平均分 | 通过率 |
|---|---:|---:|---:|---:|---:|
| 合同审查 | 10 | 0.79 | 不适用 | 0.79 | 70% |
| 法律研究 | 10 | 0.74 | 0.68 | 0.71 | 60% |
| 风险评估 | 10 | 0.73 | 0.66 | 0.70 | 60% |
| 合规检查 | 10 | 0.71 | 0.64 | 0.68 | 50% |
| 自定义问答 | 10 | 0.67 | 0.58 | 0.63 | 50% |
| 本地知识问答 | 10 | 不适用 | 0.84 | 0.84 | 80% |

`不适用` 表示该任务在当前 case 设计下，不把这类检索源作为正式评测维度。

### 预构建评测文档集合

批量评测前可以先构建在线文档集合，减少首次运行开销。

```bash
python eval/runners/build_eval_collections.py
```

或只构建单个文档：

```bash
python eval/runners/build_eval_collections.py --doc-index 1
```

## 输出目录

端到端评测默认输出到时间戳目录：

```text
eval/reports/<timestamp>_<task>_<mode>_<execution_mode>/
├── <task>_e2e_report.json
├── <task>_analysis_report.json
├── <task>_key_points_report.json
└── <task>_recommendations_report.json
```

检索评测默认输出到时间戳目录：

```text
eval/retrieval_reports/<timestamp>_<task>_retrieval_<execution_mode>/
└── <task>_retrieval_report.json
```

## 当前结论

- 多智能体相较单智能体，在在线任务端到端评测中有稳定提升，优势主要集中在主分析与后续建议阶段。
- 本地任务中，带检索器版本显著优于不带检索器版本，说明本地知识库检索对答案质量有直接帮助。
- 当前系统最稳的任务是合同审查和本地知识问答；自定义问答、合规检查仍然是后续优化重点。
- 项目已经具备比较完整的工程闭环：可交互、可检索、可多 Agent 协作、可回归评测。

## 致谢

本项目参考了 `awesome-llm-apps` 中的 `AI Legal Agent Team` 示例，并在此基础上做了多 Agent 重构、RAG 强化与评测链路补充。
