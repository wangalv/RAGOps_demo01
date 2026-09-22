---
name: project-langgraph-learning
description: 6-module LangGraph/ReAct/Eval learning plan tied to LegislationParser PDF upload pipeline project
metadata: 
  node_type: memory
  type: project
  originSessionId: ba514c41-891a-418d-a8f3-423e68287f3c
  modified: 2026-09-20T00:13:17.683Z
---

# LangGraph 学习计划 — 6个模块，以 LegislationParser PDF 上传 Pipeline 为载体

**开始日期**：2026-08-18
**目标**：在接下来数天内逐模块学完，用户希望手把手边学边做。

## 模块分配

| 模块 | 主题 | 核心交付 |
|---|---|---|
| M1 | LangGraph 核心：State / Node / Edge | 能跑通单 PDF 线性 Graph（无 LLM，假数据） |
| M2 | LangGraph 进阶：并行 / 子图 / Checkpointing / 流式 | 批量3个 PDF 并行处理 + Streamlit 实时进度 |
| M3 | ReAct 原理 + 手写循环 | ExtractionAgent 自主决定调哪些 Skill |
| M4 | ReAct 全流程 + Human-in-the-loop | 上传5个 PDF 全自动跑，低置信度时问人 |
| M5 | Evaluation 框架搭建 | `eval/golden_cases.jsonl` + `eval/run_eval.py` 4维度跑通 |
| M6 | 用评测结果迭代 Agent | 改进前后指标对比，优化提示词/工具描述/调用次数 |

## 4个评测维度（M5/M6）

1. **Tool 选择准确率** — 用 golden test set 算 precision / recall
2. **幻觉率** — LLM-as-Judge 对比原文与提取结果
3. **成本** — 每个节点记录 token 用量和 USD 花费
4. **端到端 F1** — 和人工标注对比提取结果

## 工作目录

`/Users/alvinwang/Documents/AI Development/LegislationParser_v1/`

## 模块分配（更新）

| 模块 | 主题 | 核心交付 |
|---|---|---|
| M1 | LangGraph 核心：State / Node / Edge | 能跑通单 PDF 线性 Graph（无 LLM，假数据） |
| M2 | LangGraph 进阶：并行 / 子图 / Checkpointing / 流式 | 批量3个 PDF 并行处理 + Streamlit 实时进度 |
| M3.0 | **Agent Harness 从零手写（200行）** | 不用任何框架，手写 tool registry + agent loop，理解 LangGraph 在做什么 |
| M3 | ReAct 原理 + 手写循环 | ExtractionAgent 自主决定调哪些 Skill |
| M4 | ReAct 全流程 + Human-in-the-loop | 上传5个 PDF 全自动跑，低置信度时问人 |
| M5 | Evaluation 框架搭建 | `eval/golden_cases.jsonl` + `eval/run_eval.py` 4维度跑通 |
| M6 | 用评测结果迭代 Agent | 改进前后指标对比，优化提示词/工具描述/调用次数 |
| M7 | RAG 召回率 + Precision/Recall 平衡 | Hybrid Search + Reranking + HyDE 跑通，对比 baseline recall；掌握场景判断框架 |
| M8 | Agent 可靠性：熔断 / 降级 / 限流 / 重试 | 在 LangGraph Pipeline 里实现四个可靠性模式 |
| M10 | **Autonomous ML Experimentation（autoresearch 模式）** | 用 AI agent 自主迭代 AML demo 的 XGBoost 模型，overnight 找最优特征/超参数 |

## M3.0 — Agent Harness 从零手写（200行 Python）

**参考文章**：[The Emperor Has No Clothes](https://www.mihaileric.com/The-Emperor-Has-No-Clothes/) — Mihail Eric（前 Stanford 讲师）
**Why:** 在用 LangGraph 框架之前，先手写最小可用的 agent loop，理解框架底层在做什么。用200行 Python 完全实现一个能读/列/编辑文件的 coding agent。

### 核心概念

**Agent Loop（核心循环）**：
```
用户输入 → LLM 决定要不要调工具 → 你的代码执行工具 → 结果返回 LLM → 继续对话
```
LLM 永远不接触文件系统——它只"请求"工具被调用，代码执行。

**三个组件**：
1. **Tool Registry** — `dict[name → function]`，LLM 通过名字调用
2. **System Prompt 工程** — 从函数 docstring 自动生成工具描述，告诉 LLM 格式：`tool: TOOL_NAME({JSON_ARGS})`
3. **Agent Loop** — 外循环（读用户输入）+ 内循环（LLM call → 解析 tool → 执行 → 直到无 tool call 为止）

**三个工具**（足够实现 coding agent）：
- `read_file(filename)` → 返回文件内容
- `list_files(path)` → 列出目录
- `edit_file(path, old_str, new_str)` → `old_str=""` 时创建新文件，否则替换第一个匹配

### 交付物

`src/agent/minimal_harness.py`（~200行）：
- 完整可运行的 coding agent
- 支持多步操作：LLM 自动 read → edit → confirm
- 无任何框架依赖（只用 anthropic SDK）

### 和 M3/LangGraph 的关系

| | M3.0 手写 Harness | M3 LangGraph |
|---|---|---|
| 工具注册 | `TOOL_REGISTRY = {...}` | `@tool` decorator |
| Loop 控制 | `while True` 内循环 | `StateGraph` + conditional edges |
| 状态管理 | `conversation: List[dict]` | `TypedDict` state |
| Checkpointing | 无 | `MemorySaver` |
| Human-in-loop | 自己实现 `input()` | `interrupt_before` |

**M3.0 的价值**：理解 LangGraph "帮你做了什么"，遇到框架问题时能回到原理层调试。

### 进度
- [ ] M3.0 — Agent Harness 从零手写

---

## M7 子模块 — RAG Recall 提升（详细学习计划）

**学习目标**：理解并动手实现 5 个提升 RAG 召回率的核心技术，用立法文档作为实验数据。

### 学习顺序（由易到难）

#### M7.1 — 基础 RAG Pipeline（baseline）
- 切块（按 section）→ embedding → Vector DB（ChromaDB）→ top-k 检索 → 喂给 LLM
- **交付**：能跑通基础 RAG，记录 baseline recall@5

#### M7.2 — Hybrid Search（Dense + Sparse）
- **Dense**：向量相似度（semantic meaning）；适合同义词/近义词，不适合精确关键词
- **Sparse / BM25**：词频统计匹配；适合精确术语/专有名词/法律编号，不适合同义词
- **Hybrid = 两者结合**，用 RRF（Reciprocal Rank Fusion）合并排名，互相补盲区
- **RRF 公式**：score = 1/(60+rank_dense) + 1/(60+rank_bm25)；k=60 是标准阻尼常数；两边都出现的结果自动排前
- **为什么有用**："false"、"mandatory"这种精确词 BM25 直接命中；"licensing"≈"registration"Dense 桥接
- **BM25 解决不了的**：查询用 "licensing"，Act 完全没有这个词 → 还是找不到
- **实验发现**：BM25 在法律文本上精度低——关键词出现但上下文不对（"reporting"→"annual reporting"，"procedure"→"general procedure"），RRF 把噪音放大，Hybrid 有时比 Dense 更差
- **根本原因**：BM25 默认 unigram，不理解短语；改进方向见下方"BM25 改进方法"
- **交付**：实现 BM25 + 向量 merge，recall 对比 baseline

#### M7.2 补充 — BM25 短语匹配改进方法（复盘用）

实验发现 unigram BM25 在法律文本里精度差，以下三种改进方法由易到难：

| 方法 | 术语 | 原理 | 难度 |
|---|---|---|---|
| N-gram BM25 | Bigram / Trigram | 把相邻词组合成一个 token，"mandatory reporting" 变成 `mandatory_reporting`，不会匹配 "annual reporting" | 简单，改一行 tokenizer |
| 位置索引 | Positional Index / match_phrase | 记录词在文档里的位置，检查查询词是否相邻出现；Elasticsearch 的 `match_phrase` 是这个原理 | 中等，需要搭 Elasticsearch |
| 词级别向量 | ColBERT | 每个词保留自己的向量（不压缩成一个），查询时词级别相似度匹配，短语语义保留更好 | 复杂，需要专门模型 |

**最快改法（Bigram）**：
```python
def bigram_tokenize(text):
    words = text.lower().split()
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams
```

#### M7.3 — 更大 top-k + Reranking
- 先取 top-20（召回更多候选）
- 再用 cross-encoder 重排，取前 5 喂给 LLM（保持精度）
- **为什么有用**：top-5 漏的，top-20 里可能有；reranker 比 embedding 更精准判断相关性
- **交付**：实现 reranking，precision/recall 双指标对比

#### M7.4 — Parent-Child Chunking
- 小块（sentence级）用于检索（精准定位）
- 大块（section级）用于生成（保留上下文）
- 检索到小块 → 自动返回它的父块给 LLM
- **为什么有用**：块太小丢上下文，块太大稀释相似度；两级结构两全其美
- **交付**：实现两级 chunk 结构，对比单级效果

#### M7.5 — HyDE（Hypothetical Document Embeddings）
- 用户问题 → LLM 先生成一个"假设的答案" → 对假设答案做 embedding → 检索（不直接 embed 查询）
- **为什么有用**：假设答案语言风格 ≈ 法律文档语言风格，向量距离更近；解决口语查询 → 专业文档的风格差距
- **适合场景**：用户用口语/中文查，文档是英文法律条文；查询太短信息量不足
- **不适合场景**：查询本身已是专业术语（HyDE 增加 LLM 成本但收益小）
- **和 BM25 的分工**：BM25 解决精确词命中，HyDE 解决语言风格桥接；两者互补
- **交付**：实现 HyDE，在专业术语查询上对比直接 embedding 的效果

#### M7.5 补充 — 其他提高 Recall / Precision 的方法（复盘用）

| 技术 | 类别 | 原理 | 复杂度 |
|---|---|---|---|
| Multi-Query Retrieval | Recall | LLM 把查询改写成 3~5 种表达，每个都检索，结果取并集；适合查询太短或歧义多；**风险：变体语义漂移（Q3 实验："minister's regulations" 变体生成 "delegated legislation/legislative instruments"，触发 delegation sections，把正确答案 s.140 挤出候选池）；避免方法见下方补充** | 中 |
| Query Expansion | Recall | 查询里加同义词（BM25 层有效）；Dense 层意义不大 | 低 |
| Step-back Prompting | Recall | LLM 先把具体问题抽象成宏观问题，两次检索结果合并；召回更多相关条款 | 中 |
| Better Embedding Model | Recall+Precision | 换更大/专用模型：bge-large-en-v1.5（1024维）、voyage-law-2（法律专用 API）；质的提升 | 低（换模型） |
**Multi-Query 实验结论（复盘用）**

实验了三轮，核心发现如下：

**Round 1（无约束，per_query_k=10）**：
- Q1 ✅ 改善（"notifiable conduct" 变体召回 s.99）
- Q3 ❌ 变体漂移："delegated legislation" 淹没 s.140

**Round 2（Prompt约束 + 相似度过滤 sim≥0.75，per_query_k=10）**：
- 过滤掉了漂移变体 ✅，但阈值太高误杀 "compulsory notification duties"（sim=0.689）→ Q1 失去 s.99
- per_query_k=10 太小，s.140 在原查询 rank14，进不了候选池 → Q3 仍失败

**Round 3（Prompt约束只，per_query_k=20）**：
- 变体太保守（"sanction for furnishing untruthful details" ≈ 原查询），候选池没扩大 → Q1/Q2/Q4 和 Dense→Reranked 完全一样
- Q3 找到 s.140（rank 2）但引入 s.63 by-laws 噪音（rank 1，score 3.906）

**根本矛盾**：Multi-Query 要有用，变体必须用不同词汇 → 但"不同词汇"和"不引入新概念"天然矛盾。

**实践结论**：
- Multi-Query 收益在法律长查询上不稳定；更适合查询极短/歧义大的场景（如电商搜索）
- 法律 RAG 优先用：Reranking（Q3 稳定）+ Parent-Child（Q1 稳定）+ HyDE（词汇差距）
- 若要用 Multi-Query，需要 per_query_k ≥ 正确答案在单查询里的 rank（否则正确答案根本进不了候选池）

| Metadata Filtering | Precision | 检索前用结构化条件过滤（section_type、chapter）；缩小搜索范围 | 中 |
| Contextual Compression | Precision | 检索到 section 后，LLM 过滤掉不相关句子，只传核心内容给生成模型 | 中 |
| Lost in the Middle 修复 | Precision | LLM 对长 context 首尾记得住、中间忽略；把最相关 chunk 放首尾，低相关放中间 | 低 |
| Self-RAG / Adaptive RAG | Recall+Precision | LLM 自决定要不要检索、检索后判断够不够用、不够再检索；最智能，接近 ReAct | 高 |

#### M7.6 — Precision vs Recall 平衡判断框架
- 核心问题：漏掉 vs 错掉，哪个代价更大？
- 3步判断法：① 识别使用者（专业/普通）② 有无 Human-in-the-loop ③ 错误是否对称
- 场景映射：全量提取 obligation → 优先 Recall；垃圾邮件过滤 → 优先 Precision；推荐系统 → 平衡 F1
- 技术选择映射：Recall优先→大top-k/Hybrid/HyDE；Precision优先→Reranking/元数据过滤
- 立法提取结论：强烈优先 Recall（专业用户 + Human-in-the-loop + 漏掉是沉默错误）

#### M7.6 补充 — RAG Evaluation 框架（复盘用）

RAG 评测分两层：检索质量 + 生成质量，对应不同评测方式。

**4个评测维度（RAGAS 框架）：**

| 维度 | 问的问题 | 评测方式 |
|---|---|---|
| Context Recall | 检索到的 chunks 里，回答必需的内容有多少被覆盖？ | LLM-as-Judge 对比 golden answer |
| Context Precision | 检索到的 chunks 里，有多少是真正相关的（无噪音）？ | LLM-as-Judge 逐个判断 |
| Faithfulness | LLM 最终答案有没有幻觉/编造原文没有的内容？ | LLM-as-Judge 对比答案和 chunks |
| Answer Relevance | 最终答案有没有真正回答用户问题？ | LLM-as-Judge 判断答案和查询相关性 |

**两种评测方式：**
- **Golden Set（人工标注）**：准备 20~50 个查询，人工标注哪些 chunk 相关 + 标准答案 → 自动算 Recall@k / Precision@k；客观准确，但需要人工投入
- **LLM-as-Judge**：把"查询 + 检索结果 + 标准答案"给 LLM 打分；不需要人工标注，但评分有波动

**在 M5 里的分工：**
- Golden Set → 评检索质量（Recall@3、Precision@3）
- LLM-as-Judge → 评生成质量（Faithfulness、Answer Relevance、幻觉检测）
- RAGAS 框架把 4 个维度整合成自动化 pipeline

### 评测方式
- 构造 10~20 个测试查询（每个查询人工标注哪些 section 是相关的）
- 对每种技术计算 Recall@3 和 Recall@5
- 最终形成对比表：baseline vs Hybrid vs Reranking vs Parent-Child vs HyDE vs 组合方案
- M5 整合 RAGAS 框架，同时评检索质量（Golden Set）和生成质量（LLM-as-Judge）

## M5 评测框架 — 四阶段计划

RAGOps 和 AgentOps 是两个不同层面，分开学：RAGOps 关注检索质量/延迟/成本，现在就能做；AgentOps 关注 Agent 决策/幻觉/工具选择，需要 M3/M4 先完成。

### M5.1 — Golden Set 构造 ✅
- `GoldenSet_v2.xlsx`，18个 query，4种类型（Exact/Multi/Cross/Ambiguous）
- 标注 primary + secondary relevant sections

### M5.2 — RAG 检索评测脚本 ✅
- `eval/run_eval.py`，7种技术 × 18 query
- 指标：Recall@3/5 + Precision@3/5，按 Query Type 分组
- 结论：Dense→Reranked 最优（R@3=0.954），LLM 技术无额外增益

### M5.3 — RAGOps（Langfuse，现在做）
**目标**：把 M7 的最优检索路径（Dense→Reranked）接入 Langfuse，实现生产可观测性

- **全链路 Trace**：query → embedding → ChromaDB → rerank → chunks，每步打点
- **延迟分析**：embedding / 向量检索 / rerank 各占多少时间
- **成本追踪**：按 node 拆分 token 用量（HyDE/MultiQuery 的 LLM 调用成本）
- **Prompt 版本管理**：HyDE 和 Step-back 的 prompt 接入 Langfuse Prompt Management
- **工具**：Langfuse（开源可自建，也有 cloud 版）

### M5.4 — AgentOps（LangSmith，待 M3/M4 完成后）
**目标**：评测 ReAct Agent 的端到端行为

- **LangSmith**：管理 Dataset、记录 Agent trace、LLM-as-Judge
- **评测维度**：Tool 选择准确率 / 幻觉率 / 成本 / 端到端 F1
- **前提**：M3 ReAct Agent 先做出来

## M9 — 用 Langfuse 实现 RAGOps Pipeline（生产可观测性，M5/M6 之后）

**目标**：把 M5/M6 的离线评测能力延伸到线上，让 Pipeline 具备可观测、可追溯、可持续改进的生产运维能力。

**和 LangSmith（阶段二 Agent 评测）的分工**：
- LangSmith：开发阶段跑 Agent trace + LLM-as-Judge，服务于 M3/M4 迭代
- Langfuse：生产阶段的 RAGOps 层——自建/开源可控、trace 之外还管 prompt 版本和线上成本看板；两者可以并存，也可以后续只保留一个

**核心交付**：
1. **全链路 Trace**：retrieval → rerank → parent-child 展开 → LLM 生成，每一跳都打点，定位问题出在检索还是生成
2. **Prompt 版本管理**：M6 迭代出的多版本 prompt 接入 Langfuse Prompt Management，支持线上灰度/回滚
3. **线上质量监控**：M5 的 Golden Set / LLM-as-Judge 评分接入 Langfuse Score，做成持续评测（新查询定期抽样跑分，而非只在开发时跑一次）
4. **成本/延迟看板**：按 node（embedding / rerank / LLM 生成）拆分 token 用量和延迟，定位瓶颈
5. **评测结果回流**：线上发现的 bad case 反哺 Golden Set，形成闭环

## M5.3 RAGOps — Langfuse Experiments 结果汇总

**最优技术**：HyDE→Reranked@3，avg R@3 = 0.881（21条 golden set）
**Golden Set**：GoldenSet_v2.xlsx，21条 query，4种类型
**评测脚本**：`eval/langfuse_experiment.py`，`--run` 参数指定只跑哪些

| Technique | R@3 | Total Cost/query |
|---|---|---|
| HyDE→Reranked@3 | **0.881** | ~$0.00054 |
| StepBack→Rnk@3 | 0.870 | ~$0.00027 |
| Ensemble-All@3 | 0.865 | ~$0.00120 |
| MultiQuery→Rnk@3 | 0.838 | ~$0.00038 |
| Dense→Reranked@3 | 0.870 | ~$0 |
| Hybrid→Reranked@3 | 0.771 | ~$0 |
| ParentChild→Rnk@3 | 0.724 | ~$0 |

**Ensemble 没有超过 HyDE 的原因**：候选池从20扩到~80，Reranker 面对更多噪音，反而降低排序质量。Dense+HyDE 已覆盖大多数 case，加 MultiQuery/StepBack 是边际递减。

## M5.3 Layer 3 — 生产监控模拟（待做）

### 计划 A：simulate_queries.py（生成新查询）
- 用 LLM 生成 100+ 条新 query，覆盖 procedural/definitional/penalty/authority/timeframe/cross-section 六种类型
- 通过 LLM-as-Judge 打分（无 ground truth）
- 目的：发现 HyDE→Reranked@3 在哪类 query 上失败

### 计划 B：新增真实法案（加入 corpus）
- **目标法案**：Health Administration Act 1982 No 135（目前仍有效）
  - URL：https://legislation.nsw.gov.au/view/whole/html/inforce/current/act-1982-135
  - 原因：是 Health Services Act 1997 的"姐妹法案"，Health Administration Corporation 在此定义，高度相关但有不同结构
- **方案**：抓取 HTML → chunk_by_section → 加入新 collection → 搜索时双 collection 合并 rerank
- **评测**：重跑 golden set 对比 R@3 before/after，观察 precision 是否因 corpus 扩大而下降

## M5.3 Layer 3 — 模拟查询 + Grounded-77 评测结果（完成）

### 核心发现（供总结 Demo 使用）

#### 1. 评测集规模的影响
| 数据集 | n | R@3 | R@5 | P@3 | P@5 |
|--------|---|-----|-----|-----|-----|
| 原始GoldenSet（手工精选）| 21 | 0.881 | 0.910 | 0.37 | 0.23 |
| 100条模拟（含无答案） | 100 | 0.69 | 0.73 | 0.19 | 0.13 |
| 79条 grounded only | 79 | 0.646 | 0.750 | 0.27 | 0.19 |

**结论**：原始21条高估了约17-27%的真实性能。0.75才是真实生产基线。

#### 2. 各检索方法在 Grounded-77 上的表现（HyDE→Reranked@5 最优）
| 方法 | R@5 | P@3 |
|------|-----|-----|
| HyDE→Reranked@5 | **0.750** | 0.270 |
| Dense→Reranked@5 | 0.732 | **0.278** |
| Hybrid→Reranked@5 | 0.726 | 0.266 |
| ParentChild→Rnk@5 | 0.713 | 0.266 |
| HyDE→LLMRerank@5 | 0.696 | 0.278 |（LLM重排无显著提升）

#### 3. 25% Recall 失败的根本原因（diagnose_recall_failures.py）
- **Penalty 查询最差（4/6失败，avg R@5=0.333）**：Query用"fine/penalty"，Act用"guilty of an offence"，词汇层面完全错位
- **CrossSection 跨章节关系题（3/12失败，avg R@5=0.486）**：查询问的是章节"interaction"，是元层面问题，法案没有这样的条文
- **Prohibition 反向语义（3/9失败，avg R@5=0.593）**：Query问"什么不能做"，Act只写"能做什么"，禁止是隐含的
- **Timeframe 时间细节（3/8失败，avg R@5=0.625）**：任期/期限只是section里的一句话，embedding被周围内容稀释
- **Grounded=Yes的查询 avg R@5=0.884，Partial/No只有0.49-0.50**

#### 4. 改进方向
- LLM Reranking 无显著提升（P@3: 0.270 → 0.278，在HyDE temperature噪声范围内）
- 瓶颈在Retrieval层（R@20），不在Reranking层
- 真正有效：Query改写（把"penalty/fine"转化为Act的法律语言）→ StepBack/MultiQuery值得在Grounded-77上验证
- 生产环境：0.75 R@5在法律合规场景偏低（目标≥0.85）；加"找不到答案"拒绝机制比提升recall更实际

#### 5. 脚本和数据集
- `eval/diagnose_recall_failures.py` — 逐条分析失败query，按type/grounded状态分组
- `eval/langfuse_experiment.py` — 支持 `--sim-labeled --grounded-only --run "方法名"` 在Grounded-77上跑任意方法
- Langfuse数据集：`NSW-HealthAct-SimulatedQueries-Grounded-77`
- `GoldenSet_from_100_Simulated_Queries.xlsx` — 100条模拟查询的人工标注（sheet: Golden Review）

## Query 层改进实验（Grounded-77，2026-09）

在 HyDE→Reranked@5 基线（R@5=0.750）上，尝试了三种 Query 层改进：

### 实验 1：Vocabulary-Aware HyDE（LegalHyDE）
- **思路**：改写 HyDE prompt，强制用 NSW Act 词汇生成假设文档
- **v1**（NEVER 约束）：R@5=0.48 → 更差
- **v2**（自然词汇 scaffold）：R@5=0.451 → 更差
- **根本原因**：词汇约束让 LLM 注意力从主题相关性转向语言风格，embedding 偏离主题

### 实验 2：Legal Query Expansion（LQE）
- **思路**：LLM 把用户 query 用 Act 词汇改写（如 "fine" → "maximum penalty in penalty units"），再用 Hybrid 检索
- **结果**：R@5 = 0.71（低于 HyDE 基线 0.75）
- **对 16 条失败 query 的影响**：救回 2 条，退步 2 条，净效果为负
- **原因**：改写对 ~75 条正常 query 无效，BM25 引入噪音；真正靠词汇翻译救回的只有 1 条（Timeframe: "term of" → "period of"）
- **文件**：`src/rag/legal_query_expansion_retriever.py`

### 实验 3：Query Routing
- **思路**：LLM 分类 query → Penalty/Prohibition/CrossSection/Other，路由到不同检索器
- **分类准确率**：79/79 = 100%
- **结果**：R@5 = 0.72（低于 HyDE 基线 0.75）
- **退步原因**：分类正确，但 Penalty 走 LQE、CrossSection 走 MultiQuery，这两种方法本身就比 HyDE 弱
- **文件**：`src/rag/query_router.py`、`src/rag/routed_retriever.py`

### Query 层改进总结
所有 Query 层方法均未超越 HyDE→Reranked@5 基线（0.750）。根本原因：剩余 25% 失败是结构性问题，不是词汇桥接能解决的。

---

## Chunking 粒度实验（Grounded-77，2026-09）

### 背景
当前 chunking：552 个 section-level chunks。

### 实验：Subsection-level Chunking
- **实现**：`chunk_by_subsection()` 按 `(1)`, `(2)` 等条款编号切分
- **结果**：552 sections → 1169 subsection chunks
- **文件**：`src/rag/chunker.py`（新增 `chunk_by_subsection`、`chunk_subsection_parent_child`）
- **ChromaDB collection**：`health-services-act-subsections`（1169），`health-services-act-subsec-children`（1514）

### 各 Subsec 方法结果（Grounded-77，R@5）

| 方法 | R@5 | 对比基线 | 备注 |
|------|-----|---------|------|
| **SubsecRoutedPC→Rnk@5** | **0.753** | **突破天花板** | 路由+PC+rerank，三重叠加 |
| SubsecPC→Rnk@5 | 0.750 | 持平最优 | 无 LLM，零额外成本 |
| SubsecRouted→Rnk@5 (BGE) | 0.741 | +0 | BGE 比 MiniLM +0.013 |
| SubsecRouted→Rnk@5 (MiniLM) | 0.728 | — | 比 Routed 略好 |
| SubsecHyDE→Rnk@5 | 0.68 | -0.07 | HyDE 在细粒度库效果差 |

### 关键发现
- **SubsecRoutedPC→Rnk@5（0.753）首次突破 0.75 天花板**：路由做 query 转换 → 搜句子层 → 取 subsection 父级 → rerank
- **SubsecPC（sentence→subsection parent-child）** 以零 LLM 成本达到并列第二（0.750）
- **SubsecHyDE 反而更差**：HyDE 生成的假设文档比单个 subsection 长，粒度不匹配
- **embedding 模型换 BGE-large-en-v1.5（1024维）** 比 MiniLM（384维）提升 +0.013（SubsecRouted: 0.728→0.741）
- SubsecRoutedPC 成功的原因：LQE 对 Penalty 类 query 用 Act 词汇重写后搜句子层，词汇对齐+粒度细化双重增益

### SubsecRoutedPC→Rnk@5 实现细节
路由 + parent-child 的结合方式（`eval/langfuse_experiment.py`）：
- **Penalty** → `expand_query()` 扩展词汇 → search subsec_child_store → 取 subsection 父级
- **Prohibition** → `generate_stepback_query()` 抽象化 → search subsec_child_store（原+抽象各一次合并）
- **CrossSection** → `generate_query_variants()` 3 个变体 → search subsec_child_store（4次合并去重）
- **Other** → `generate_hypothetical_doc()` HyDE → search subsec_child_store
- 所有分支最后：`get_parents_by_ids()` 取 subsection 父级 → `reranker.rerank()` top-5

### 全实验排行榜（Grounded-79，R@5，截至 2026-09-16）

| Rank | 方法 | R@5 | LLM调用/query | 备注 |
|------|------|-----|--------------|------|
| **1** | **SubsecRoutedPC→LLMRnk@5** | **0.802** | 2次 | 路由+MiniLM top10→LLM重排→top5；12/79次fallback |
| 2 | SubsecRoutedPC→Rnk@10 | 0.797 | 1次 | 返回10条，非@5 |
| 3 | SubsecRoutedPC→Rnk@5 | 0.753 | 1次 | 首个破0.75 |
| 4 | SubsecPC→Rnk@5 | 0.750 | 0次 | 零LLM成本 |
| 5 | BGE Embed SubsecRouted | 0.741 | 1次 | |
| 6 | Cascade MiniLM→BGE | 0.728 | 1次 | 比单MiniLM差 |
| 7 | bge-reranker-large | 0.709 | 1次 | 比MiniLM差 |

### 剩余失败原因（仍有 ~24.7% 失败）
- Penalty（~4条）：词汇扩展仍有未覆盖到的同义词
- Prohibition（~4条）：语义方向相反，embedding 距离天然远
- CrossSection（~4条）：答案跨多节，单次检索无法覆盖
- Timeframe（~3条）：细节只在 section 某一句

### 进一步突破方向
1. **GraphRAG**：建节-节引用关系图，CrossSection 类沿边扩展
2. **法律专用 embedding 模型**（voyage-law-2）：解决 Prohibition 语义距离
3. **Answer-layer 修复**：top-5 全传 LLM，让 LLM 判断哪条有答案（解决 CrossSection）

---

## M10 — Autonomous ML Experimentation（autoresearch 模式）

**参考项目**：[karpathy/autoresearch](https://github.com/karpathy/autoresearch)（2026-03）
**应用对象**：`/Users/alvinwang/Documents/AI Development/aml-triage-demo/` — XGBoost ML alert 模型
**Why:** AML demo 的 XGBoost 目前"good enough"但未深度优化。autoresearch 模式让 AI agent 自主过夜跑实验，每轮改一个变量 → 训练 → 检查指标 → 保留或回滚，人类早上看结果，不用手动调参。

### autoresearch 核心思想（Karpathy）

| 原版（LLM训练） | AML XGBoost 对应 |
|---|---|
| `train.py` — agent 唯一可改的文件 | `train_xgboost.py` + `features.py` — agent 可改文件 |
| `prepare.py` — 固定，不可改 | 数据加载/预处理脚本 — 固定 |
| `program.md` — agent 指令，人类维护 | `program_xgb.md` — 定义搜索方向和约束 |
| 固定时间预算（5分钟/实验） | 固定评测预算（XGBoost秒级，按实验次数而非时间） |
| 指标：`val_bpb`（越低越好） | 指标：`recall@precision≥0.3`（尽量不漏报，同时精度不太低）|
| 接受/拒绝规则：指标改善则保留 | 接受/拒绝规则：目标指标提升 ≥ 0.5% 才保留 |

### agent 可探索的方向（写入 program_xgb.md）

**特征工程**（当前已有：per-day graph degree features）：
- 时间窗口聚合（7天/30天 交易频率、金额统计）
- 对手方多样性（unique counterparties / 14天）
- 金额分布特征（z-score、odd-amount flag、round-number flag）
- 循环转账特征（2-hop、3-hop cycle indicator from transaction graph）
- 速度特征（avg time between transactions）

**超参数**：`max_depth`, `n_estimators`, `learning_rate`, `scale_pos_weight`（类别不平衡），`subsample`, `colsample_bytree`

**阈值选择**：ROC 上选 recall≥0.70 时的最优 precision 点（当前 AML 场景：宁可误报，不能漏报）

**类别不平衡处理**：SMOTE vs `scale_pos_weight` vs threshold moving

### 实现设计

```
aml-triage-demo/
  autoresearch/
    program_xgb.md      ← 人类维护的 agent 指令（类似 Karpathy 的 program.md）
    train_xgboost.py    ← agent 可改（特征工程 + 超参数）
    evaluate.py         ← 固定不可改（评测逻辑 + 指标计算）
    experiment_log.md   ← agent 自动追加实验结果
    baseline/           ← 保存 baseline 权重，失败时回滚
```

**关键设计原则（来自 autoresearch）**：
1. **只改一个文件** — agent 只改 `train_xgboost.py`，diff 人类可 review
2. **固定评测预算** — 每个实验跑完整 CV，不能只跑一个 fold 就判断
3. **程序化接受/拒绝** — 指标提升 ≥ 0.5% 才保留，避免统计噪声
4. **保留实验日志** — `experiment_log.md` 记录每次改动 + 指标，早上回来看 narrative

### 评测指标设计

AML 场景特殊性：漏报（FN）比误报（FP）代价高得多
- **主指标**：`Recall@Precision≥0.30`（在精度至少30%时的最大召回率）
- **次指标**：`F1` at threshold 0.5
- **成本指标**：alerts per 1000 transactions（过多误报让人工审核不堪重负）

### 和 M3.0 的关系

M3.0 是"手写 agent harness"；M10 是"把那个 harness 用起来解决真实问题"。M10 的 agent loop 就是 M3.0 的 `run_coding_agent_loop()` + 加了：
- 实验前保存 baseline checkpoint
- 评测脚本自动运行并返回结果
- accept/reject 决策逻辑
- 实验日志追加

### 进度
- [ ] M10 — autoresearch 模式 XGBoost 自主实验

---

## 配置记录
- **Z.ai 模型**：从 glm-5.2 升级为 glm-5.3-flash（2026-09-16）
- **Langfuse 数据集**：`NSW-HealthAct-SimulatedQueries-Grounded-77`（79条）

---

## 进度

- [x] M1 — LangGraph 核心（StateGraph, Node, add_edge, add_conditional_edges）
- [x] M2 — LangGraph 进阶（Send API fan-out, MemorySaver checkpointing, stream()）
- [ ] M3.0 — Agent Harness 从零手写（ref: mihaileric.com/The-Emperor-Has-No-Clothes）
- [ ] M3 — ReAct 原理
- [ ] M4 — ReAct 全流程
- [ ] M5 — Eval 框架（M5.1✅ M5.2✅ M5.3 Langfuse Experiments✅ Layer3监控模拟待做 M5.4待M3/M4）
- [ ] M6 — 迭代优化
- [x] M7.1 — 基础 RAG baseline（chunker + ChromaDB + MiniLM embedding）
- [x] M7.2 — Hybrid Search（Dense + BM25 → RRF；发现 unigram BM25 在法律文本噪音大）
- [x] M7.3 — Reranking（Cross-Encoder 两阶段检索；s.140 从 rank14 救到 rank1）
- [x] M7.4 — Parent-Child Chunking（sentence 级检索 → section 级生成；s.99 被召回）
- [x] M7.5 — HyDE（GLM-5.2 生成假设条文再 embed；桥接词汇差距，Q4 证明内容缺口无法弥补）
- [x] M7.6 补充 — Multi-Query Retrieval（3轮实验；prompt约束+per_query_k=20为最佳配置）
- [x] M7.7 补充 — Step-back Prompting（实现并测试；per_query_k=20下无增量，原因是候选池已够宽）
- [x] M7.8 — EnsembleRetriever（Dense+HyDE+MultiQuery+StepBack；R@3=0.865，低于 HyDE 单技术）
- [x] M7.9 — Query 层改进（LegalHyDE/LQE/QueryRouting，均未超越 HyDE 基线，结构性失败）
- [x] M7.10 — Subsection Chunking（SubsecPC R@5=0.75；SubsecRoutedPC R@5=0.753 首次突破天花板）
- [x] M7.11 — BGE-large-en-v1.5 embedding 模型替换（MiniLM 384维 → BGE 1024维；SubsecRouted +0.013）
- [x] M7.12 — SubsecRoutedPC→Rnk@5（路由+PC+rerank三重叠加，R@5=0.753，当前最优）
- [x] M7.13 — SubsecRoutedPC→LLMRnk@5（MiniLM top10 → LLM 9条规则重排 → top5，R@5=0.802，首次破0.80；12/79次JSON解析error fallback；文件：src/rag/llm_reranker.py）
- [ ] M10 — Autonomous ML Experimentation（autoresearch 模式，应用于 AML XGBoost）
- [ ] M7.6 — Precision vs Recall 判断框架（理论已讲，待整理）
- [x] M5.1 — Golden Set 构造（GoldenSet_v2.xlsx，21个query，4种类型：Exact/Multi/Cross/Ambiguous）
- [x] M5.2 — eval/run_eval.py（多种技术对比，Recall@3/5 + Precision@3/5）
- [x] M5.3 — Langfuse Experiments（所有技术跑通，Total Cost 显示，HyDE→Reranked@3 最优）
- [ ] M5.3 Layer3 — 生产监控模拟（simulate_queries.py + Health Admin Act 1982 新增文件）
- [ ] M5.4 — AgentOps（LangSmith，待 M3/M4 完成后）

**How to apply:** 每次对话开始时检查当前模块进度，从上次停下来的地方继续，手把手写代码+解释。
