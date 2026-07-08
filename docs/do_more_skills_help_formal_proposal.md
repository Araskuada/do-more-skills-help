# Do More Skills Help? 前期调研与实验规划

**研究题目**：Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents  
**中文题目**：大量 Skills 是否真的有用？LLM Agent Skill Library 的规模化研究  
**项目周期**：1 个月  
**小组规模**：5 人  
**最终产出**：前期汇报、实验结果、研究海报

---

## Executive Summary

本项目研究一个很实际的问题：当 LLM Agent 的 skill library 不断变大时，Agent 是否真的会变得更强？我们的核心观点是：

> Skills can expand the capability ceiling of LLM agents, but under finite retrieval and context budgets, larger skill libraries may introduce retrieval noise, skill competition, and context pollution.

因此，本项目不把重点放在“skills 是否有用”这个泛问题上，而是聚焦一个更可测、可复现、适合一个月完成的方向：

**skill library 变大后，retrieval 是否更容易选错 skill？**

首要实验建议基于 **SkillRet** 做 skill retrieval scaling：固定 query 和 gold skill，逐步扩大 candidate skill pool，并比较 random distractor、same-category distractor、same-subcategory distractor 和 semantic-near distractor 对 Top-1 Accuracy、Recall@K、MRR、NDCG@10 的影响。补充实验可以用 **SkillsBench** 或小规模 agent task 验证 retrieval 错误是否会进一步影响 downstream task performance。

---

## 1. 研究背景与动机

### 1.1 Skill Library 是什么

在 LLM Agent 中，**skill** 通常指可以被复用的任务能力单元。它可能是：

- 一段可执行代码；
- 一个工具调用流程；
- 一个网页操作脚本；
- 一个领域任务指南；
- 一段 procedural knowledge；
- 一个带有输入、输出、约束和验证方式的操作模板。

**skill library** 则是这些 skills 的集合。Agent 在执行任务时，可以从 skill library 中检索、加载并复用相关 skills，从而减少重复探索，提高任务完成率。

典型例子包括：

- Voyager 在 Minecraft 中持续生成、保存和复用 executable skills；
- SkillWeaver 在 web agent 场景中自动发现、练习和封装 reusable APIs；
- 工具调用型 Agent 会从 tool library 中选择 API、函数或插件完成任务。

### 1.2 为什么 Agent 需要 Skills

LLM 本身擅长语言理解和推理，但在复杂任务中仍有明显限制：

- 缺少稳定的 procedural memory；
- 对长任务流程容易遗忘或走偏；
- 对特定环境、网页、软件仓库或 API 的操作经验有限；
- 每次遇到类似任务都要重新规划，成本高且不稳定；
- 对工具参数、调用顺序、前置条件和失败恢复机制掌握不牢。

Skills 可以作为外部能力扩展层，让 Agent：

- 复用过去成功经验；
- 把长流程压缩成可调用能力；
- 降低 token cost 和交互步数；
- 提升特定任务上的成功率；
- 在不同模型或不同 Agent 之间迁移能力。

### 1.3 现有工作中的隐含假设

许多已有 agent-skill 工作默认 skill set 是：

- 小规模的；
- 与当前任务高度相关的；
- 人工筛选或专家构造的；
- 描述清楚、边界明确的；
- 不存在大量冗余、过期或相似冲突的。

但真实场景中，skill library 很可能不断膨胀：

- 用户、开发者和 Agent 自己都会持续添加 skills；
- 不同任务产生的 skills 可能高度相似；
- 旧 skills 可能过期；
- skill metadata 可能不完整；
- skill 描述可能宽泛、含糊或互相重叠；
- library 中可能混入低质量、重复或不安全的 skills。

因此，**more skills** 并不必然等于 **better agent**。当 skill library 规模变大时，Agent 面临的核心问题会从“有没有 skill”转向“能不能找到并正确使用合适的 skill”。

### 1.4 中心论点

本项目的中心论点是：

> Skills can expand the capability ceiling of LLM agents, but under finite retrieval and context budgets, larger skill libraries may introduce retrieval noise, skill competition, and context pollution.

换句话说，大规模 skill library 的收益受限于三个预算：

- **retrieval budget**：检索器能否从大量候选 skill 中找出正确 skill；
- **context budget**：Agent 能否在有限上下文中理解并使用少量相关 skill；
- **decision budget**：LLM 能否判断何时应该加载 skill，以及加载哪个 skill。

---

## 2. 核心研究问题

本项目建议围绕以下 Research Questions 展开。

### RQ1: Skill library size 增大时，skill retrieval accuracy 是否下降？

研究 candidate pool 从 10、50、100、500、1000、5000、10000 到 full library 时，Top-1 Accuracy、Recall@K、MRR、NDCG@10 是否下降。

### RQ2: 哪种 distractor 最容易导致 Agent 选错 skill？

比较四类 distractor：

- random distractor；
- same-category distractor；
- same-subcategory distractor；
- semantic-near distractor。

重点观察 semantic-near distractor 是否比 random distractor 更容易造成错误检索。

### RQ3: 不同 retriever 在大规模 skill library 下表现有何差异？

比较：

- BM25；
- dense embedding retriever，例如 BGE 或 E5；
- hybrid retriever；
- reranker；
- 可选 LLM chooser。

目标是分析不同检索方法的准确率、稳定性、延迟和成本。

### RQ4: 检索到正确 skill 是否一定能提升 downstream task performance？

即使 gold skill 出现在 top-K 中，Agent 也可能：

- 没有加载它；
- 加载了错误 skill；
- 同时加载多个冲突 skill；
- 被 skill description 误导；
- 因上下文过长而忽略关键内容。

因此需要小规模 downstream validation。

### RQ5: 暴露更多 skill description 是否会导致 context pollution？

研究 top-K skills 越多时，LLM chooser 或 Agent 的表现是否持续提升，还是出现：

- token cost 增加；
- wrong skill invocation 增加；
- skill overloading；
- skill underloading；
- task pass rate 下降。

---

## 3. 相关文献综述

### 3.1 Skill-Based Agents

#### Voyager: An Open-Ended Embodied Agent with Large Language Models

链接：https://arxiv.org/abs/2305.16291

Voyager 是 skill library 在 LLM Agent 中的经典案例。它在 Minecraft 环境中通过 automatic curriculum 发现任务，通过 iterative prompting 生成代码，并把成功的 executable code 存入 skill library。之后 Agent 可以检索和复用这些 skills 来解决更复杂的任务。

与本项目关系：

- 证明 skill library 是 Agent 能力积累的重要机制；
- 但 Voyager 更关注 skill 生成和复用，没有系统研究 library 变大后的 retrieval degradation；
- 可作为本项目的背景动机。

#### SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills

链接：https://arxiv.org/abs/2504.07079

SkillWeaver 面向 web agents，让 Agent 在网站中自主发现技能、练习技能，并把成功经验蒸馏成 reusable APIs。论文报告其在 WebArena 和真实网站上带来相对成功率提升。

与本项目关系：

- 说明 skills 可以在 web agent 中自动积累；
- library 持续增长后会自然带来检索、组织和复用问题；
- 可支持“大规模、自动生成 skill library 会越来越常见”的动机。

#### SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

链接：https://arxiv.org/abs/2602.12670

SkillsBench 系统评估 agent skills 在多样任务中的效果。该 benchmark 包含多个 domains 和 trajectories，用于比较 no skill、curated skill 和 self-generated skill 等设置。论文发现 curated skills 平均有帮助，但不同任务差异很大，并且部分任务会出现负增益。

与本项目关系：

- 支持“skills 有用但收益不稳定”；
- 可作为 downstream validation 的候选数据集；
- 适合用于比较 no skill、gold skill、retrieved skill 和 noisy retrieved skill。

### 3.2 Skill Retrieval / Skill Library Benchmarks

#### SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents

链接：https://arxiv.org/abs/2605.05726

SkillRet 是目前最直接适配本项目的 benchmark。论文构建了大规模 public agent skills、evaluation queries 和 qrels，并按 category/subcategory 组织 skill corpus。

已确认信息：

- 17,810 个 public agent skills；
- 63,259 个 training samples；
- 4,997 个 evaluation queries；
- 6 个 major categories 和 18 个 sub-categories；
- train/eval skill pools disjoint。

与本项目关系：

- 可作为主实验数据集；
- 有 skill library、query 和 qrels，天然适合做 retrieval scaling；
- 可直接构造不同大小的 candidate pools；
- 可做 random distractor、same-category distractor、same-subcategory distractor 和 semantic-near distractor。

#### SRA-Bench / Skill Retrieval Augmentation for Agentic AI

链接：https://arxiv.org/abs/2604.24594

Skill Retrieval Augmentation for Agentic AI 提出 SRA 范式，将 Agent 使用 skills 的过程拆成 retrieval、incorporation 和 end-task execution。论文引入 SRA-Bench，用于分阶段评估 skill retrieval、skill incorporation 和最终任务执行。

已确认信息：

- 5,400 capability-intensive test instances；
- 636 manually constructed gold skills；
- 与 web-collected distractor skills 混合形成 26,262 skills 的大规模 corpus。

与本项目关系：

- 很适合支撑本项目理论框架；
- 可以把问题拆成三个阶段：retrieval 是否找对、LLM 是否加载、加载后是否完成任务；
- 若数据可用，可作为选做实验或补充分析。

#### How Well Do Agentic Skills Work in the Wild

链接：https://arxiv.org/abs/2604.04323

该工作研究 realistic settings 下 agentic skills 的实际效果。论文指出，很多已有 skill benchmark 太理想化：Agent 直接获得手工构造、任务高度匹配的 skills。但真实场景中，Agent 需要从大规模 skill collection 中检索相关 skills，而且检索到的 skills 未必完全适配任务。

已确认信息：

- 涉及 34k real-world skills；
- 研究 realistic settings 下 skill utility 退化；
- 发现 skill gains 在更现实条件下会逐步下降，最困难设置中 pass rate 接近 no-skill baseline；
- query-specific refinement 可以部分恢复性能。

与本项目关系：

- 强力支持“skills 在真实环境中效果会退化”；
- 支持本项目从 ideal skill injection 转向 retrieval + noisy skill setting。

#### SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software Ecosystems

链接：https://arxiv.org/abs/2605.13716

SkillOps 把 skill library 看作需要维护的软件生态系统，提出 skill technical debt 的概念。随着 skills 不断增加、复用、修补和依赖外部环境，library 会积累冗余、兼容性问题、风险和验证缺失。

与本项目关系：

- 支持“大型 skill library 会产生 technical debt”；
- 可作为本项目讨论 library maintenance、deduplication、validation 的理论依据；
- 与我们关注的 retrieval noise、skill conflict、stale skills 高度相关。

### 3.3 Tool Retrieval / Tool Menu 相关工作

虽然 tools 和 skills 不完全等价，但二者在 Agent 系统中都面临类似问题：候选集合很大，而 Agent 需要在有限上下文中选择少量相关能力。

#### ToolRet: Retrieval Models Aren't Tool-Savvy

链接：https://arxiv.org/abs/2503.01763

ToolRet 研究大规模 tool retrieval，指出很多 tool-use benchmark 会预先给定少量相关 tools，这不符合真实场景。论文构建了大规模 tool retrieval benchmark，并发现普通 IR 模型在 tool retrieval 上并不可靠。

已确认信息：

- 7.6K retrieval tasks；
- 43K tools；
- 提供 200K+ training instances。

与本项目关系：

- 可作为 SkillRet 不可用时的 proxy experiment；
- 支持“大规模 tool/skill collection 下 retrieval 是核心瓶颈”；
- 指标和实验设计可直接借鉴。

#### ToolMenuBench: Benchmarking Tool-Menu Filtering Strategies for Reliable and Efficient LLM Agents

链接：https://arxiv.org/abs/2606.15508

ToolMenuBench 研究 tool menu construction 对 Agent 可靠性、效率和风险暴露的影响。它系统变化 tool-menu size、distractor type 和 filtering methods，并报告 visible-tool count、wrong-tool calls、premature actions、token usage 等指标。

与本项目关系：

- 与本项目的 context pollution 和 controlled exposure 高度相关；
- 支持“把所有 tools/skills 都暴露给 Agent 并不可靠”；
- 可借鉴 visible skill count、wrong call rate、token usage 等指标。

#### PORTS: Preference-Optimized Retrievers for Tool Selection

检索状态：**需要进一步核实**。本次检索未找到可靠来源，因此不把它作为核心证据。若后续确认论文或项目存在，可纳入 tool retriever optimization 相关工作。

#### ToolBench / ToolLLM

链接：https://arxiv.org/abs/2307.16789

ToolLLM / ToolBench 收集大规模 real-world APIs，并训练 ToolLLaMA 学会调用工具。它将真实 API 文档组织成 benchmark，并使用 neural API retriever 推荐相关 API。

已确认信息：

- 16,464 个 real-world RESTful APIs；
- 覆盖 49 个 categories。

与本项目关系：

- 可作为下游 tool-use 场景；
- 支持 retrieval + tool use pipeline；
- 但原始在线 API 可能不稳定，若做实验建议优先考虑 StableToolBench。

#### API-Bank

链接：https://arxiv.org/abs/2304.08244

API-Bank 是 tool-augmented LLM benchmark，包含可运行 APIs、tool-use dialogues 和 API calls。

已确认信息：

- 73 个 runnable APIs；
- 314 个 tool-use dialogues；
- 753 API calls；
- 1,888 training dialogues from 2,138 APIs。

与本项目关系：

- 可作为小规模 tool-use sanity check；
- 不适合作为主数据集，因为其规模和 skill retrieval scaling 主题不完全匹配。

#### Gorilla / APIBench

链接：https://arxiv.org/abs/2305.15334

Gorilla 研究如何让 LLM 更好地连接 massive APIs，并通过 APIBench 评估 API 调用能力。其核心发现之一是 API 文档检索可以帮助减少 hallucination。

与本项目关系：

- 支持“检索正确外部能力比直接依赖参数记忆更可靠”；
- 可作为 tool retrieval 背景文献。

### 3.4 文献总结表

| 文献 | 研究对象 | 数据规模 / benchmark | 与我们项目的关系 | 可借用点 |
|---|---|---|---|---|
| Voyager | Minecraft Agent skills | Minecraft open-ended tasks，具体规模按论文核实 | 证明 skill library 可扩展 Agent 能力 | skill 生成、存储、复用机制 |
| SkillWeaver | Web agent skills | WebArena 和真实网站实验，具体样本需按论文核实 | 说明 web skills 可自动积累 | skill discovery、practice、API-style skills |
| SkillsBench | Agent skills | 多任务、多领域 trajectories，具体规模按论文核实 | 证明 skills 有帮助但不稳定 | no skill / curated skill / self-generated skill 对照 |
| SkillRet | Skill retrieval | 17,810 skills，63,259 train samples，4,997 eval queries | 主实验首选 | library scaling、qrels、category distractors |
| SRA-Bench | Skill retrieval + incorporation + execution | 5,400 test instances，636 gold skills，26,262 skills corpus | 支撑三阶段框架 | retrieval、incorporation、execution 分解评估 |
| How Well Do Agentic Skills Work in the Wild | Realistic skill usage | 34k real-world skills | 支持真实环境中 skill 效果退化 | realistic setting、skill refinement |
| SkillOps | Skill library maintenance | ALFWorld 等，具体实验规模需核实 | 支持 technical debt 论点 | skill contracts、library health、validation |
| ToolRet | Tool retrieval | 7.6K retrieval tasks，43K tools | SkillRet 备选 / 类比实验 | 大规模 retrieval、tool-skill proxy |
| ToolMenuBench | Tool menu filtering | 具体配置需按论文核实 | 支持 controlled exposure | visible tool count、wrong calls、token usage |
| PORTS | Tool selection retriever | 需进一步核实 | 暂不作为核心证据 | 若确认，可借鉴 preference optimization |
| ToolBench / ToolLLM | Tool use / API retrieval | 16,464 RESTful APIs，49 categories | downstream tool-use 扩展 | API retriever、tool-use pipeline |
| API-Bank | Tool-augmented LLMs | 73 runnable APIs，314 dialogues，753 calls | 小规模 sanity check | executable API evaluation |
| Gorilla / APIBench | Massive API calling | 具体 benchmark 规模需按论文核实 | 背景文献 | API retrieval reduces hallucination |

---

## 4. 可用数据集和 Benchmark

### 4.1 数据集优先级表

| 优先级 | 数据集 | 用途 | 优点 | 风险 / 成本 |
|---|---|---|---|---|
| 必做 P0 | SkillRet | 主实验，skill retrieval scaling | 有 skill library、query、qrels、category/subcategory，天然适合研究 library size 增大后 retrieval 是否变差 | 新数据集，需确认下载方式、license、字段格式 |
| 必做 P1 | SkillsBench | 小规模 downstream validation | 可比较 no skill、gold skill、retrieved skill、noisy retrieved skill 对任务完成率的影响 | 不一定天然支持大规模 retrieval，需要改造 |
| 选做 P1 | SRA-Bench | 分析 retrieval、incorporation、execution 三阶段 | 与本项目理论框架高度一致 | 数据和代码可用性需确认；可能较新 |
| 选做 P1 | ToolRet | tool retrieval 类比实验 | 规模大，适合做 library scaling proxy | tools 不等于 skills，需在报告中说明 |
| 选做 P2 | StableToolBench / ToolBench | tool-use downstream validation | 可观察 retrieval 错误对 task success 的影响 | 环境搭建和评价成本较高 |
| 选做 P2 | API-Bank | 小规模 executable API sanity check | 可运行 API，有 dialogue 和 call 记录 | 规模较小，不适合作为主实验 |
| 选做 P3 | Gorilla / APIBench | 背景或补充 tool retrieval | 支持 API retrieval 动机 | 与 skill library scaling 问题间接相关 |

### 4.2 推荐使用方式

首选路线：

1. 用 SkillRet 做 retrieval scaling 主实验；
2. 用 SkillsBench 做小规模 downstream validation；
3. 若 SkillRet 数据暂时不可用，用 ToolRet 做 proxy；
4. 若时间充足，用 SRA-Bench 支撑 retrieval -> incorporation -> execution 的完整链条。

---

## 5. 首要实验设计

首要实验：

> Skill library 变大后，retrieval 是否更容易选错 skill？

### 5.1 实验假设

H1: Library size 越大，Top-1 Accuracy、Recall@K、MRR、NDCG@K 会下降。

H2: 加入 semantic-near distractors 时，性能下降会比 random distractors 更明显。

H3: 即使 gold skill 出现在 top-K 中，LLM chooser 仍可能因为相似 skill 冲突而选错。

H4: 给 LLM 暴露更多 skill description 不一定提升表现，可能增加 token cost 和 context pollution。

### 5.2 数据构造

基于 SkillRet 构造不同大小的 candidate pools。

Library size：

- 10；
- 50；
- 100；
- 500；
- 1000；
- 5000；
- 10000；
- full library。

每个 pool 必须包含 gold skill，其余部分填充 distractors。每种设置建议使用 3 个 seeds，若时间允许使用 5 个 seeds。

#### Distractor 设计

1. **random distractor**

   随机抽取非 gold skills。用于模拟普通 library 扩张带来的背景噪声。

2. **same-category distractor**

   抽取与 gold skill 属于同一 major category 的 skills。用于模拟领域内相似能力造成的竞争。

3. **same-subcategory distractor**

   抽取与 gold skill 属于同一 subcategory 的 skills。比 same-category 更难，适合测试细粒度 skill 冲突。

4. **semantic-near distractor**

   使用 query embedding 或 gold skill embedding 找到语义最相近但不是 gold 的 skills。用于构造 hard negatives。

如果 category 信息缺失，可用 embedding clustering 或 skill metadata 自动构造 pseudo-category。

### 5.3 检索方法

建议比较以下方法。

#### BM25

优点：

- 实现简单；
- 计算成本低；
- 关键词明确时表现稳定；
- 适合作为 baseline。

风险：

- 对语义改写和隐式需求泛化较弱。

#### Dense Embedding Retriever

可选模型：

- BGE 系列；
- E5 系列；
- 其他开源 sentence embedding 模型。

优点：

- 语义匹配能力更强；
- 对 query wording 更鲁棒。

风险：

- 可能被 semantic-near distractors 误导；
- embedding 计算成本更高。

#### Reranker

例如 BGE reranker。

用法：

- 先用 BM25 / dense 召回 top-50 或 top-100；
- 再用 reranker 排序。

优点：

- 通常能提高 top rank accuracy；
- 适合处理相似候选。

风险：

- 速度慢；
- 成本高；
- 不适合直接 rerank full library。

#### Hybrid Retriever

结合 BM25 score 和 dense score：

```text
score = alpha * normalized_bm25 + (1 - alpha) * normalized_dense
```

优点：

- 兼顾 lexical matching 和 semantic matching；
- 可能比单一路线更稳。

风险：

- 需要调 alpha；
- score normalization 会影响结果。

#### LLM Chooser（选做）

给 LLM 展示 top-K skill descriptions，让它选择最相关 skill。

可测试：

- top-3；
- top-5；
- top-10；
- top-20。

重点观察：

- gold skill 在 top-K 中时，LLM 是否仍选错；
- K 增大是否导致 context pollution；
- token usage 和 latency 是否显著增加。

### 5.4 评价指标

#### Retrieval Metrics

- Top-1 Accuracy；
- Recall@1；
- Recall@3；
- Recall@5；
- Recall@10；
- MRR；
- NDCG@10；
- gold skill rank；
- score margin = score(best gold) - score(best wrong)。

#### Cost Metrics

- token usage；
- latency；
- number of visible skills；
- prompt length。

#### Error Analysis

- wrong skill category；
- wrong but same-category；
- wrong but semantically similar；
- near-duplicate conflict；
- overly broad skill selected；
- gold skill missing from top-K；
- gold skill retrieved but LLM selected wrong skill。

### 5.5 图表设计

最终汇报和海报建议展示：

1. **library size vs Top-1 Accuracy**

   展示不同 library size 下检索正确率是否下降。

2. **library size vs Recall@K**

   展示 gold skill 是否仍能进入 top-K。

3. **library size vs MRR**

   展示 gold skill 排名是否整体后移。

4. **distractor type × library size heatmap**

   展示不同 distractor 设置下的性能退化强度。

5. **retriever comparison bar chart**

   比较 BM25、dense、hybrid、reranker。

6. **token cost vs accuracy**

   展示 LLM chooser 或 reranker 的收益是否值得成本。

7. **case study table**

   展示错误检索样例，包括 query、gold skill、wrong retrieved skill、错误原因。

---

## 6. Downstream 验证实验

Downstream 实验不是本项目主实验，而是用来证明 retrieval scaling 的实际影响。

推荐使用 SkillsBench 或一个小规模可执行任务子集。

### 6.1 实验目标

验证：

> retrieval 错误是否真的会降低 agent task performance？

### 6.2 比较条件

1. **no skill**

   Agent 不使用任何 skill。

2. **gold skill**

   直接给 Agent 正确 skill，作为 oracle upper bound。

3. **retrieved top-1 skill**

   给 Agent retriever 排名第一的 skill。

4. **retrieved top-3 skills**

   给 Agent top-3 skill descriptions，让 Agent 自行选择或使用。

5. **noisy retrieved skills**

   给 Agent 包含 semantic-near wrong skill 的 top-K。

6. **all visible skills**

   只适用于小 library setting，用来测试过多 skill 暴露是否造成 context pollution。

### 6.3 评价指标

- task pass rate；
- wrong skill invocation rate；
- skill overloading rate；
- skill underloading rate；
- average token usage；
- average execution cost。

### 6.4 预期分析

需要比较：

- gold skill 是否显著优于 no skill；
- retrieved top-1 是否接近 gold skill；
- noisy retrieved skills 是否低于 clean retrieved skills；
- top-3/top-5 是否比 top-1 更好，还是因 context pollution 变差；
- all visible skills 是否成本最高且不一定最好。

---

## 7. 预期结果与可能结论

我们可能观察到以下结果：

- random distractor 下性能下降较慢；
- same-category 和 semantic-near distractor 下性能下降更明显；
- dense retriever 在语义匹配上更好，但可能被相似 skill 误导；
- BM25 在关键词明确时稳定，但泛化较弱；
- hybrid / reranker 可能表现最好，但计算成本更高；
- top-K recall 不一定等于 downstream success；
- library 越大，Agent 越需要 pruning、hierarchical indexing、reranking 和 controlled skill exposure。

最终可能结论：

> More skills increase potential capability, but raw library size alone does not guarantee better agent performance. Without effective retrieval, pruning, reranking, and validation, larger skill libraries may introduce more noise than benefit.

---

## 8. 一个月项目时间规划

成员分工：

- A：文献综述与理论框架负责人；
- B：数据与 candidate pool 构造负责人；
- C：retrieval baseline 与模型负责人；
- D：downstream agent 实验负责人；
- E：数据分析、可视化和报告整合负责人。

### Week 1: Literature Review and Data Pipeline

目标：完成文献调研，跑通 SkillRet 数据加载和 BM25 baseline。

| 成员 | 任务 |
|---|---|
| A | 精读 SkillRet、SkillRouter、SRA-Bench、SkillsBench，整理理论框架 |
| B | 确认 SkillRet 下载方式、schema、qrels、category 字段 |
| C | 实现 BM25 baseline 和基本 metrics |
| D | 调研 SkillsBench / SRA-Bench downstream 可运行性 |
| E | 建立结果表格式、海报初版结构、可视化模板 |

交付物：

- `literature_matrix.csv`；
- SkillRet data loader；
- BM25 baseline；
- 初步 scaling curve；
- poster outline。

### Week 2: Retrieval Scaling Experiments

目标：完成 library size × distractor type × retriever 的主实验。

| 成员 | 任务 |
|---|---|
| A | 检查实验设计是否和 RQs 对齐，补充 related work |
| B | 实现 candidate pool sampler，包括 random、same-category、same-subcategory、semantic-near |
| C | 实现 dense retriever、hybrid retriever，跑主实验 |
| D | 设计 LLM chooser prompt 和 downstream 条件 |
| E | 生成初步图表和 error case 表 |

交付物：

- `retrieval_scaling_results.jsonl`；
- `summary.csv`；
- 初步图表；
- error cases。

### Week 3: LLM Chooser and Downstream Validation

目标：测试 top-K skill selection 和小规模 downstream task performance。

| 成员 | 任务 |
|---|---|
| A | 写 methodology 和 expected findings |
| B | 整理 top-K skill exposure 数据 |
| C | 跑 reranker 或 top-50 reranking |
| D | 跑 SkillsBench 小规模条件：no skill、gold skill、retrieved skill、noisy skill |
| E | 整合 LLM chooser results、downstream results、case studies |

交付物：

- LLM chooser results；
- SkillsBench downstream results；
- case studies；
- updated plots。

### Week 4: Analysis, Writing, and Presentation

目标：完成最终报告、图表和汇报材料。

| 成员 | 任务 |
|---|---|
| A | 完成最终报告中的背景、问题和文献综述 |
| B | 完成数据集和实验设置说明 |
| C | 清理代码、固定随机种子、生成最终结果 |
| D | 完成 downstream validation 和 limitations |
| E | 完成最终海报、图表、README 和 presentation script |

交付物：

- `final_report.md`；
- `slides.md` 或 `slides.pptx` outline；
- `plots/`；
- `README` reproduction guide；
- final poster PDF。

---

## 9. 技术实现规划

建议项目结构：

```text
skill_scaling_study/
  README.md
  requirements.txt
  configs/
    skillret_scaling.yaml
  src/
    datasets.py
    pool_sampler.py
    retrievers/
      bm25.py
      dense.py
      hybrid.py
      reranker.py
    metrics.py
    run_retrieval_scaling.py
    analyze_results.py
    plot_results.py
  tests/
    test_metrics.py
    test_pool_sampler.py
  results/
    plots/
```

### 文件作用

| 文件 | 作用 |
|---|---|
| `README.md` | 项目说明、安装方式、数据准备、复现实验命令 |
| `requirements.txt` | Python 依赖，如 rank_bm25、sentence-transformers、pandas、numpy、matplotlib |
| `configs/skillret_scaling.yaml` | 实验配置，包括 pool sizes、distractor modes、retrievers、seeds |
| `src/datasets.py` | 加载 SkillRet / ToolRet / SkillsBench 数据，统一 query、skill、qrels 格式 |
| `src/pool_sampler.py` | 构造 candidate pools，支持 random、same-category、same-subcategory、semantic-near |
| `src/retrievers/bm25.py` | BM25 retriever |
| `src/retrievers/dense.py` | Dense embedding retriever |
| `src/retrievers/hybrid.py` | BM25 + dense hybrid score |
| `src/retrievers/reranker.py` | Cross-encoder reranker 或 BGE reranker |
| `src/metrics.py` | Top-1 Accuracy、Recall@K、MRR、NDCG@10、gold rank、score margin |
| `src/run_retrieval_scaling.py` | 主实验入口，输出 jsonl 结果 |
| `src/analyze_results.py` | 汇总 jsonl 为 summary.csv |
| `src/plot_results.py` | 生成曲线图、热力图、bar chart |
| `tests/test_metrics.py` | 测试指标实现是否正确 |
| `tests/test_pool_sampler.py` | 测试 candidate pool 是否包含 gold skill、distractor 是否符合要求 |
| `results/` | 保存实验结果 |
| `results/plots/` | 保存最终图表 |

### CLI 设计

运行 retrieval scaling：

```bash
python -m src.run_retrieval_scaling \
  --dataset skillret \
  --pool-sizes 10 50 100 500 1000 5000 10000 \
  --distractor-modes random same_category same_subcategory semantic_near \
  --retrievers bm25 dense hybrid \
  --seeds 0 1 2 \
  --out results/skillret_scaling.jsonl
```

分析结果：

```bash
python -m src.analyze_results \
  --input results/skillret_scaling.jsonl \
  --out results/summary.csv
```

绘图：

```bash
python -m src.plot_results \
  --summary results/summary.csv \
  --out-dir plots/
```

---

## 10. 风险与备选方案

| 风险 | 影响 | 备选方案 |
|---|---|---|
| SkillRet 数据字段和预期不一致 | candidate pool 或 category distractor 难以构造 | 先统一成 query、skill_id、skill_text、qrels 四列；category 缺失时用 embedding clustering 或 metadata 构造 pseudo-category |
| same-category 信息缺失 | 无法做 category distractor | 使用 semantic clustering、skill tags 或 LLM 自动标注小样本 |
| embedding 模型计算成本较高 | dense retriever 进度慢 | 先抽样 100-300 queries；缓存 embeddings；优先跑 BM25 |
| reranker 太慢 | 难以 full-library rerank | 只在 top-50 或 top-100 上 rerank |
| SkillsBench 环境难以运行 | downstream validation 延迟 | 只做 retrieval + LLM chooser；或做少量 qualitative case study |
| LLM API 成本过高 | LLM chooser 无法大规模跑 | 使用开源小模型；只抽样 100-300 queries；降低 top-K |
| 一个月内无法完成全部 downstream 实验 | 项目范围过大 | 优先完成 SkillRet retrieval scaling 主实验，把 downstream 作为附加实验或 future work |
| 结果不明显 | 海报主结论不够强 | 加强 hard distractor 设置；分 query difficulty、category、skill length 做细粒度分析 |

---

## 11. 最终可执行任务清单

### 必须完成

- 确认 SkillRet 或 ToolRet 数据可用；
- 实现 candidate pool sampler；
- 实现 BM25 baseline；
- 至少跑完 random distractor 和 semantic-near distractor；
- 输出 Top-1 Accuracy、Recall@K、MRR、NDCG@10；
- 生成 library size vs accuracy 曲线；
- 完成前期汇报和研究海报。

### 尽量完成

- dense retriever；
- hybrid retriever；
- reranker；
- field ablation；
- LLM chooser；
- SkillsBench 小规模 downstream validation；
- error case study。

### 可以作为 Future Work

- SRA-Bench 完整三阶段评估；
- SkillOps-style library maintenance；
- hierarchical skill routing；
- automatic skill deduplication；
- adaptive context budgeting。

---

## 12. 海报建议结构

海报可以采用三栏结构。

### 左栏：Motivation and Prior Evidence

内容：

- skill library 为什么重要；
- more skills 不一定 better；
- 文献证据表：SkillRet、SRA-Bench、SkillsBench、SkillOps、ToolRet。

### 中栏：Method

内容：

```text
Query
  -> Candidate Skill Pool
  -> Retriever / Reranker
  -> Top-K Skills
  -> Metrics + Error Analysis
```

展示变量：

- library size；
- distractor type；
- retriever type；
- top-K visible skills。

### 右栏：Results and Takeaways

内容：

- library size vs Top-1 Accuracy；
- random vs semantic-near distractor；
- retriever comparison；
- error case study；
- final takeaways。

建议 takeaway：

1. More skills increase potential capability, but also increase retrieval burden.
2. Semantic-near distractors are more damaging than random distractors.
3. Controlled exposure, reranking, pruning, and validation are necessary for scalable skill libraries.

---

## References

1. Voyager: An Open-Ended Embodied Agent with Large Language Models. https://arxiv.org/abs/2305.16291
2. SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills. https://arxiv.org/abs/2504.07079
3. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. https://arxiv.org/abs/2602.12670
4. SkillRet: A Large-Scale Benchmark for Skill Retrieval in LLM Agents. https://arxiv.org/abs/2605.05726
5. Skill Retrieval Augmentation for Agentic AI. https://arxiv.org/abs/2604.24594
6. How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings. https://arxiv.org/abs/2604.04323
7. SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software Ecosystems. https://arxiv.org/abs/2605.13716
8. Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models. https://arxiv.org/abs/2503.01763
9. ToolMenuBench: Benchmarking Tool-Menu Filtering Strategies for Reliable and Efficient LLM Agents. https://arxiv.org/abs/2606.15508
10. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. https://arxiv.org/abs/2307.16789
11. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. https://arxiv.org/abs/2304.08244
12. Gorilla: Large Language Model Connected with Massive APIs. https://arxiv.org/abs/2305.15334
