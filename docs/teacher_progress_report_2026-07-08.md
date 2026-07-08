# 项目进度汇报草稿

**项目题目**：Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents  
**汇报日期**：2026-07-08  
**当前阶段**：研究问题确定、数据集准备完成、第一版测试实验已跑通

---

## 1. 我们的研究问题

我们想研究的问题是：

> 当 LLM Agent 可以使用的 skill library 变得越来越大时，更多 skills 是否一定会让 Agent 表现更好？

我们的初步判断是：skills 可以提高 Agent 的能力上限，但在有限 retrieval budget 和 context budget 下，skill library 变大也会带来 retrieval noise、skill competition 和 context pollution。

因此，我们的完整研究框架围绕 proposal 中的五个 Research Questions 展开：

| RQ | 问题 | 我们要测什么 |
|---|---|---|
| RQ1 | Skill library size 增大时，skill retrieval accuracy 是否下降？ | candidate pool 从 10、50、100、500、1000、5000、10000 到 full library 时，Top-1 Accuracy、Recall@K、MRR、NDCG@10 是否下降。 |
| RQ2 | 哪种 distractor 最容易导致 Agent 选错 skill？ | 比较 random、same-category、same-subcategory 和 semantic-near distractor，观察哪类干扰项最容易挤掉 gold skill。 |
| RQ3 | 不同 retriever 在大规模 skill library 下表现有何差异？ | 比较 BM25、dense embedding retriever、hybrid retriever、reranker 和可选 LLM chooser 的准确率、稳定性、延迟和成本。 |
| RQ4 | 检索到正确 skill 是否一定能提升 downstream task performance？ | 在真实任务中比较 no skill、gold skill、retrieved skill 和 noisy retrieved skills 的 pass rate。 |
| RQ5 | 暴露更多 skill description 是否会导致 context pollution？ | 研究 top-K skills 增多时，是否出现 token cost 增加、wrong skill invocation、skill overloading、skill underloading 或 pass rate 下降。 |

这五个 RQ 是递进关系：RQ1 先确认“大库是否让检索变难”；RQ2 解释“为什么变难”；RQ3 比较“什么检索方法更稳”；RQ4 验证“检索质量是否影响真实任务”；RQ5 进一步研究“即使检索到多个相关 skills，给 Agent 看得越多是否反而污染上下文”。

---

## 2. 我们选择的数据集

目前已经下载并整理了三个本地数据源：

| 数据集 | 本地状态 | 规模 | 用途 |
|---|---|---:|---|
| Skill-Usage | 已下载 | 35,554 个 `SKILL.md`，87 个任务 | 主实验：大规模 skill retrieval scaling |
| SkillsBench | 已下载 | 87 个默认任务 | 后续 downstream validation |
| SWE-Skills-Bench | 已下载 | 49 个 skills | 软件工程场景补充分析 |

我们选择 **Skill-Usage** 作为当前主实验数据，原因是它已经包含：

- 大规模真实 skill pool；
- task query；
- task-to-skill gold mapping；
- skill metadata；
- 可复现的本地检索实验设置。

SkillsBench 和 SWE-Skills-Bench 更适合作为第二阶段实验，用来验证 retrieval error 是否会影响真实任务表现。

---

## 3. 已完成的测试实验

我们已经完成了一个 retrieval scaling pilot。这个测试实验主要对应 **RQ1**，也为后续 RQ2-RQ5 提供 baseline。

**实验设置**：

- 数据：`data/raw/Skill-Usage`
- 任务数：87
- Gold skill：`task_skill_mapping.json`
- Query：`task_queries.json`
- Candidate pool：从 10、50、100、500、1000、5000、10000 到 full library
- Distractor：random distractor
- Retriever：本地轻量 BM25，基于 skill name + description
- 指标：Top-1 Accuracy、Recall@3、Recall@5、Recall@10、MRR、NDCG@10
- 输出：`data/experiments/retrieval_scaling_pilot`

**初步结果**：

| Pool size | Top-1 | R@3 | R@5 | R@10 | MRR | NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.963 | 0.820 | 0.871 | 0.880 | 0.975 | 0.898 |
| 50 | 0.915 | 0.768 | 0.846 | 0.879 | 0.943 | 0.864 |
| 100 | 0.890 | 0.736 | 0.825 | 0.870 | 0.926 | 0.845 |
| 500 | 0.763 | 0.650 | 0.724 | 0.790 | 0.836 | 0.746 |
| 1000 | 0.736 | 0.607 | 0.688 | 0.750 | 0.802 | 0.703 |
| 5000 | 0.609 | 0.485 | 0.558 | 0.648 | 0.686 | 0.580 |
| 10000 | 0.531 | 0.433 | 0.482 | 0.582 | 0.615 | 0.511 |
| full | 0.414 | 0.359 | 0.401 | 0.449 | 0.507 | 0.406 |

**对 RQ1 的初步回答**：

- Candidate pool 从 10 扩大到 full 时，Top-1 Accuracy 从 0.963 降到 0.414，相对下降约 57%。
- Recall@10 从 0.880 降到 0.449，说明即使允许 top-10，full library 下仍有大量 gold skill 排不进前 10。
- 这支持我们的核心假设：skill library 规模增大后，检索噪声会显著增加，Agent 更难选到正确 skill。

---

## 4. 一个失败案例

任务 `court-form-filling` 的 query 是：

> PDF form filling court form automation

Gold skill 是：

> `benchflow-ai--pdf-1`

但在 full library 下，BM25 的 top results 更倾向于返回 browser automation、PDF processing、pypdf、pdf-tools 等相似但不完全正确的 skills，gold skill 没有进入 top-10。

这个案例说明，大规模 skill library 中会出现很多表面相关的 skills，它们会和真正需要的 skill 竞争排名。

---

## 5. 下一步计划

1. **RQ2**：扩展 distractor 类型，从 random distractor 扩展到 same-category、same-subcategory、lexical-overlap 和 semantic-near distractor。
2. **RQ3**：比较更多 retriever，包括 BM25、semantic embedding、hybrid retriever、reranker，以及可选 LLM chooser。
3. **RQ4**：做小规模 downstream validation，在 SkillsBench 中比较 no skill、gold skill、retrieved top-1、retrieved top-k 和 noisy skills 的任务通过率。
4. **RQ5**：控制暴露给 Agent 的 top-K skill descriptions 数量，观察 token cost、wrong skill invocation 和 pass rate 是否随 K 增大而恶化。
5. 做 qualitative error analysis：分析错误检索主要来自命名相似、描述重叠、skill 过泛化，还是 gold skill metadata 不充分。

---
