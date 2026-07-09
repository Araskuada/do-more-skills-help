# RQ4 实验分析：Retrieval Correctness 是否足以支持 Downstream Task Completion

**日期**：2026-07-09  
**研究问题**：RQ4 - Does retrieving the correct skill always improve downstream task performance?  
**实验脚本**：`experiments/rq4_downstream_skill_exposure.py`  
**输出目录**：`data/experiments/rq4_downstream_skill_exposure/`

---

## 1. 结论摘要

这次 RQ4 先做一个可复现的 downstream skill exposure / readiness proxy，而不是直接跑 agent pass rate。

原因是：真正的 SkillsBench agent pass rate 需要外部模型 API key、BenchFlow runner 和 sandbox backend。当前实验先回答一个更基础的问题：

**不同 retrieval 条件会给 downstream agent 暴露多少必要 skill、多少错误 skill，以及是否形成 underload / overload？**

核心结论是：

1. **Top-1 correct 不等于 downstream-ready。**  
   BM25 Top-1 的 `top1_is_gold` 是 **0.747**，但完整覆盖任务所需全部 gold skills 的比例只有 **0.195**。

2. **Top-K 越大，gold coverage 越高，但噪声也快速增加。**  
   `hybrid_bm25_neural_top10` 的完整 gold coverage 最高，达到 **0.747**；但平均每个任务也会暴露 **7.68** 个额外 non-gold skills。

3. **多 skill composition 是 RQ4 的核心难点。**  
   SkillsBench 87 个默认任务平均需要 **2.67** 个 curated skills。只看 Top-1 retrieval 会系统性低估 downstream 所需 skill set。

4. **“给全量 skills”不是解决方案。**  
   `all_skills_visible` 覆盖率是 **1.000**，但平均额外暴露 **199.33** 个 non-gold skills，中位 context token 数约 **165,227**，几乎必然带来 context pollution。

5. **当前最好的 retrieval exposure 条件是 `hybrid_bm25_neural_top10`。**  
   它在完整 gold coverage、gold recall 和上下文 token 成本之间表现最好，但仍不是 clean downstream condition。

因此，RQ4 的当前答案是：

**检索到正确 skill 有帮助，但不一定足以提升 downstream task completion；downstream success 更依赖“完整且干净的 skill set exposure”，而不是单个 Top-1 是否正确。**

---

## 2. 实验边界

这不是实际 agent pass-rate 实验。

它测量的是 downstream task completion 的必要条件 proxy：

- agent 是否看到了至少一个 gold skill；
- agent 是否看到了任务所需的全部 gold skills；
- agent 是否只看到了 gold skills；
- agent 同时看到了多少 non-gold skills；
- skill context token load 有多大。

不能从本实验直接推出“agent 一定通过/失败”。但可以推出：

- 如果 gold skills 没有完整暴露，复杂任务的 downstream upper bound 会被 retrieval 限制；
- 如果 non-gold skills 大量暴露，即使 gold skills 在上下文中，也可能产生 context pollution 和 skill confusion。

---

## 3. 实验设置

### 数据

- 数据集：SkillsBench default tasks
- 任务数：87
- unique skill documents：202
- task gold skill references：232
- 平均每个任务 gold skill 数：2.67

每个任务的 `environment/skills/*/SKILL.md` 被视为该任务的 curated / gold skill set。

### 条件

| Condition | Definition |
|---|---|
| `no_skill` | 不给 agent 暴露任何 skill |
| `oracle_gold_all` | 暴露该任务全部 curated gold skills |
| `oracle_gold_plus_5_noise` | 暴露全部 gold skills，再随机加入 5 个 non-gold skills |
| `bm25_topK` | BM25 从全 SkillsBench skill pool 中检索 Top-K |
| `tfidf_topK` | TF-IDF cosine 检索 Top-K |
| `neural_minilm_topK` | MiniLM dense retriever 检索 Top-K |
| `hybrid_bm25_neural_topK` | BM25 + MiniLM reciprocal-rank fusion 检索 Top-K |
| `all_skills_visible` | 暴露全部 202 个 unique skill documents |

K 取值：1、3、5、10。

### 指标

| Metric | Meaning |
|---|---|
| `top1_is_gold` | Top-1 暴露 skill 是否属于 gold skill set |
| `any_gold_coverage` | 是否至少暴露一个 gold skill |
| `complete_gold_coverage` | 是否暴露全部 gold skills |
| `strict_skill_set_match` | 是否暴露全部 gold skills 且没有 non-gold skills |
| `gold_recall` | gold skills 被覆盖比例 |
| `skill_precision` | 暴露 skills 中 gold skills 的比例 |
| `underload` | 是否缺少至少一个 gold skill |
| `overload` | 是否暴露至少一个 non-gold skill |
| `extra_skill_count` | 平均额外 non-gold skill 数 |
| `context_tokens` | 暴露 skill 文本的 token 量 proxy |

---

## 4. 主要结果

### 4.1 Oracle 与极端条件

| Condition | Complete Coverage | Strict Match | Gold Recall | Precision | Extra Skills | Median Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `no_skill` | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| `oracle_gold_all` | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1,836 |
| `oracle_gold_plus_5_noise` | 1.000 | 0.000 | 1.000 | 0.324 | 5.00 | 5,657 |
| `all_skills_visible` | 1.000 | 0.000 | 1.000 | 0.013 | 199.33 | 165,227 |

这个对照说明：

- gold skills 完整暴露是必要条件，但不是唯一条件；
- 只要加入 5 个 noisy skills，precision 就从 1.000 降到 **0.324**；
- 暴露全部 skills 虽然 coverage 完美，但上下文污染极其严重。

### 4.2 Top-1 retrieval 不足以支持多 skill task

| Condition | Top-1 Is Gold | Complete Coverage | Gold Recall | Missing Gold Skills |
|---|---:|---:|---:|---:|
| `bm25_top1` | **0.747** | 0.195 | 0.379 | 1.92 |
| `tfidf_top1` | 0.736 | 0.184 | 0.369 | 1.94 |
| `hybrid_bm25_neural_top1` | 0.805 | 0.184 | 0.400 | 1.86 |
| `neural_minilm_top1` | 0.701 | 0.161 | 0.352 | 2.01 |

这是 RQ4 最重要的发现之一：

**即使 Top-1 是正确 skill，任务仍然常常缺少其他必要 skills。**

例如 `video-silence-remover` 需要 7 个 curated skills，但 Top-1 只能暴露其中 1 个。这种情况下 Top-1 correct 对 downstream completion 只提供部分帮助，不能保证任务可完成。

### 4.3 Top-K retrieval 提高 coverage，但带来 overload

| Condition | Complete Coverage | Gold Recall | Precision | Extra Skills | Median Tokens |
|---|---:|---:|---:|---:|---:|
| `bm25_top3` | 0.506 | 0.713 | 0.552 | 1.34 | 3,474 |
| `bm25_top5` | 0.575 | 0.794 | 0.393 | 3.03 | 6,041 |
| `bm25_top10` | 0.667 | 0.845 | 0.217 | 7.83 | 13,663 |
| `hybrid_bm25_neural_top3` | 0.448 | 0.685 | 0.548 | 1.36 | 2,897 |
| `hybrid_bm25_neural_top5` | 0.575 | 0.805 | 0.398 | 3.01 | 4,990 |
| `hybrid_bm25_neural_top10` | **0.747** | **0.899** | 0.232 | 7.68 | 10,525 |

Top-K 的 trade-off 很清楚：

- K 从 1 增加到 10，complete gold coverage 明显提升；
- 但 skill precision 明显下降；
- extra non-gold skills 和 context tokens 快速增加。

这支持 RQ5 的前置动机：暴露更多 skill description 可能引发 context pollution。

### 4.4 Retriever comparison under RQ4 proxy

Top-10 条件下：

| Retriever | Complete Coverage | Gold Recall | Precision | Extra Skills | Median Tokens |
|---|---:|---:|---:|---:|---:|
| `hybrid_bm25_neural_top10` | **0.747** | **0.899** | **0.232** | 7.68 | 10,525 |
| `neural_minilm_top10` | 0.713 | 0.866 | 0.229 | 7.71 | 7,481 |
| `tfidf_top10` | 0.678 | 0.833 | 0.208 | 7.92 | 9,841 |
| `bm25_top10` | 0.667 | 0.845 | 0.217 | 7.83 | 13,663 |

增强版 RQ3 的结论在 RQ4 proxy 下也基本成立：

- hybrid BM25+MiniLM 是最好的整体检索条件；
- MiniLM dense 的 context token load 更低；
- BM25 覆盖率不错，但 Top-10 上下文更长；
- 所有 Top-10 条件都有明显 overload。

---

## 5. Case Studies

### 5.1 Top-1 correct but incomplete

`video-silence-remover` 是最典型例子：

- gold skill count：7
- BM25 Top-1 暴露：`report-generator`
- missing skills：`audio-extractor`、`energy-calculator`、`segment-combiner`、`silence-detector`、`video-processor`、`pause-detector`

这说明 Top-1 correct 只是局部正确。对于 composition task，agent 仍然缺少完整 workflow 所需的技能模块。

### 5.2 Complete gold but noisy

`oracle_gold_plus_5_noise` 覆盖所有 gold skills，但 precision 只有 **0.324**。

这对应真实 agent 场景中的问题：

- gold skill 在上下文里；
- 但 agent 还要在多个 irrelevant skills 中选择；
- 如果模型误读、混用或优先执行错误 skill，downstream performance 仍可能下降。

### 5.3 All skills visible is not a useful upper bound

`all_skills_visible` 的 complete coverage 是 **1.000**，但 median context tokens 是 **165,227**，平均额外 non-gold skills 是 **199.33**。

这说明“把所有 skills 都给 agent”不能作为实际解决方案；它只证明 gold skill 存在于上下文，并不证明 agent 能有效使用它。

---

## 6. 对 RQ4 的回答

RQ4 问的是：检索到正确 skill 是否一定提升 downstream task performance？

当前实验给出的回答是：

**不一定。**

更准确地说：

1. 如果只检索 Top-1，即使 Top-1 是 gold，也常常无法覆盖多 skill task 的完整需求。
2. 如果扩大到 Top-K，完整覆盖率提升，但 non-gold skills 和 context tokens 快速增加。
3. 如果暴露全部 skills，coverage 满分，但 context pollution 极其严重。
4. 因此 downstream completion 更依赖“完整、干净、可执行的 skill set”，而不只是 retrieval Top-1 accuracy。

这也解释了为什么 RQ1-RQ3 的 retrieval metrics 很重要但还不够：retrieval 是 downstream success 的入口，但不是充分条件。

---

## 7. 当前限制

- 当前实验是 skill exposure / readiness proxy，不是实际 SkillsBench agent pass rate。
- 没有运行 verifier，因此不能报告 task success rate。
- gold skill set 来自 SkillsBench task-local curated skills，默认认为这些 skills 是任务所需集合。
- skill 去重基于完整 `SKILL.md` 内容 hash；语义等价但文本不同的 skill 仍会被视为不同 skill。
- 没有评估 agent 是否真的阅读、调用或遵循暴露的 skills。

---

## 8. 下一步

1. 选 5-10 个代表性 SkillsBench tasks，实际跑 agent 条件：`no_skill`、`oracle_gold_all`、`bm25_topK`、`hybrid_topK`、`gold_plus_noise`。
2. 对这些任务运行 verifier，报告真实 pass rate。
3. 对比 exposure proxy 与 pass rate，检查 proxy 是否预测 downstream success。
4. 进入 RQ5：系统改变 K 和 noisy skill 数量，观察 context pollution 是否进一步恶化。

---

## 9. Reproducibility

运行 RQ4 exposure proxy：

```bash
python3 experiments/rq4_downstream_skill_exposure.py
```

主要输出：

- `data/experiments/rq4_downstream_skill_exposure/summary.csv`
- `data/experiments/rq4_downstream_skill_exposure/summary.json`
- `data/experiments/rq4_downstream_skill_exposure/per_task_exposure.csv`
- `data/experiments/rq4_downstream_skill_exposure/case_studies.json`
