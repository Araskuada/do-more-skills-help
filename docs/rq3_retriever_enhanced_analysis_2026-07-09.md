# RQ3 增强实验分析：True Neural Retriever、Hybrid、Full SKILL.md 与 Hard Distractors

**日期**：2026-07-09  
**研究问题**：RQ3 - How do different retrievers behave under large-scale skill libraries?  
**增强实验脚本**：`experiments/rq3_retriever_enhanced.py`  
**输出目录**：`data/experiments/rq3_retriever_enhanced/`

---

## 1. 结论摘要

这次增强版 RQ3 直接回应了上一版分析里列出的主要限制：

- 加入了本地缓存的真实神经 dense retriever：`sentence-transformers/all-MiniLM-L6-v2`
- 加入了 neural + BM25 hybrid
- 加入了基于完整 `SKILL.md` 内容的 BM25 / TF-IDF 检索
- 在 RQ2 风格的 hard distractor 条件下测试 retriever robustness

增强实验的核心结论是：

1. **在 full library 下，最强方法变成了 `hybrid_bm25_neural`**，Top-1 Accuracy = **0.460**，高于原始 `bm25_desc` 的 **0.425**。
2. **纯神经 dense (`neural_minilm_desc`) 并没有在 full library 下单独超过 BM25**，但它与 BM25 融合后可以带来稳定收益。
3. **完整 `SKILL.md` 内容不是天然有害**。修正 full-document BM25 查询语义后，`bm25_full_skill` 在 hard distractor 和 full-library 设置下都表现良好。
4. **Hard distractors 依然显著拉低所有 retrievers**，说明 RQ2 的结论在 RQ3 中仍然成立：retriever choice 能缓解问题，但不能消除 skill competition。
5. **Naive full-document TF-IDF 反而变差**，说明更长文本需要 field weighting、chunking 或 late interaction，而不是把整份 `SKILL.md` 直接摊平成一个大文档。

因此，增强版 RQ3 的答案可以更新为：

**在当前 Skill-Usage 设置下，最佳方案不再是单独 BM25，而是 BM25 与轻量神经 dense retriever 的 hybrid。完整 `SKILL.md` 内容在 BM25 下有帮助，但在 naive TF-IDF 下会引入明显噪声。**

---

## 2. 这次具体解决了什么

上一版 RQ3 的限制与这次处理方式如下：

| 上一版限制 | 这次如何处理 |
|---|---|
| 不是 true neural retriever comparison | 加入本地缓存的 `all-MiniLM-L6-v2` dense retriever |
| LSA dense 只是 proxy | 用 MiniLM dense 替代 LSA proxy |
| 只用了 skill name + description | 加入完整 `SKILL.md` 的 BM25 / TF-IDF 条件 |
| 只测 random distractors | 加入 `query_overlap:100` 和 `embedding_semantic_near:100` |
| latency 只是脚本级别 | 这点仍未解决，当前仍是本地脚本测量 |

一个额外修正也很重要：

- 初版 `bm25_full_skill` 使用 SQLite FTS5 的默认查询语义，会把空格分隔的 token 近似当作 AND 查询，导致 full-document BM25 被不公平压低。
- 增强版脚本已显式把自然语言 query 转成 lexical OR 查询，使它更接近前面 bag-of-words BM25 的设定。

---

## 3. 实验设置

### 数据

- 数据集：Skill-Usage
- Tasks：87
- Skill library size：34,396
- Query 来源：`data/raw/Skill-Usage/data/task_queries.json`
- Gold skill 来源：`data/raw/Skill-Usage/data/task_skill_mapping.json`

### 检索条件

| Retriever | Definition |
|---|---|
| `bm25_desc` | BM25 over skill name + description |
| `tfidf_desc` | TF-IDF cosine over skill name + description |
| `neural_minilm_desc` | Cached MiniLM dense retriever over skill name + description |
| `hybrid_bm25_neural` | Reciprocal-rank fusion of BM25 and MiniLM |
| `bm25_full_skill` | SQLite FTS5 BM25 over name + description + full `SKILL.md` |
| `tfidf_full_skill` | TF-IDF cosine over name + description + full `SKILL.md` |

### Candidate Settings

| Setting | Meaning |
|---|---|
| `random:1000` | 1000-size candidate pool with random distractors |
| `random:full` | Full 34,396-skill library |
| `query_overlap:100` | Hard lexical distractors close to the task query |
| `embedding_semantic_near:100` | Hard semantic distractors close to the gold skill embedding centroid |

对非 full setting 使用 3 次重复；full library 使用 1 次完整检索。

---

## 4. 主要结果

### 4.1 Full library 结果

| Retriever | Top-1 | Hit@10 | MRR@10 |
|---|---:|---:|---:|
| `hybrid_bm25_neural` | **0.460** | 0.724 | **0.549** |
| `bm25_full_skill` | 0.437 | **0.759** | 0.535 |
| `bm25_desc` | 0.425 | 0.667 | 0.513 |
| `neural_minilm_desc` | 0.414 | 0.701 | 0.512 |
| `tfidf_desc` | 0.356 | 0.655 | 0.461 |
| `tfidf_full_skill` | 0.241 | 0.517 | 0.314 |

相对原始 RQ3 baseline：

- `bm25_desc` 与上一版 BM25 完全一致：Top-1 = **0.425**
- `hybrid_bm25_neural` 把 full-library Top-1 提升到 **0.460**，绝对提升 **+0.035**
- `bm25_full_skill` 也优于 description-only BM25：Top-1 **0.437**，Hit@10 **0.759**

这说明：

- 轻量神经 dense 本身不是 full-library 下的单独赢家
- 但它为 BM25 提供了有价值的补充信号
- 完整文档内容在 BM25 框架下可以带来额外 recall

### 4.2 Hard distractor 结果

#### Query-overlap hard distractors (`query_overlap:100`)

| Retriever | Top-1 | Hit@10 | MRR@10 |
|---|---:|---:|---:|
| `bm25_full_skill` | **0.506** | **0.885** | **0.620** |
| `neural_minilm_desc` | 0.448 | 0.851 | 0.572 |
| `hybrid_bm25_neural` | 0.448 | 0.759 | 0.554 |
| `bm25_desc` | 0.437 | 0.690 | 0.527 |
| `tfidf_desc` | 0.391 | 0.655 | 0.486 |
| `tfidf_full_skill` | 0.287 | 0.747 | 0.400 |

#### Semantic-near hard distractors (`embedding_semantic_near:100`)

| Retriever | Top-1 | Hit@10 | MRR@10 |
|---|---:|---:|---:|
| `bm25_full_skill` | **0.563** | 0.816 | **0.640** |
| `bm25_desc` | 0.540 | 0.816 | 0.628 |
| `hybrid_bm25_neural` | 0.540 | 0.816 | 0.632 |
| `neural_minilm_desc` | 0.529 | 0.816 | 0.626 |
| `tfidf_desc` | 0.506 | 0.782 | 0.594 |
| `tfidf_full_skill` | 0.391 | 0.736 | 0.474 |

这里有两个很有意思的观察：

1. **在 hard distractor 条件下，`bm25_full_skill` 反而是最稳的。**  
   完整 `SKILL.md` 提供了更多 task-specific lexical anchors，可以帮助 BM25 区分看起来相似的候选技能。

2. **神经 dense 有帮助，但没有压倒性优势。**  
   这意味着 Skill-Usage 的 query-skill 对齐仍然保留了很强的 lexical structure，dense semantic similarity 不是唯一主导因素。

---

## 5. 与原始 RQ3 的关系

原始 RQ3 的主要结论是：

- BM25 是最强 baseline
- TF-IDF 次之
- LSA dense proxy 很弱
- LSA-based hybrid / rerank 会伤害结果

增强版 RQ3 说明这个结论需要更精确地改写：

- **“dense retrieval 不行” 这个结论不成立。**  
  不行的是 LSA proxy，不是真正的神经 dense。

- **“hybrid retrieval 会伤害结果” 也不应泛化。**  
  LSA-based hybrid 会伤害结果，但 MiniLM-based hybrid 在 full library 下是当前最好的设置。

- **“full `SKILL.md` 会造成噪声” 只能算部分成立。**  
  Naive TF-IDF full-content 会退化，但 BM25 full-content 在这次修正后是有效的，尤其在 hard distractor 下表现突出。

换句话说，原始 RQ3 更像是在说明：

**弱 dense proxy 会误导 hybrid；但一个真实、轻量、已缓存的神经 dense retriever 可以让 hybrid 重新成立。**

---

## 6. 解释

### 6.1 为什么 hybrid 在 full library 下最好

BM25 擅长抓显式关键词、工具名、文件格式和技术名词。MiniLM dense 对 lexical mismatch 更宽容，可以把语义上接近但表达略有差异的 gold skill 拉上来。二者做 reciprocal-rank fusion 后，full library 下获得了最好的 Top-1 和 MRR。

### 6.2 为什么 `bm25_full_skill` 在 hard distractor 下这么强

Hard distractors 往往在短 description 上长得很像，但完整 `SKILL.md` 包含更多命令、环境、文件、限制条件和任务步骤。对于 BM25 这类 lexical retriever，这些额外 token 会形成更细粒度的区分信号，因此它在 `query_overlap` 和 `embedding_semantic_near` 下都超过了 `bm25_desc`。

### 6.3 为什么 `tfidf_full_skill` 反而退化

TF-IDF 在长文本下更容易被稀有 token、模板性内容和局部噪声放大。完整 `SKILL.md` 并非所有段落都与 task query 同等相关，因此把整份文档当成单一向量，会稀释最关键的匹配区域。

这提示后续更合适的做法应该是：

- field-weighted retrieval
- chunk-level retrieval
- first-stage retrieve + rerank

---

## 7. 成本与延迟说明

当前 summary 里的 `build_seconds` 反映的是**当前运行方式**：

- `neural_minilm_desc` / `hybrid_bm25_neural` 这次直接复用了已有的 `neural_doc_embeddings.npy` 缓存，所以 warm-cache build time 很低
- `tfidf_full_skill` 的 build 成本明显更高：约 **9.84s**
- `bm25_full_skill` 的查询成本最高：约 **0.079s / query**
- description-only BM25 仍然是最快的一阶段检索器：约 **0.00038s / query**

因此，增强版 RQ3 支持这样的工程结论：

- 如果只追求最快速度，`bm25_desc` 仍然很强
- 如果允许稍高计算成本，`hybrid_bm25_neural` 值得作为更强主方案
- 如果目标是 hard-distractor robustness，`bm25_full_skill` 值得单独保留

---

## 8. 当前仍然存在的限制

- 当前 neural retriever 使用的是 `all-MiniLM-L6-v2`，它是轻量可缓存模型，不等于项目计划书里提到的 BGE / E5 / Qwen embedding。
- `bm25_full_skill` 虽然已经可比，但仍然依赖 SQLite FTS5，而不是与 `bm25_desc` 完全同一实现。
- 当前 hard negatives 只覆盖 `query_overlap` 和 `embedding_semantic_near`；还没有为 neural retriever 单独构造 retriever-specific dense-hard negatives。
- latency 仍然是本地脚本级测量，不是生产 serving benchmark。
- 这轮实验依然只评估 retrieval，不评估 downstream task completion。

---

## 9. 下一步建议

1. 增加 **dense-hard negatives**：对 `neural_minilm_desc` 取其最高排名 non-gold 作为 retriever-specific hard negatives。
2. 做 **field-aware full-content retrieval**：分别给 title / description / `SKILL.md` body 设权重，而不是直接平铺。
3. 做 **chunk retrieval + reranking**：先检索 `SKILL.md` chunk，再回溯到 skill 级别。
4. 在 **RQ4** 中验证：这些 retrieval 改进是否真的提升 downstream task completion。

---

## 10. Reproducibility

运行增强版 RQ3：

```bash
python3 experiments/rq3_retriever_enhanced.py
```

主要输出：

- `data/experiments/rq3_retriever_enhanced/summary.csv`
- `data/experiments/rq3_retriever_enhanced/summary.json`
- `data/experiments/rq3_retriever_enhanced/per_query_metrics.csv`
- `data/experiments/rq3_retriever_enhanced/neural_doc_embeddings.npy`
