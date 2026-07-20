# Question 8 Literature Review


## 1. Large skill libraries and retrieval bottlenecks

### SkillsBench

Li et al. (2026) 将 Agent Skills 定义为推理时提供程序性知识的结构化包，并在 87 个任务、8 个领域上进行 paired evaluation。论文报告 curated skills 平均提高通过率，但 focused bundles 优于更大或穷举式 bundles。这为本实验的固定 Top-10 预算与 context-pollution 指标提供直接依据。

- Source: https://arxiv.org/abs/2602.12670

### SkillRet

Cho, Kang, and Kim (2026) 构建 17,810 个 skills、63,259 个训练样本和 4,997 个评估 queries 的大规模 skill retrieval benchmark，指出 off-the-shelf retrieval 在真实大规模 skill library 上仍远未解决。其结论直接支持本实验把技能选择视为独立的信息检索问题。

- Source: https://arxiv.org/abs/2605.05726

### Skill Retrieval Augmentation / SRA-Bench

Su et al. (2026) 将 agent skill usage 分解为 retrieval、incorporation、execution，并指出显式枚举全部技能会消耗上下文且降低选择准确性。本实验只评价 retrieval/exposure 层，不把 retrieval 改善误写成 downstream execution 改善。

- Source: https://arxiv.org/abs/2604.24594

## 2. Why a graph may help

### GraphRAG

Edge et al. (2024) 将语料组织为实体知识图与社区摘要，说明图索引可以补充传统局部文本检索。不过 GraphRAG 的任务是全局语料问答，而本实验的图节点是 skills、边表示可组合关系，两者只能作为结构化 retrieval 的方法动机，不能直接类比结果。

- Source: https://arxiv.org/abs/2404.16130

### HippoRAG

Gutiérrez et al. (NeurIPS 2024) 结合 knowledge graph 与 Personalized PageRank 进行多跳检索，并报告相对迭代检索更低的成本。本实验借鉴其“先取局部 seeds，再沿图扩展”的基本思想，但使用可解释的一跳加权扩展而非 PageRank。

- Source: https://arxiv.org/abs/2405.14831

### Multi-Hop Dense Retrieval

Xiong et al. (2020/2021) 表明复杂问题可能需要分阶段找到互补证据，而一次性相似度排序未必能覆盖全部证据。本实验将同一逻辑迁移到 multi-skill retrieval：命中一个 seed skill 后，尝试补全 companion skills。

- Source: https://arxiv.org/abs/2009.12756

## 3. Retrieval fusion and co-occurrence relations

### Reciprocal Rank Fusion

Cormack, Clarke, and Büttcher (SIGIR 2009) 提出 Reciprocal Rank Fusion。本实验沿用原 RQ3 的 BM25 + MiniLM RRF 作为强 baseline，避免把图方法与弱检索器比较。

- Publisher: https://dl.acm.org/doi/10.1145/1571941.1572114

### Item-based collaborative filtering

Sarwar et al. (WWW 2001) 证明 item–item 共现关系可用于推荐。`skill_co_required_with` 与 item-based co-occurrence 在数学上相似：如果两个 skills 在训练任务中共同出现，就给二者建立加权关系。但本数据中绝大多数 skills 只出现一次，因此共现图存在严重 cold-start。

- Publisher: https://dl.acm.org/doi/10.1145/371920.372071

## 4. Gap addressed by Question 8

已有工作分别说明了四点：skills 有价值但 focused exposure 更好；大规模 skill retrieval 很难；图结构能够支持多跳检索；共现关系能连接互补 items。尚未被这些工作直接回答的问题是：在 Markdown skill library 中，内容关系和任务共现关系能否在固定 context budget 下补全 multi-skill set，同时控制额外 skills。

Question 8 的贡献不是提出一个已被充分训练的新图模型，而是做一个受控诊断实验：

1. 在固定 5,000-skill hard pool 和 Top-10 budget 下比较文本检索与图扩展；
2. 用 5-fold task split 防止当前测试 task 的 gold edges 泄漏；
3. 同时报告 leakage-free 主结果和 transductive potential upper bound；
4. 用 edge-type ablation 判断 semantic、metadata、co-required 边各自的作用与风险。
