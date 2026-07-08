# 第一个实验：Retrieval Scaling Pilot

**日期**：2026-07-08  
**目的**：先验证 skill library 规模增大时，检索准确率是否下降。  
**数据**：`data/raw/Skill-Usage`  
**脚本**：`experiments/retrieval_scaling_pilot.py`  
**输出目录**：`data/experiments/retrieval_scaling_pilot`

---

## 实验设置

- 任务数：87
- Query 来源：`data/raw/Skill-Usage/data/task_queries.json`
- Gold skill 来源：`data/raw/Skill-Usage/data/task_skill_mapping.json`
- Skill pool 来源：`data/raw/Skill-Usage/skills-34k/skills_meta.jsonl`
- Candidate pool size：10、50、100、500、1000、5000、10000、full
- Distractor 类型：random distractor
- Retriever：本地轻量 BM25，基于 skill name + description
- 重复次数：非 full pool 每档 5 次；full pool 1 次
- 指标：Top-1 Accuracy、Recall@3、Recall@5、Recall@10、MRR、NDCG@10

---

## 初步结果

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

---

## 观察

1. 随着 candidate pool 从 10 扩大到 full，Top-1 Accuracy 从 0.963 降到 0.414。
2. Recall@10 从 0.880 降到 0.449，说明即使允许 top-10，full library 下仍有大量 gold skill 排不进候选结果。
3. 这组结果支持项目主假设：更多 skills 会显著增加 retrieval noise，使 agent 更难找到正确 skill。

---

## 当前限制

- 这是 pilot，不是最终实验。
- 当前 retriever 只用 skill name + description，没有使用完整 `SKILL.md` 内容。
- 当前 distractor 只有 random；后续还需要加入 same-owner、same-repo、lexical-overlap、semantic-near、hard negative。
- 当前 BM25 是本地轻量实现；后续需要接入官方 search server 的 keyword / semantic / hybrid 检索结果。

---

## 下一步

1. 安装 search server 依赖，跑官方 keyword / semantic / hybrid baseline。
2. 扩展 `experiments/retrieval_scaling_pilot.py`，加入 same-owner、same-repo 和 lexical-overlap distractors。
3. 增加 plots：pool size vs Top-1、Recall@10、NDCG@10。
4. 抽取 full pool 失败案例，做 qualitative analysis。

