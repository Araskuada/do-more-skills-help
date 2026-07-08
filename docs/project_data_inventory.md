# 项目数据清单与下载状态

**项目**：Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents  
**更新日期**：2026-07-08  
**结论**：第一阶段可复现实验所需的核心数据已经下载完成，不需要重复下载。

---

## 1. 全套流程数据列表

| 优先级 | 数据 | 本地路径 | 当前状态 | 在流程中的作用 |
|---|---|---|---|---|
| 必需 | Skill-Usage repo | `data/raw/Skill-Usage` | 已下载 | 主实验代码、任务、脚本、检索服务 |
| 必需 | Skill-Usage 34k skill pool | `data/raw/Skill-Usage/skills` | 已解压 | 大规模 candidate skill library |
| 必需 | Skill-Usage metadata | `data/raw/Skill-Usage/skills-34k/skills_meta.jsonl` | 已下载 | skill id、source、repo、description、license、安全标签等 |
| 必需 | Skill-Usage compressed skills | `data/raw/Skill-Usage/skills-34k/skills.zip` | 已下载 | 原始压缩备份，后续无需重复下载 |
| 必需 | Skill-Usage search index | `data/raw/Skill-Usage/search_server/index` | 已解压 | BM25 / semantic / hybrid 检索索引 |
| 必需 | Skill-Usage compressed search index | `data/raw/Skill-Usage/search_index/search_index.zip` | 已下载 | 预建索引压缩备份 |
| 必需 | Skill-Usage task queries | `data/raw/Skill-Usage/data/task_queries.json` | 已下载 | retrieval query 输入 |
| 必需 | Skill-Usage task-skill mapping | `data/raw/Skill-Usage/data/task_skill_mapping.json` | 已下载 | gold / curated skill 对照 |
| 推荐 | SkillsBench | `data/raw/skillsbench` | 已下载 | downstream validation、curated skill 对照、小规模 agent task 实验 |
| 推荐 | SWE-Skills-Bench | `data/raw/SWE-Skills-Bench` | 已下载 | SWE 场景 skill/no-skill paired evaluation、token overhead 分析 |
| 可选 | Terminal-Bench 2.0 subset | `data/raw/Skill-Usage/terminal-bench-2` | 已下载 | Skill-Usage 论文复现实验的第二类任务 |
| 待公开/待确认 | SkillRet | `data/raw/SkillRet` | 未下载，未发现明确公开入口 | 最贴近 retrieval scaling 的理想主数据；当前不阻塞第一版实验 |
| 待公开/待确认 | SRA-Bench | `data/raw/SRA-Bench` | 未下载，未发现明确公开入口 | 可用于拆分 retrieval / incorporation / execution 的补充实验 |
| 可选 proxy | ToolRet | `data/raw/ToolRet` | 未下载，当前没有稳定下载入口 | tool retrieval proxy；不是 skill 主实验必需数据 |
| 可选 proxy | ToolMenuBench | `data/raw/ToolMenuBench` | 未下载，当前没有稳定下载入口 | tool-menu filtering 指标借鉴；不是第一阶段必需数据 |

---

## 2. 已核验的本地数据规模

| 项目 | 核验结果 |
|---|---:|
| `data/raw/Skill-Usage` 总体大小 | 7.0G |
| `data/raw/Skill-Usage/skills-34k/skills.zip` | 751M |
| `data/raw/Skill-Usage/skills-34k/skills_meta.jsonl` | 27M |
| `data/raw/Skill-Usage/search_index/search_index.zip` | 1.1G |
| `data/raw/Skill-Usage/skills` 中 `SKILL.md` 数 | 35,554 |
| `data/raw/Skill-Usage/tasks` 任务数 | 87 |
| `data/raw/Skill-Usage/terminal-bench-2` 任务数 | 89 |
| `data/raw/skillsbench` 总体大小 | 1.1G |
| SkillsBench 默认任务数 | 87 |
| SkillsBench extra 任务数 | 14 |
| SkillsBench `SKILL.md` 数 | 276 |
| `data/raw/SWE-Skills-Bench` 总体大小 | 14M |
| SWE-Skills-Bench skill 数 | 49 |
| SWE-Skills-Bench task markdown 数 | 499 |

---

## 3. 推荐项目启动顺序

### Stage A：先跑 retrieval scaling 主实验

使用 `Skill-Usage`：

1. 从 `data/task_queries.json` 读 query；
2. 从 `data/task_skill_mapping.json` 读 gold / curated skill；
3. 从 `skills_meta.jsonl` 和 `skills/` 构造候选 skill pool；
4. 用 `search_server/index` 启动 keyword / semantic / hybrid 检索；
5. 构造 pool size：10、50、100、500、1000、5000、10000、full；
6. 构造 distractor：random、same-owner、same-repo、lexical-overlap、semantic-near；
7. 记录 Top-1 Accuracy、Recall@K、MRR、NDCG@10、latency、candidate size。

### Stage B：做 downstream validation

使用 `SkillsBench` 和 `Skill-Usage/tasks`：

1. no skill baseline；
2. curated / gold skill；
3. retrieved top-1；
4. retrieved top-3 / top-5；
5. retrieved + distractors；
6. all visible skills 的小库压力测试。

### Stage C：做 SWE 补充实验

使用 `SWE-Skills-Bench`：

1. no-skill vs use-skill；
2. pass rate；
3. token usage；
4. failure case 分类：wrong skill、irrelevant skill、outdated guidance、context pollution。

---

## 4. 如需重新下载的命令

当前无需执行。只有在数据损坏或换机器时再用：

```bash
cd data/raw
git clone https://github.com/UCSB-NLP-Chang/Skill-Usage.git
cd Skill-Usage
hf download Shiyu-Lab/Skill-Usage skills-34k/skills.zip skills-34k/skills_meta.jsonl --repo-type dataset --local-dir .
unzip skills-34k/skills.zip -d skills/
cp skills-34k/skills_meta.jsonl skills/
hf download Shiyu-Lab/Skill-Usage search_index/search_index.zip --repo-type dataset --local-dir .
unzip search_index/search_index.zip -d search_server/index/
```

```bash
cd data/raw
git clone https://github.com/benchflow-ai/skillsbench.git
git clone https://github.com/GeniusHTX/SWE-Skills-Bench.git
```

---

## 5. 当前不下载的原因

- `SkillRet`：论文最贴近本项目主问题，但截至本次检查没有发现明确公开 GitHub 或 Hugging Face 数据入口。
- `SRA-Bench`：论文数据设计很适合补充实验，但截至本次检查没有发现明确公开数据入口。
- `ToolRet` / `ToolMenuBench`：属于 tool retrieval / tool menu proxy，不是当前 skill-library scaling 主流程的必需数据；等主实验跑通后再决定是否加入。

