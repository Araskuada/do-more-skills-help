# 数据下载与使用文档

**项目题目**：Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents  
**当前阶段**：项目第一步 - 数据下载、盘点与使用说明  
**更新日期**：2026-07-05

---

## 0. 当前数据状态总览

本阶段已经下载并整理了三个可立即使用的数据源：

| 数据源 | 本地路径 | 当前状态 | 用途 |
|---|---|---|---|
| SkillsBench | `data/raw/skillsbench` | 已下载 | downstream validation、task/verifier、curated skill 对照 |
| SWE-Skills-Bench | `data/raw/SWE-Skills-Bench` | 已下载 | 软件工程场景 skill/no-skill paired evaluation |
| Skill-Usage | `data/raw/Skill-Usage` | 已下载 repo、34k skills、metadata、search index | 主线替代数据：大规模 real-world skill retrieval scaling |

仍需继续确认下载入口的数据：

| 数据源 | 状态 | 说明 |
|---|---|---|
| SkillRet | 待确认下载入口 | 论文最贴近主实验，但当前未找到明确公开 GitHub/Hugging Face 链接 |
| SRA-Bench | 待确认下载入口 | 理论框架很适合，但当前未找到明确公开数据链接 |
| ToolRet | 待确认下载入口 | 可作为 tool retrieval proxy，但当前未找到明确公开数据链接 |
| ToolMenuBench | 待确认下载入口 | 适合 controlled exposure 指标借鉴，暂不作为第一阶段数据 |

---

## 1. 本地目录结构

```text
data/
  raw/
    skillsbench/
    SWE-Skills-Bench/
    Skill-Usage/
  external/
docs/
  literature_deep_reading_summary.md
  data_usage_guide.md
```

### 数据体积

当前本地统计：

| 路径 | 体积 |
|---|---:|
| `data/raw/skillsbench` | 1.1G |
| `data/raw/SWE-Skills-Bench` | 14M |
| `data/raw/Skill-Usage` | 7.0G |
| `data/raw/Skill-Usage/skills` | 1.9G |
| `data/raw/Skill-Usage/search_server/index` | 2.2G |

---

## 2. SkillsBench

### 2.1 来源

GitHub：

```text
https://github.com/benchflow-ai/skillsbench
```

Hugging Face 页面：

```text
https://huggingface.co/datasets/benchflow/skillsbench
```

本地路径：

```text
data/raw/skillsbench
```

### 2.2 本地统计

当前本地仓库统计：

- 默认任务目录：`data/raw/skillsbench/tasks`
- 默认任务数：87
- 额外任务目录：`data/raw/skillsbench/tasks-extra`
- 额外任务数：14
- `SKILL.md` 文件数：276
- 文件总数：2895

注意：arXiv 摘要中当前版本提到 87 tasks across 8 domains；本地仓库也包含 87 个默认任务目录。

### 2.3 关键文件

| 文件 / 目录 | 作用 |
|---|---|
| `README.md` | 官方使用说明 |
| `registry.json` | benchmark registry 信息 |
| `taxonomy.yaml` / `taxonomy.md` | 任务/技能分类体系 |
| `tasks/` | 默认可运行任务 |
| `tasks-extra/` | 额外任务，可能需要特殊依赖或 credentials |
| `experiments/run_experiment.ipynb` | 官方实验流程示例 |
| `skillsbench_agentbeats/` | benchmark 运行相关代码 |

### 2.4 任务结构

README 中给出的任务结构：

```text
tasks/<task-id>/
  task.md
  environment/
    Dockerfile
    skills/
  oracle/
    solve.sh
  verifier/
    test.sh
    test_outputs.py
```

### 2.5 推荐用途

SkillsBench 更适合作为 **downstream validation**，而不是第一阶段的 retrieval scaling 主数据。

推荐比较条件：

1. no skill；
2. gold / curated skill；
3. retrieved top-1 skill；
4. retrieved top-3 skills；
5. noisy retrieved skills；
6. all visible skills，小库场景。

推荐指标：

- task pass rate；
- wrong skill invocation rate；
- skill overloading rate；
- skill underloading rate；
- average token usage；
- average execution cost。

### 2.6 快速检查命令

```bash
find data/raw/skillsbench/tasks -maxdepth 1 -mindepth 1 -type d | wc -l
find data/raw/skillsbench -name 'SKILL.md' | wc -l
```

### 2.7 运行成本与注意事项

- 完整运行需要 BenchFlow CLI、Docker 和模型 API key；
- `tasks-extra/` 中的任务可能需要额外凭据或外部服务；
- 一个项目月内建议只抽样 5-10 个任务做 downstream demo；
- 主实验优先不要依赖完整 SkillsBench 执行环境。

---

## 3. SWE-Skills-Bench

### 3.1 来源

GitHub：

```text
https://github.com/GeniusHTX/SWE-Skills-Bench
```

Hugging Face 使用方式见仓库 README：

```python
from datasets import load_dataset
ds = load_dataset("GeniusHTX/SWE-Skills-Bench", split="train")
```

本地路径：

```text
data/raw/SWE-Skills-Bench
```

### 3.2 本地统计

当前本地仓库统计：

- `skills/` 下有 49 个 `SKILL.md`；
- `tasks/` 下有 499 个 markdown task instance 文件；
- 论文摘要称 approximately 565 task instances，本地仓库任务文件数与论文摘要可能因版本或生成方式不同，需要以后按官方 dataset split 再核对。

### 3.3 关键文件

| 文件 / 目录 | 作用 |
|---|---|
| `README.md` | 官方使用说明 |
| `skills/` | 49 个 SWE skill packages |
| `tasks/` | 按 batch 组织的 task prompts |
| `tests/` | 对应任务测试 |
| `config/benchmark_config.yaml` | benchmark 配置 |
| `main.py` | 验证配置、列出 skills 等入口 |
| `run_all_skills.py` | 批量运行 Agent |
| `run_all_skills_eval.py` | 批量评估 |
| `scripts/compare_pass_rate.py` | skill vs no-skill pass rate 对比 |
| `scripts/analyze_tokens.py` | token 和时长分析 |

### 3.4 推荐用途

SWE-Skills-Bench 适合用于：

- downstream validation；
- skill/no-skill paired evaluation；
- token overhead 分析；
- contextual mismatch / outdated guidance 的案例分析；
- 软件工程子领域的补充证据。

不建议一开始作为主实验，因为完整运行需要：

- Docker；
- Claude Code CLI；
- Anthropic API key；
- 真实 repo/container 运行时间。

### 3.5 官方运行步骤摘要

验证配置：

```bash
cd data/raw/SWE-Skills-Bench
python main.py validate --config config/benchmark_config.yaml
python main.py list-skills --config config/benchmark_config.yaml
```

运行实验：

```bash
python run_all_skills.py --use-skill
python run_all_skills.py --no-use-skill
```

评估：

```bash
python run_all_skills_eval.py --use-skill --use-agent
python run_all_skills_eval.py --no-use-skill --use-agent
```

汇总：

```bash
python scripts/compare_pass_rate.py --all
python scripts/extract_failed_tests.py
python scripts/analyze_tokens.py
```

### 3.6 快速检查命令

```bash
find data/raw/SWE-Skills-Bench/skills -name 'SKILL.md' | wc -l
find data/raw/SWE-Skills-Bench/tasks -name '*.md' | wc -l
```

---

## 4. Skill-Usage

### 4.1 来源

GitHub：

```text
https://github.com/UCSB-NLP-Chang/Skill-Usage
```

Hugging Face：

```text
https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage
```

本地路径：

```text
data/raw/Skill-Usage
```

### 4.2 已下载内容

已经完成：

1. GitHub 仓库：

```text
data/raw/Skill-Usage
```

2. 34k skills 压缩包和 metadata：

```text
data/raw/Skill-Usage/skills-34k/skills.zip
data/raw/Skill-Usage/skills-34k/skills_meta.jsonl
```

3. 解压后的 skills：

```text
data/raw/Skill-Usage/skills
```

4. 预建 search index：

```text
data/raw/Skill-Usage/search_index/search_index.zip
data/raw/Skill-Usage/search_server/index
```

### 4.3 本地统计

当前本地统计：

- `skills-34k/skills.zip`：751M；
- `skills-34k/skills_meta.jsonl`：27M；
- `skills/`：1.9G；
- `search_server/index`：2.2G；
- 本地 `SKILL.md` 文件数：35,554；
- 官方 README 描述：34k real-world skill collection。

说明：本地 `SKILL.md` 计数略高于“34k”，可能因为目录结构、整理版本或额外条目造成。实验报告中建议写作：

> We use the Skill-Usage real-world skill pool, described by the authors as a 34k skill collection; the local unpacked copy contains 35,554 `SKILL.md` files.

### 4.4 关键文件

| 文件 / 目录 | 作用 |
|---|---|
| `README.md` | 官方使用说明 |
| `data/task_queries.json` | task query 信息 |
| `data/task_skill_mapping.json` | task 到 skill 的 mapping |
| `skills-34k/skills_meta.jsonl` | skills metadata，一行一个 JSON |
| `skills/` | 解压后的 skill packages |
| `search_server/` | BM25、semantic、hybrid search server |
| `search_server/index/` | 官方预建检索索引 |
| `scripts/eval_retrieval.py` | retrieval 评价脚本 |
| `scripts/sweep_retrieval.py` | retrieval sweep |
| `scripts/retrieve_skills.py` | agentic retrieval |
| `scripts/select_top_k_skills.py` | 为任务复制 top-k skills |
| `scripts/calculate_results.py` | 结果汇总 |

### 4.5 Metadata 字段样例

`skills_meta.jsonl` 每行是一个 skill metadata。常见字段包括：

| 字段 | 说明 |
|---|---|
| `id` | 原始路径式 ID |
| `skillId` | skill 标识 |
| `name` | skill 名称 |
| `source` | GitHub owner/repo 来源 |
| `owner` | GitHub owner |
| `repo` | GitHub repo |
| `github_url` | GitHub 链接 |
| `skill_id` | 规范化 skill id |
| `github_stars` | 来源仓库 stars |
| `description` | skill description |
| `github_license` | 来源 license |
| `safety_judge` | 安全判断标签 |

样例路径：

```bash
head -3 data/raw/Skill-Usage/skills-34k/skills_meta.jsonl
```

### 4.6 推荐用途

Skill-Usage 是当前最适合立即启动主实验的数据：

- 有大规模 real-world skill pool；
- 有官方 search server；
- 有 BM25 / semantic / hybrid search；
- 有 task queries；
- 有 task-skill mapping；
- 有 SkillsBench 和 Terminal-Bench 任务；
- 支持 retrieved_w_curated、retrieved_wo_curated、distractors、refinement 等设置。

如果 SkillRet 暂时无法下载，建议用 Skill-Usage 做第一版 retrieval scaling。

### 4.7 启动 Search Server

进入目录：

```bash
cd data/raw/Skill-Usage
```

安装 search server 依赖：

```bash
pip install -r search_server/requirements.txt
```

启动服务：

```bash
python search_server/http_server.py --include-content
```

默认端口：

```text
8742
```

可用 endpoint：

```text
GET /keyword?q=...&top_k=10
GET /semantic?q=...&top_k=10
GET /hybrid?q=...&top_k=10
GET /detail/{skill_id}
```

### 4.8 推荐的 retrieval scaling 改造

官方 search server 是一个很好的起点，但我们的项目还需要额外实现 candidate pool scaling。

建议构造数据表：

```text
query_id
query_text
gold_skill_ids
candidate_pool_size
distractor_mode
candidate_skill_ids
retriever
ranked_skill_ids
scores
metrics
```

候选池大小：

- 10；
- 50；
- 100；
- 500；
- 1000；
- 5000；
- 10000；
- full。

Distractor 模式：

- random；
- same-source repo；
- same-owner；
- metadata lexical overlap；
- semantic-near；
- retrieved hard negative。

如果没有 SkillRet 那样的 category/subcategory，可以用以下字段替代：

- `owner`；
- `repo`；
- `github_license`；
- `description` lexical similarity；
- embedding clustering；
- top semantic nearest non-gold。

---

## 5. 尚未下载的数据

### 5.1 SkillRet

论文链接：

```text
https://arxiv.org/abs/2605.05726
```

状态：

- 论文最贴近我们的主实验；
- 摘要确认有 17,810 skills、63,259 training samples、4,997 eval queries；
- 当前未找到明确公开 GitHub/Hugging Face 数据入口。

后续动作：

- 继续搜索作者主页、CatalyzeX、Hugging Face；
- 检查 arXiv source 中是否包含 data URL；
- 必要时邮件联系作者；
- 一旦拿到数据，将其放入：

```text
data/raw/SkillRet
```

### 5.2 SRA-Bench

论文链接：

```text
https://arxiv.org/abs/2604.24594
```

状态：

- 摘要确认有 5,400 test instances、636 gold skills、26,262 skills corpus；
- 当前未找到明确公开 GitHub/Hugging Face 数据入口。

后续动作：

- 继续跟进作者发布；
- 如果可用，放入：

```text
data/raw/SRA-Bench
```

### 5.3 ToolRet

论文链接：

```text
https://arxiv.org/abs/2503.01763
```

状态：

- 摘要确认有 7.6K retrieval tasks、43K tools、200K+ training instances；
- 当前未找到明确公开数据下载入口。

后续动作：

- 继续搜索论文附录、作者仓库、Hugging Face；
- 如果可用，放入：

```text
data/raw/ToolRet
```

---

## 6. 推荐实验数据路线

### 路线 A：立即可做

使用本地已下载数据：

```text
Skill-Usage 34k skills
  + task_queries.json
  + task_skill_mapping.json
  + search_server/index
```

目标：

- 先复用官方 search server；
- 构造不同大小 candidate pools；
- 实现 BM25 / semantic / hybrid 对比；
- 做 random vs semantic-near distractors。

优点：

- 数据已经下载；
- 规模足够大；
- 贴近真实 skill pool；
- 有检索脚本可参考。

缺点：

- category/subcategory 不如 SkillRet 标准；
- gold qrels 需要从 task_skill_mapping 或 retrieval settings 中整理；
- 不完全等价于 SkillRet 的 benchmark setting。

### 路线 B：首选但需等待

使用 SkillRet：

```text
SkillRet skills + queries + qrels + taxonomy
```

优点：

- 最贴合论文题目；
- 有标准 qrels；
- 有 category/subcategory；
- 非常适合 library size scaling。

缺点：

- 当前数据入口未确认。

### 路线 C：Downstream Validation

使用 SkillsBench 或 SWE-Skills-Bench：

```text
SkillsBench tasks
SWE-Skills-Bench tasks
```

目标：

- 小规模验证 retrieval 错误是否影响 task pass rate；
- 不作为主实验，只作为补充证据。

---

## 7. 数据使用规范

### 7.1 不要直接修改 raw data

`data/raw/` 下的数据保持原样。任何清洗或转换结果放到：

```text
data/processed/
```

建议目录：

```text
data/processed/
  skill_usage/
    skills_catalog.jsonl
    task_queries.jsonl
    qrels.jsonl
    candidate_pools/
  skillsbench/
  swe_skills_bench/
```

### 7.2 建议统一格式

统一 skill catalog：

```json
{
  "skill_id": "...",
  "name": "...",
  "description": "...",
  "content_path": "...",
  "source": "...",
  "owner": "...",
  "repo": "...",
  "metadata": {}
}
```

统一 query：

```json
{
  "query_id": "...",
  "query_text": "...",
  "benchmark": "skill_usage",
  "gold_skill_ids": ["..."]
}
```

统一 candidate pool：

```json
{
  "query_id": "...",
  "pool_size": 1000,
  "distractor_mode": "semantic_near",
  "seed": 0,
  "gold_skill_ids": ["..."],
  "candidate_skill_ids": ["..."]
}
```

统一 retrieval result：

```json
{
  "query_id": "...",
  "retriever": "bm25",
  "pool_size": 1000,
  "distractor_mode": "random",
  "seed": 0,
  "ranked_skill_ids": ["..."],
  "scores": [1.0, 0.8],
  "metrics": {
    "top1": 1,
    "recall_at_10": 1,
    "mrr": 1.0,
    "ndcg_at_10": 1.0
  }
}
```

### 7.3 License 注意事项

当前已下载仓库 license：

| 数据源 | License |
|---|---|
| SkillsBench | Apache 2.0 |
| SWE-Skills-Bench | MIT |
| Skill-Usage | 需进一步查看仓库和 Hugging Face dataset card；skill metadata 中每个来源 repo 有 `github_license` 字段 |

使用 Skill-Usage 时要注意：

- skill pool 来自多个 GitHub repositories；
- 每条 skill metadata 有 `github_license`；
- 如果公开分发处理后的数据，应保留来源和 license 字段；
- 项目内部实验使用一般问题较小，但公开发布需谨慎。

---

## 8. 推荐下一步

### Step 1：生成统一 skill catalog

从 Skill-Usage 的 `skills_meta.jsonl` 和 `skills/` 目录生成：

```text
data/processed/skill_usage/skills_catalog.jsonl
```

### Step 2：整理 query 和 qrels

使用：

```text
data/raw/Skill-Usage/data/task_queries.json
data/raw/Skill-Usage/data/task_skill_mapping.json
```

生成：

```text
data/processed/skill_usage/queries.jsonl
data/processed/skill_usage/qrels.jsonl
```

### Step 3：实现 candidate pool sampler

支持：

- random；
- same-owner；
- same-repo；
- lexical-overlap；
- semantic-near。

### Step 4：先跑 BM25 baseline

优先完成：

- pool size: 10, 50, 100, 500, 1000；
- distractor modes: random, semantic-near；
- metrics: Top-1 Accuracy, Recall@10, MRR, NDCG@10。

### Step 5：再加 semantic/hybrid/reranker

在 BM25 baseline 跑通后，再加入：

- Skill-Usage search server semantic endpoint；
- hybrid endpoint；
- reranker top-50。

---

## 9. 可复制的下载命令

### SWE-Skills-Bench

```bash
git clone --depth 1 https://github.com/GeniusHTX/SWE-Skills-Bench.git data/raw/SWE-Skills-Bench
```

### Skill-Usage

```bash
git clone --depth 1 https://github.com/UCSB-NLP-Chang/Skill-Usage.git data/raw/Skill-Usage

hf download Shiyu-Lab/Skill-Usage \
  skills-34k/skills.zip \
  skills-34k/skills_meta.jsonl \
  --repo-type dataset \
  --local-dir data/raw/Skill-Usage

unzip -q data/raw/Skill-Usage/skills-34k/skills.zip \
  -d data/raw/Skill-Usage/skills

hf download Shiyu-Lab/Skill-Usage \
  search_index/search_index.zip \
  --repo-type dataset \
  --local-dir data/raw/Skill-Usage

mkdir -p data/raw/Skill-Usage/search_server/index
unzip -q data/raw/Skill-Usage/search_index/search_index.zip \
  -d data/raw/Skill-Usage/search_server/index
```

### SkillsBench

```bash
git clone --depth 1 https://github.com/benchflow-ai/skillsbench.git data/raw/skillsbench
```

---

## 10. 一句话建议

第一轮实验先不要等待 SkillRet。当前最稳的执行路线是：

> 用 Skill-Usage 的 34k real-world skill pool 做 retrieval scaling 主实验，用 SkillsBench / SWE-Skills-Bench 做小规模 downstream validation；同时继续跟进 SkillRet、SRA-Bench、ToolRet 的正式数据发布。

