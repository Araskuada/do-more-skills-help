# RQ4 Adaptive Qwen Downstream Readiness Experiment

**Date**: 2026-07-15  
**Final script**: `experiments/rq4_adaptive_qwen_experiment.py`  
**Output directory**: `data/experiments/rq4_adaptive_qwen/`  
**Recommended model**: `qwen3.6-flash`

## 1. Research Question

RQ4 asks whether better skill exposure improves downstream task performance.

The final RQ4 implementation uses a paired downstream-readiness design instead of a row-by-row pass/fail proxy. For each selected SkillsBench task, Qwen solves the same task under multiple skill-exposure conditions, and a task-level judge compares all condition outputs together.

This is still an LLM-judged proxy, not an execution-based Docker verifier pass rate. It is designed to be cheaper and more diagnostic than a full agent run while measuring downstream task quality more directly than retrieval exposure alone.

## 2. Experimental Design

The experiment has four stages:

1. Load SkillsBench tasks and the existing RQ4 exposure table from `data/experiments/rq4_downstream_skill_exposure/per_task_exposure.csv`.
2. Select high-information tasks where skill coverage differs across conditions.
3. Ask Qwen to produce one concrete execution plan per condition.
4. Ask Qwen to judge all condition plans for the same task in one calibrated comparison.

The judge returns:

- `readiness_score`: 0-10 execution-readiness score;
- `likely_pass`: conservative judged pass proxy;
- `delta_vs_no_skill`: improvement over the no-skill condition;
- `best_condition`: best plan for the task;
- `major_gaps`: verifier-relevant missing details.

## 3. Conditions

| Condition | Meaning |
|---|---|
| `no_skill` | No skill context is shown. |
| `oracle_gold_all` | All curated gold skills for the task are shown. |
| `bm25_top1` | Only the top BM25 retrieved skill is shown. |
| `hybrid_bm25_neural_top10` | Top-10 hybrid retrieved skills are shown. |
| `oracle_gold_plus_5_noise` | All gold skills plus 5 unrelated noisy skills are shown. |

These conditions test whether complete gold skills help, whether Top-1 retrieval underloads multi-skill tasks, whether Top-10 retrieval recovers enough task context, and whether noisy context dilutes otherwise useful gold skills.

## 4. Task Selection

The script automatically ranks tasks by signal value. It prefers tasks with:

- multiple gold skills;
- Top-1 underload;
- retrieval coverage variation across conditions;
- noisy-context stress-test value.

The current 6-task plan is:

| Task | Gold Skills | Selection Reason |
|---|---:|---|
| `video-silence-remover` | 7 | Top-1 underload, retrieval coverage varies, noisy context test |
| `travel-planning` | 6 | Top-1 underload, retrieval coverage varies, noisy context test |
| `drone-planning-control` | 6 | Top-1 underload, retrieval coverage varies, noisy context test |
| `fix-erlang-ssh-cve` | 6 | Top-1 underload, retrieval coverage varies, noisy context test |
| `multilingual-video-dubbing` | 6 | Top-1 underload, retrieval coverage varies, noisy context test |
| `python-scala-translation` | 6 | Top-1 underload, retrieval coverage varies, noisy context test |

Dry-run estimate:

```bash
python3 experiments/rq4_adaptive_qwen_experiment.py \
  --dry-run \
  --max-tasks 6 \
  --token-budget 250000 \
  --model qwen3.6-flash
```

Estimated total: about **174k tokens**. Real usage can be higher because the model may emit longer completions.

## 5. Running With Alibaba Cloud Bailian

Use Model Studio / Bailian API keys, not RAM AccessKey.

For Japan Tokyo workspaces, the OpenAI-compatible endpoint has this shape:

```text
https://{WORKSPACE_ID}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions
```

Run or resume:

```bash
python3 experiments/rq4_adaptive_qwen_experiment.py \
  --max-tasks 6 \
  --token-budget 250000 \
  --model qwen3.6-flash \
  --resume \
  --api-key-prompt \
  --base-url "https://YOUR_WORKSPACE_ID.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
```

The API key is read with hidden terminal input and is never written to the repository.

## 6. Output Files

| File | Purpose |
|---|---|
| `experiment_plan.csv` | Selected tasks and estimated tokens. |
| `experiment_plan.json` | Machine-readable plan and metadata. |
| `solutions.jsonl` | Solver output for each task-condition. |
| `judged_tasks.jsonl` | Task-level judge output comparing all conditions. |
| `per_condition_results.csv` | Flattened condition-level results. |
| `summary.csv` | Main aggregate table by condition. |
| `summary.json` | Summary plus metadata and best-condition counts. |
| `readiness_score_bar.svg` | Visualization of mean readiness score. |

## 7. Current Partial Result

The local run stopped when the available Alibaba Cloud quota/purchase permission ended. One task was completely judged: `video-silence-remover`.

| Condition | Readiness | Likely Pass | Delta vs No Skill |
|---|---:|---:|---:|
| `no_skill` | 6 | false | 0 |
| `oracle_gold_all` | 10 | true | +4 |
| `bm25_top1` | 5 | false | -1 |
| `hybrid_bm25_neural_top10` | 7 | false | +1 |
| `oracle_gold_plus_5_noise` | 10 | true | +4 |

Interpretation:

- Complete gold-skill exposure substantially improves downstream readiness.
- A single retrieved skill is insufficient for this multi-skill task.
- Hybrid Top-10 improves over Top-1 but still misses part of the calibrated oracle pipeline.
- Adding five noisy skills does not hurt this task when all gold skills remain present.

This is an early pilot signal, not a full RQ4 conclusion. The 6-task run should be completed by resuming the script with a working Bailian quota.

## 8. Error Handling

The script stops cleanly and keeps completed outputs when it sees quota or access errors such as:

- `AccessDenied.Unpurchased`
- `free quota has been exhausted`
- HTTP `403`

Rerun the same command with `--resume` after quota or paid access is available.

