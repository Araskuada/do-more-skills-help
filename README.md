# Do More Skills Help?

Course project repository for:

**Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents**

The project studies whether larger skill libraries actually help LLM agents, or whether larger libraries introduce retrieval noise, skill competition, underloaded skill context, and context pollution.

## Research Questions

1. **RQ1:** Does skill retrieval accuracy decrease as skill library size grows?
2. **RQ2:** Which distractor type is most likely to make an agent select the wrong skill?
3. **RQ3:** How do different retrievers behave under large-scale skill libraries?
4. **RQ4:** Does better skill exposure improve downstream task readiness?

RQ5 remains a future extension and is not included as a final experiment in this repository.

## Final Experiment Set

| RQ | Final script | Final analysis | Output directory |
|---|---|---|---|
| RQ1 | `experiments/rq1_retrieval_scaling.py` | `docs/rq1_retrieval_scaling_analysis_2026-07-09.md` | `data/experiments/rq1_retrieval_scaling/` |
| RQ2 | `experiments/rq2_distractor_types.py` | `docs/rq2_distractor_type_analysis_2026-07-09.md` | `data/experiments/rq2_distractor_types/` |
| RQ3 | `experiments/rq3_retriever_enhanced.py` | `docs/rq3_retriever_enhanced_analysis_2026-07-09.md` | `data/experiments/rq3_retriever_enhanced/` |
| RQ4 | `experiments/rq4_adaptive_qwen_experiment.py` | `docs/rq4_adaptive_qwen_analysis_2026-07-15.md` | `data/experiments/rq4_adaptive_qwen/` |

Supporting project documents:

- `docs/do_more_skills_help_formal_proposal.md`
- `docs/data_usage_guide.md`
- `docs/project_data_inventory.md`

## Main Results

### RQ1

Using Skill-Usage with a 34,396-skill library, BM25 retrieval accuracy drops as the candidate library grows:

- Top-1 Accuracy: **0.964** at pool size 10 to **0.414** at full library.
- Hit@10: **1.000** to **0.667**.
- Recall@10: **1.000** to **0.449**.

### RQ2

Distractor type matters. At pool size 100:

- Random distractors: **0.889** Top-1 Accuracy.
- Query-overlap distractors: **0.379**.
- BM25-hard distractors: **0.425**.
- Embedding-semantic-near distractors: **0.540**.
- Gold-skill-near distractors: **0.563**.

### RQ3

The enhanced retriever experiment compares sparse, dense, hybrid, and full-skill-document retrieval:

- Hybrid BM25+MiniLM: **0.460** Top-1, **0.724** Hit@10.
- BM25 full `SKILL.md`: **0.437** Top-1, **0.759** Hit@10.
- BM25 description-only: **0.425** Top-1, **0.667** Hit@10.
- MiniLM dense: **0.414** Top-1, **0.701** Hit@10.

Hybrid retrieval helps, but longer full skill documents require careful retrieval design.

### RQ4

The final RQ4 implementation uses a paired Qwen-judged downstream-readiness design. It compares the same SkillsBench task under:

- `no_skill`
- `oracle_gold_all`
- `bm25_top1`
- `hybrid_bm25_neural_top10`
- `oracle_gold_plus_5_noise`

The current partial pilot completed `video-silence-remover` before local Alibaba Cloud quota ended:

| Condition | Readiness | Likely Pass | Delta vs No Skill |
|---|---:|---:|---:|
| `no_skill` | 6 | false | 0 |
| `oracle_gold_all` | 10 | true | +4 |
| `bm25_top1` | 5 | false | -1 |
| `hybrid_bm25_neural_top10` | 7 | false | +1 |
| `oracle_gold_plus_5_noise` | 10 | true | +4 |

The result supports the RQ4 mechanism: complete gold-skill context improves downstream readiness, while Top-1 retrieval can underload multi-skill tasks.

## Repository Structure

```text
docs/
  do_more_skills_help_formal_proposal.md
  data_usage_guide.md
  project_data_inventory.md
  rq1_retrieval_scaling_analysis_2026-07-09.md
  rq2_distractor_type_analysis_2026-07-09.md
  rq3_retriever_enhanced_analysis_2026-07-09.md
  rq4_adaptive_qwen_analysis_2026-07-15.md

experiments/
  rq1_retrieval_scaling.py
  rq2_distractor_types.py
  rq3_retriever_comparison.py      # shared retriever helper functions
  rq3_retriever_enhanced.py
  rq4_downstream_skill_exposure.py # supporting exposure table for RQ4
  rq4_adaptive_qwen_experiment.py

data/experiments/
  rq1_retrieval_scaling/
  rq2_distractor_types/
  rq3_retriever_enhanced/
  rq4_downstream_skill_exposure/   # required input for adaptive RQ4
  rq4_adaptive_qwen/
```

## Data

Raw datasets are intentionally not committed because they are large.

| Dataset | Source | Expected local path | Project use |
|---|---|---|---|
| Skill-Usage | `https://github.com/UCSB-NLP-Chang/Skill-Usage` | `data/raw/Skill-Usage` | RQ1-RQ3 |
| SkillsBench | `https://github.com/benchflow-ai/skillsbench` | `data/raw/skillsbench` | RQ4 |

See `docs/data_usage_guide.md` and `docs/project_data_inventory.md` for expected paths and setup notes.

## Reproducing Final Experiments

RQ1:

```bash
python3 experiments/rq1_retrieval_scaling.py
```

RQ2:

```bash
python3 experiments/rq2_distractor_types.py
```

RQ3:

```bash
python3 experiments/rq3_retriever_enhanced.py
```

RQ4 first requires the exposure table:

```bash
python3 experiments/rq4_downstream_skill_exposure.py
```

Then dry-run the adaptive Qwen plan:

```bash
python3 experiments/rq4_adaptive_qwen_experiment.py \
  --dry-run \
  --max-tasks 6 \
  --token-budget 250000 \
  --model qwen3.6-flash
```

Run or resume the paid Qwen pilot:

```bash
python3 experiments/rq4_adaptive_qwen_experiment.py \
  --max-tasks 6 \
  --token-budget 250000 \
  --model qwen3.6-flash \
  --resume \
  --api-key-prompt \
  --base-url "https://YOUR_WORKSPACE_ID.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
```

Do not store API keys in files or commit them.
