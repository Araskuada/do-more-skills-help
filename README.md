# Do More Skills Help?

This repository contains our course project materials for:

**Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents**

The project studies whether larger skill libraries actually help LLM agents, or whether larger libraries introduce retrieval noise, skill competition, and context pollution.

## Research Questions

1. **RQ1:** Does skill retrieval accuracy decrease as skill library size grows?
2. **RQ2:** Which distractor type is most likely to make an agent select the wrong skill?
3. **RQ3:** How do different retrievers behave under large-scale skill libraries?
4. **RQ4:** Does retrieving the correct skill always improve downstream task performance?
5. **RQ5:** Does exposing more skill descriptions cause context pollution?

## Current Pilot Experiment

We completed a first retrieval-scaling pilot using the Skill-Usage dataset.

- Tasks: 87
- Candidate pool sizes: 10, 50, 100, 500, 1000, 5000, 10000, full
- Retriever: lightweight local BM25 over skill name + description
- Metrics: Top-1 Accuracy, Recall@3, Recall@5, Recall@10, MRR, NDCG@10

Main result:

- Top-1 Accuracy drops from **0.963** at pool size 10 to **0.414** at full library.
- Recall@10 drops from **0.880** to **0.449**.

This supports our initial hypothesis that larger skill libraries can make correct skill retrieval substantially harder.

## Repository Structure

```text
docs/
  do_more_skills_help_formal_proposal.md
  teacher_progress_report_2026-07-08.md
  first_experiment_retrieval_scaling_pilot.md
  data_usage_guide.md
  project_data_inventory.md

experiments/
  retrieval_scaling_pilot.py

data/experiments/
  retrieval_scaling_pilot/
    summary.csv
    summary.json
    per_query_metrics.csv
    ranking_examples.json
```

## Data

Raw datasets are intentionally not committed because they are large. The local project used:

- Skill-Usage
- SkillsBench
- SWE-Skills-Bench

See `docs/data_usage_guide.md` and `docs/project_data_inventory.md` for download notes and expected local paths.

## Reproducing the Pilot

After downloading the raw Skill-Usage data to `data/raw/Skill-Usage`, run:

```bash
python3 experiments/retrieval_scaling_pilot.py
```

The script writes results to:

```text
data/experiments/retrieval_scaling_pilot/
```
