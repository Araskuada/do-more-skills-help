#!/usr/bin/env python3
"""Print the gold-skill sparsity audit used to motivate Question 8."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    path = workspace / "data" / "raw" / "Skill-Usage" / "data" / "task_skill_mapping.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = {task: [f"benchflow-ai--{sid}" for sid in skills] for task, skills in raw.items()}
    skill_freq = Counter(s for skills in mapping.values() for s in skills)
    pair_freq = Counter(
        pair
        for skills in mapping.values()
        for pair in itertools.combinations(sorted(set(skills)), 2)
    )
    result = {
        "tasks": len(mapping),
        "multi_skill_tasks": sum(len(v) > 1 for v in mapping.values()),
        "unique_gold_skills": len(skill_freq),
        "gold_assignments": sum(skill_freq.values()),
        "singleton_gold_skills": sum(v == 1 for v in skill_freq.values()),
        "repeated_gold_skills": sum(v > 1 for v in skill_freq.values()),
        "co_required_pairs": len(pair_freq),
        "repeated_co_required_pairs": sum(v > 1 for v in pair_freq.values()),
        "leave_one_task_out_tasks_with_any_reusable_pair": sum(
            any(pair_freq[p] > 1 for p in itertools.combinations(sorted(set(skills)), 2))
            for skills in mapping.values() if len(set(skills)) > 1
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
