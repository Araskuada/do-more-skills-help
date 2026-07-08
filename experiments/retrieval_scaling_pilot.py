#!/usr/bin/env python3
"""Run a small retrieval-scaling pilot over the Skill-Usage skill pool.

This is a dependency-light first experiment. It uses task queries and curated
skill mappings from Skill-Usage, samples random distractors at several candidate
pool sizes, ranks candidates with a local BM25 implementation over skill name
and description, and writes aggregate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
GT_OWNER = "benchflow-ai"
DEFAULT_POOL_SIZES = ["10", "50", "100", "500", "1000", "5000", "10000", "full"]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_json(path: Path):
    return json.loads(path.read_text())


def load_skill_docs(meta_path: Path) -> dict[str, dict]:
    docs = {}
    with meta_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("skill_id")
            if not sid:
                continue
            name = row.get("name") or row.get("skill_name") or row.get("skillId") or ""
            desc = row.get("description") or ""
            text = f"{name} {desc}"
            tokens = tokenize(text)
            docs[sid] = {
                "skill_id": sid,
                "name": name,
                "owner": row.get("owner", ""),
                "repo": row.get("repo", ""),
                "description": desc,
                "tokens": tokens,
                "tf": Counter(tokens),
                "length": len(tokens),
            }
    return docs


def normalize_gt(raw_mapping: dict[str, list[str]]) -> dict[str, set[str]]:
    return {
        task: {f"{GT_OWNER}--{skill}" for skill in skills}
        for task, skills in raw_mapping.items()
    }


def bm25_rank(query: str, candidate_ids: list[str], docs: dict[str, dict], top_k: int) -> list[dict]:
    query_terms = tokenize(query)
    if not query_terms or not candidate_ids:
        return []

    n_docs = len(candidate_ids)
    doc_freq = Counter()
    lengths = []
    for sid in candidate_ids:
        doc = docs[sid]
        lengths.append(doc["length"])
        for term in set(doc["tokens"]):
            doc_freq[term] += 1

    avg_len = sum(lengths) / len(lengths) if lengths else 1.0
    k1 = 1.5
    b = 0.75
    scores = []
    for sid in candidate_ids:
        doc = docs[sid]
        doc_len = doc["length"] or 1
        tf = doc["tf"]
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            scores.append((score, sid))

    scores.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "rank": rank,
            "skill_id": sid,
            "score": score,
            "name": docs[sid]["name"],
        }
        for rank, (score, sid) in enumerate(scores[:top_k], start=1)
    ]


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    for index, sid in enumerate(ranked_ids, start=1):
        if sid in gold:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for rank, sid in enumerate(ranked_ids[:k], start=1):
        if sid in gold:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def parse_pool_size(value: str, total: int) -> int:
    return total if value == "full" else int(value)


def sample_candidates(
    all_skill_ids: list[str],
    gold: set[str],
    pool_size: int,
    rng: random.Random,
) -> list[str]:
    gold_present = sorted(sid for sid in gold if sid in all_skill_ids)
    target_size = max(pool_size, len(gold_present))
    if target_size >= len(all_skill_ids):
        return list(all_skill_ids)

    gold_lookup = set(gold_present)
    distractor_count = target_size - len(gold_present)
    non_gold = [sid for sid in all_skill_ids if sid not in gold_lookup]
    distractors = rng.sample(non_gold, distractor_count)
    candidates = gold_present + distractors
    rng.shuffle(candidates)
    return candidates


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    keys = ["top1_accuracy", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]
    return {key: sum(row[key] for row in rows) / n for key in keys} | {"n": n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--output-dir", default="data/experiments/retrieval_scaling_pilot")
    parser.add_argument("--pool-sizes", nargs="+", default=DEFAULT_POOL_SIZES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.skill_usage_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_json(root / "data" / "task_queries.json")
    gt = normalize_gt(load_json(root / "data" / "task_skill_mapping.json"))
    docs = load_skill_docs(root / "skills-34k" / "skills_meta.jsonl")

    all_skill_ids = sorted(docs)
    tasks = sorted(set(queries) & set(gt))
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    per_query_rows = []
    ranking_examples = {}
    summary_rows = []

    for pool_label in args.pool_sizes:
        pool_size = parse_pool_size(pool_label, len(all_skill_ids))
        repeat_count = 1 if pool_label == "full" else args.repeats
        pooled_rows = []

        for repeat in range(repeat_count):
            rng = random.Random(args.seed + repeat + pool_size * 1009)
            for task in tasks:
                gold = gt[task]
                query = " ".join(queries[task])
                candidates = sample_candidates(all_skill_ids, gold, pool_size, rng)
                ranked = bm25_rank(query, candidates, docs, args.top_k)
                ranked_ids = [row["skill_id"] for row in ranked]

                row = {
                    "task": task,
                    "pool_size": pool_label,
                    "actual_pool_size": len(candidates),
                    "repeat": repeat,
                    "gold_count": len(gold),
                    "top1_accuracy": 1.0 if ranked_ids and ranked_ids[0] in gold else 0.0,
                    "recall@3": recall_at_k(ranked_ids, gold, 3),
                    "recall@5": recall_at_k(ranked_ids, gold, 5),
                    "recall@10": recall_at_k(ranked_ids, gold, 10),
                    "mrr": reciprocal_rank(ranked_ids, gold),
                    "ndcg@10": ndcg_at_k(ranked_ids, gold, 10),
                }
                per_query_rows.append(row)
                pooled_rows.append(row)

                example_key = f"{pool_label}:{repeat}:{task}"
                if repeat == 0 and len(ranking_examples) < 20:
                    ranking_examples[example_key] = {
                        "query": query,
                        "gold": sorted(gold),
                        "top_results": ranked,
                    }

        metrics = summarize(pooled_rows)
        summary_rows.append({
            "pool_size": pool_label,
            "tasks": len(tasks),
            "repeats": repeat_count,
            **metrics,
        })

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    per_query_csv = output_dir / "per_query_metrics.csv"
    examples_json = output_dir / "ranking_examples.json"

    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")
    examples_json.write_text(json.dumps(ranking_examples, indent=2) + "\n")

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with per_query_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0]))
        writer.writeheader()
        writer.writerows(per_query_rows)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_query_csv}")
    print(f"Wrote {examples_json}")
    print()
    print(f"{'pool':>8} {'top1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'NDCG@10':>10}")
    for row in summary_rows:
        print(
            f"{row['pool_size']:>8} "
            f"{row['top1_accuracy']:>8.3f} "
            f"{row['recall@3']:>8.3f} "
            f"{row['recall@5']:>8.3f} "
            f"{row['recall@10']:>8.3f} "
            f"{row['mrr']:>8.3f} "
            f"{row['ndcg@10']:>10.3f}"
        )


if __name__ == "__main__":
    main()
