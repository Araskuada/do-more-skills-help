#!/usr/bin/env python3
"""Run the enhanced RQ3 retriever experiment.

This addendum addresses the main limitations from the first RQ3 run:

- adds a true cached neural dense retriever using sentence-transformers/all-MiniLM-L6-v2;
- adds neural + BM25 hybrid retrieval;
- adds full SKILL.md content retrieval for BM25 and TF-IDF;
- tests retrievers under RQ2-style hard distractor settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from rq1_retrieval_scaling import (
    GT_OWNER,
    SUMMARY_METRICS,
    first_gold_rank_at_k,
    hit_at_k,
    load_json,
    load_skill_docs,
    ndcg_at_k,
    normalize_gt,
    parse_pool_size,
    recall_at_k,
    reciprocal_rank_at_k,
    sample_candidates,
    tokenize,
)
from rq2_distractor_types import (
    load_embedding_index,
    rank_by_embedding_semantic_near,
    rank_by_query_overlap,
)
from rq3_retriever_comparison import (
    bm25_full_scores,
    rank_candidates,
    reciprocal_rank_fusion,
)


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_RETRIEVERS = [
    "bm25_desc",
    "tfidf_desc",
    "neural_minilm_desc",
    "hybrid_bm25_neural",
    "bm25_full_skill",
    "tfidf_full_skill",
]
DEFAULT_SETTINGS = ["random:1000", "random:full", "query_overlap:100", "embedding_semantic_near:100"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def skill_desc_text(doc: dict) -> str:
    return f"{doc['name']} {doc['description']}"


def sanitize_fts5_query(query: str) -> str:
    # FTS5 treats whitespace-separated tokens as AND by default, which is too
    # strict for natural-language task queries. We explicitly OR lexical tokens
    # so full-document BM25 is comparable to the bag-of-words retrievers above.
    tokens = tokenize(query)
    if not tokens:
        return '""'
    return " OR ".join(tokens)


def load_full_skill_content(db_path: Path) -> dict[str, str]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT skill_id, name, description, skill_md_content FROM skills").fetchall()
    con.close()
    return {
        row["skill_id"]: f"{row['name']} {row['description']} {row['skill_md_content'] or ''}"
        for row in rows
    }


def sqlite_bm25_full_skill_scores(query: str, db_path: Path, skill_id_to_index: dict[str, int]) -> np.ndarray:
    scores = np.zeros(len(skill_id_to_index), dtype=np.float32)
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT s.skill_id, bm25(skills_fts, 10.0, 5.0, 1.0) AS score
        FROM skills_fts f
        JOIN skills s ON s.rowid = f.rowid
        WHERE skills_fts MATCH ?
        ORDER BY score
        """,
        (sanitize_fts5_query(query),),
    ).fetchall()
    con.close()
    for sid, score in rows:
        index = skill_id_to_index.get(sid)
        if index is not None:
            scores[index] = -float(score)
    return scores


def parse_setting(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Setting must be distractor_type:pool_size, got {value}")
    distractor_type, pool_size = value.split(":", 1)
    return distractor_type, pool_size


def sample_setting_candidates(
    distractor_type: str,
    pool_label: str,
    task: str,
    gold: set[str],
    all_skill_ids: list[str],
    docs: dict[str, dict],
    ranked_distractors: dict[str, dict[str, list[str]]],
    rng: random.Random,
) -> tuple[list[str], int, float]:
    pool_size = parse_pool_size(pool_label, len(all_skill_ids))
    if distractor_type == "random":
        candidates = sample_candidates(all_skill_ids, gold, pool_size, rng)
        return candidates, 0, 0.0

    gold_present = sorted(sid for sid in gold if sid in docs)
    target_size = max(pool_size, len(gold_present))
    distractor_count = target_size - len(gold_present)
    ranked = ranked_distractors[task][distractor_type]
    selected = ranked[:distractor_count]
    fallback_count = distractor_count - len(selected)
    if fallback_count:
        selected_lookup = set(selected)
        gold_lookup = set(gold_present)
        fallback_pool = [
            sid
            for sid in all_skill_ids
            if sid not in selected_lookup and sid not in gold_lookup
        ]
        selected.extend(rng.sample(fallback_pool, fallback_count))
    candidates = gold_present + selected
    rng.shuffle(candidates)
    purity = (distractor_count - fallback_count) / distractor_count if distractor_count else 1.0
    return candidates, fallback_count, purity


def summarize(rows: list[dict]) -> dict:
    summary = {key: mean(row[key] for row in rows) for key in SUMMARY_METRICS}
    summary["top1_error_rate"] = 1.0 - summary["top1_accuracy"]
    summary["mean_fallback_distractors"] = mean(row["fallback_distractors"] for row in rows)
    summary["mean_hard_negative_purity"] = mean(row["hard_negative_purity"] for row in rows)
    summary["n"] = len(rows)
    return summary


def metric_stdev(rows: list[dict], metric: str) -> float:
    values = [row[metric] for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def encode_neural_texts(model_name: str, texts: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_neural_texts_with_model(model, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--output-dir", default="data/experiments/rq3_retriever_enhanced")
    parser.add_argument("--retrievers", nargs="+", default=DEFAULT_RETRIEVERS)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neural-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--neural-batch-size", type=int, default=128)
    parser.add_argument("--tfidf-content-max-features", type=int, default=200000)
    parser.add_argument(
        "--neural-doc-cache",
        default="",
        help="Optional .npy cache for neural document embeddings. Defaults to output-dir/neural_doc_embeddings.npy",
    )
    args = parser.parse_args()

    root = Path(args.skill_usage_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_db_path = root / "search_server" / "index" / "skills_full.db"

    queries = load_json(root / "data" / "task_queries.json")
    gt = normalize_gt(load_json(root / "data" / "task_skill_mapping.json"))
    docs = load_skill_docs(root / "skills-34k" / "skills_meta.jsonl")
    all_skill_ids = sorted(docs)
    skill_id_to_index = {sid: index for index, sid in enumerate(all_skill_ids)}
    desc_texts = [skill_desc_text(docs[sid]) for sid in all_skill_ids]
    full_content = load_full_skill_content(full_db_path)
    full_texts = [full_content[sid] for sid in all_skill_ids]
    tasks = sorted(set(queries) & set(gt))
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    build_times = {}
    start = time.perf_counter()
    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True)
    counts = count_vectorizer.fit_transform(desc_texts).astype(np.float32)
    counts_csc = counts.tocsc()
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    doc_lengths[doc_lengths == 0] = 1.0
    doc_freq = np.diff(counts_csc.indptr).astype(np.float32)
    avg_doc_len = float(doc_lengths.mean())
    build_times["bm25_desc"] = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_desc_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True, norm="l2")
    tfidf_desc_matrix = tfidf_desc_vectorizer.fit_transform(desc_texts).astype(np.float32)
    build_times["tfidf_desc"] = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_full_vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
        lowercase=True,
        norm="l2",
        max_features=args.tfidf_content_max_features,
    )
    tfidf_full_matrix = tfidf_full_vectorizer.fit_transform(full_texts).astype(np.float32)
    build_times["tfidf_full_skill"] = time.perf_counter() - start
    build_times["bm25_full_skill"] = 0.0

    from sentence_transformers import SentenceTransformer

    neural_model = SentenceTransformer(args.neural_model_name, local_files_only=True)
    neural_doc_cache = Path(args.neural_doc_cache) if args.neural_doc_cache else output_dir / "neural_doc_embeddings.npy"

    start = time.perf_counter()
    if neural_doc_cache.exists():
        neural_doc_embeddings = np.load(neural_doc_cache).astype(np.float32)
    else:
        neural_doc_embeddings = encode_neural_texts_with_model(
            neural_model,
            desc_texts,
            args.neural_batch_size,
        )
        np.save(neural_doc_cache, neural_doc_embeddings)
    build_times["neural_minilm_desc"] = time.perf_counter() - start
    build_times["hybrid_bm25_neural"] = build_times["bm25_desc"] + build_times["neural_minilm_desc"]

    embedding_index = root / "search_server" / "index"
    official_embeddings, official_skill_id_to_index, official_index_skill_ids = load_embedding_index(
        embedding_index,
        root / "skills-34k" / "skills_meta.jsonl",
    )
    max_pool_size = max(parse_pool_size(pool, len(all_skill_ids)) for _, pool in map(parse_setting, args.settings))
    ranked_distractors = {}
    for task in tasks:
        gold = gt[task]
        query = " ".join(queries[task])
        gold_present = sorted(sid for sid in gold if sid in docs)
        non_gold_ids = [sid for sid in all_skill_ids if sid not in set(gold_present)]
        max_needed = max_pool_size - len(gold_present)
        ranked_distractors[task] = {
            "query_overlap": rank_by_query_overlap(query, non_gold_ids, docs)[:max_needed],
            "embedding_semantic_near": rank_by_embedding_semantic_near(
                gold,
                non_gold_ids,
                official_embeddings,
                official_skill_id_to_index,
                official_index_skill_ids,
            )[:max_needed],
        }

    task_scores = {}
    score_times = Counter()
    for task in tasks:
        query = " ".join(queries[task])
        task_scores[task] = {}

        start = time.perf_counter()
        task_scores[task]["bm25_desc"] = bm25_full_scores(
            query,
            count_vectorizer,
            counts_csc,
            doc_lengths,
            doc_freq,
            avg_doc_len,
        )
        score_times["bm25_desc"] += time.perf_counter() - start

        start = time.perf_counter()
        q_desc = tfidf_desc_vectorizer.transform([query]).astype(np.float32)
        task_scores[task]["tfidf_desc"] = np.asarray((tfidf_desc_matrix @ q_desc.T).todense()).ravel().astype(np.float32)
        score_times["tfidf_desc"] += time.perf_counter() - start

        start = time.perf_counter()
        q_full = tfidf_full_vectorizer.transform([query]).astype(np.float32)
        task_scores[task]["tfidf_full_skill"] = np.asarray((tfidf_full_matrix @ q_full.T).todense()).ravel().astype(np.float32)
        score_times["tfidf_full_skill"] += time.perf_counter() - start

        start = time.perf_counter()
        task_scores[task]["bm25_full_skill"] = sqlite_bm25_full_skill_scores(query, full_db_path, skill_id_to_index)
        score_times["bm25_full_skill"] += time.perf_counter() - start

    start = time.perf_counter()
    query_texts = [" ".join(queries[task]) for task in tasks]
    neural_query_embeddings = encode_neural_texts_with_model(
        neural_model,
        query_texts,
        args.neural_batch_size,
    )
    neural_query_by_task = dict(zip(tasks, neural_query_embeddings))
    score_times["neural_minilm_desc"] = time.perf_counter() - start
    for task in tasks:
        task_scores[task]["neural_minilm_desc"] = neural_doc_embeddings @ neural_query_by_task[task]
    score_times["hybrid_bm25_neural"] = score_times["bm25_desc"] + score_times["neural_minilm_desc"]

    per_query_rows = []
    summary_rows = []
    for retriever in args.retrievers:
        for setting in args.settings:
            distractor_type, pool_label = parse_setting(setting)
            pool_size = parse_pool_size(pool_label, len(all_skill_ids))
            repeat_count = 1 if pool_label == "full" else args.repeats
            pooled_rows = []
            for repeat in range(repeat_count):
                rng = random.Random(args.seed + repeat + pool_size * 1009 + len(retriever) * 271 + len(distractor_type) * 991)
                for task in tasks:
                    gold = gt[task]
                    candidates, fallback_count, purity = sample_setting_candidates(
                        distractor_type,
                        pool_label,
                        task,
                        gold,
                        all_skill_ids,
                        docs,
                        ranked_distractors,
                        rng,
                    )
                    scores = task_scores[task]
                    if retriever == "hybrid_bm25_neural":
                        ranked_ids = reciprocal_rank_fusion(
                            scores["bm25_desc"],
                            scores["neural_minilm_desc"],
                            candidates,
                            skill_id_to_index,
                            args.top_k,
                        )
                    else:
                        ranked_ids = rank_candidates(scores[retriever], candidates, skill_id_to_index, args.top_k)

                    row = {
                        "task": task,
                        "retriever": retriever,
                        "distractor_type": distractor_type,
                        "pool_size": pool_label,
                        "actual_pool_size": len(candidates),
                        "repeat": repeat,
                        "gold_count": len(gold),
                        "fallback_distractors": fallback_count,
                        "hard_negative_purity": purity,
                        "top1_accuracy": 1.0 if ranked_ids and ranked_ids[0] in gold else 0.0,
                        "hit@3": hit_at_k(ranked_ids, gold, 3),
                        "hit@5": hit_at_k(ranked_ids, gold, 5),
                        "hit@10": hit_at_k(ranked_ids, gold, 10),
                        "recall@3": recall_at_k(ranked_ids, gold, 3),
                        "recall@5": recall_at_k(ranked_ids, gold, 5),
                        "recall@10": recall_at_k(ranked_ids, gold, 10),
                        "mrr@10": reciprocal_rank_at_k(ranked_ids, gold, args.top_k),
                        "ndcg@10": ndcg_at_k(ranked_ids, gold, 10),
                        "first_gold_rank@10": first_gold_rank_at_k(ranked_ids, gold, args.top_k),
                    }
                    per_query_rows.append(row)
                    pooled_rows.append(row)

            summary = summarize(pooled_rows)
            summary_row = {
                "retriever": retriever,
                "distractor_type": distractor_type,
                "pool_size": pool_label,
                "tasks": len(tasks),
                "repeats": repeat_count,
                "build_seconds": build_times.get(retriever, 0.0),
                "score_seconds_total": score_times.get(retriever, 0.0),
                "score_seconds_per_query": score_times.get(retriever, 0.0) / len(tasks),
                **summary,
            }
            for metric in SUMMARY_METRICS:
                summary_row[f"{metric}_std_across_queries"] = metric_stdev(pooled_rows, metric)
            summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: (row["retriever"], row["distractor_type"], row["pool_size"]))
    metadata = {
        "neural_model_name": args.neural_model_name,
        "full_skill_db": str(full_db_path),
        "settings": args.settings,
        "retrievers": {
            "bm25_desc": "BM25 over skill name + description.",
            "tfidf_desc": "TF-IDF cosine over skill name + description.",
            "neural_minilm_desc": "Cached all-MiniLM-L6-v2 dense retriever over skill name + description.",
            "hybrid_bm25_neural": "Reciprocal-rank fusion of BM25 description and MiniLM dense scores.",
            "bm25_full_skill": "SQLite FTS5 BM25 over name + description + full SKILL.md content.",
            "tfidf_full_skill": "TF-IDF cosine over name + description + full SKILL.md content.",
        },
    }

    (output_dir / "summary.json").write_text(json.dumps({"metadata": metadata, "summary": summary_rows}, indent=2) + "\n")
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_query_metrics.csv", per_query_rows)

    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'per_query_metrics.csv'}")
    print()
    print(f"{'retriever':>22} {'setting':>28} {'top1':>8} {'hit@10':>8} {'mrr':>8}")
    for row in summary_rows:
        setting = f"{row['distractor_type']}:{row['pool_size']}"
        print(
            f"{row['retriever']:>22} {setting:>28} "
            f"{row['top1_accuracy']:>8.3f} {row['hit@10']:>8.3f} {row['mrr@10']:>8.3f}"
        )


if __name__ == "__main__":
    main()
