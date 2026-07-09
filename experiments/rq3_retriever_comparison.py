#!/usr/bin/env python3
"""Run the formal RQ3 retriever comparison experiment.

RQ3 asks how different retrievers behave under large-scale skill libraries.
This script compares local, fully reproducible retrievers over Skill-Usage:

- bm25: corpus-level BM25 over skill name + description.
- tfidf: sparse TF-IDF cosine similarity.
- lsa_dense: dense latent semantic analysis using TF-IDF + TruncatedSVD.
- hybrid_bm25_lsa: reciprocal-rank fusion of BM25 and LSA.
- bm25_lsa_rerank: BM25 first-stage retrieval reranked by LSA scores.

The official neural query embedding model is not assumed to be locally cached,
so lsa_dense is a local dense retrieval proxy rather than a neural embedding
retriever.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from rq1_retrieval_scaling import (
    DEFAULT_POOL_SIZES as RQ1_POOL_SIZES,
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


DEFAULT_POOL_SIZES = ["100", "1000", "10000", "full"]
DEFAULT_RETRIEVERS = ["bm25", "tfidf", "lsa_dense", "hybrid_bm25_lsa", "bm25_lsa_rerank"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def skill_text(doc: dict) -> str:
    return f"{doc['name']} {doc['description']}"


def bm25_full_scores(
    query: str,
    count_vectorizer: CountVectorizer,
    counts_csc: sparse.csc_matrix,
    doc_lengths: np.ndarray,
    doc_freq: np.ndarray,
    avg_doc_len: float,
) -> np.ndarray:
    terms = tokenize(query)
    scores = np.zeros(counts_csc.shape[0], dtype=np.float32)
    if not terms:
        return scores

    vocab = count_vectorizer.vocabulary_
    n_docs = counts_csc.shape[0]
    k1 = 1.5
    b = 0.75
    for term in terms:
        term_index = vocab.get(term)
        if term_index is None:
            continue
        col = counts_csc.getcol(term_index)
        if col.nnz == 0:
            continue
        df = doc_freq[term_index]
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        doc_idx = col.indices
        freq = col.data.astype(np.float32)
        denom = freq + k1 * (1 - b + b * doc_lengths[doc_idx] / avg_doc_len)
        scores[doc_idx] += idf * (freq * (k1 + 1)) / denom
    return scores


def rank_candidates(
    scores: np.ndarray,
    candidate_ids: list[str],
    skill_id_to_index: dict[str, int],
    top_k: int,
) -> list[str]:
    scored = [(float(scores[skill_id_to_index[sid]]), sid) for sid in candidate_ids]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sid for _, sid in scored[:top_k]]


def reciprocal_rank_fusion(
    bm25_scores: np.ndarray,
    lsa_scores: np.ndarray,
    candidate_ids: list[str],
    skill_id_to_index: dict[str, int],
    top_k: int,
    k: int = 60,
) -> list[str]:
    bm25_ranked = rank_candidates(bm25_scores, candidate_ids, skill_id_to_index, len(candidate_ids))
    lsa_ranked = rank_candidates(lsa_scores, candidate_ids, skill_id_to_index, len(candidate_ids))
    fused = {}
    for rank, sid in enumerate(bm25_ranked, start=1):
        fused[sid] = fused.get(sid, 0.0) + 1.0 / (k + rank)
    for rank, sid in enumerate(lsa_ranked, start=1):
        fused[sid] = fused.get(sid, 0.0) + 1.0 / (k + rank)
    return [
        sid
        for sid, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]


def bm25_lsa_rerank(
    bm25_scores: np.ndarray,
    lsa_scores: np.ndarray,
    candidate_ids: list[str],
    skill_id_to_index: dict[str, int],
    top_k: int,
    rerank_depth: int,
) -> list[str]:
    first_stage = rank_candidates(
        bm25_scores,
        candidate_ids,
        skill_id_to_index,
        min(rerank_depth, len(candidate_ids)),
    )
    return rank_candidates(lsa_scores, first_stage, skill_id_to_index, top_k)


def summarize(rows: list[dict]) -> dict:
    summary = {key: mean(row[key] for row in rows) for key in SUMMARY_METRICS}
    summary["top1_error_rate"] = 1.0 - summary["top1_accuracy"]
    summary["n"] = len(rows)
    return summary


def metric_stdev(rows: list[dict], metric: str) -> float:
    values = [row[metric] for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def write_metric_svg(summary_rows: list[dict], path: Path) -> None:
    width = 980
    height = 500
    left = 82
    right = 190
    top = 38
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "bm25": "#1f77b4",
        "tfidf": "#2ca02c",
        "lsa_dense": "#ff7f0e",
        "hybrid_bm25_lsa": "#9467bd",
        "bm25_lsa_rerank": "#d62728",
    }
    labels = {
        "bm25": "BM25",
        "tfidf": "TF-IDF",
        "lsa_dense": "LSA dense",
        "hybrid_bm25_lsa": "Hybrid",
        "bm25_lsa_rerank": "BM25->LSA",
    }

    pool_values = [
        34396 if row["pool_size"] == "full" else int(row["pool_size"])
        for row in summary_rows
    ]
    min_x = math.log10(min(pool_values))
    max_x = math.log10(max(pool_values))

    def sx(size: int) -> float:
        return left + (math.log10(size) - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - value) * plot_h

    pool_labels = []
    for row in summary_rows:
        value = 34396 if row["pool_size"] == "full" else int(row["pool_size"])
        label = row["pool_size"]
        if (value, label) not in pool_labels:
            pool_labels.append((value, label))
    pool_labels.sort(key=lambda item: item[0])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18" font-weight="700">RQ3 retriever comparison: Top-1 accuracy</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for value, label in pool_labels:
        x = sx(value)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')

    for retriever in DEFAULT_RETRIEVERS:
        rows = [row for row in summary_rows if row["retriever"] == retriever]
        rows.sort(key=lambda row: 34396 if row["pool_size"] == "full" else int(row["pool_size"]))
        points = [
            (
                sx(34396 if row["pool_size"] == "full" else int(row["pool_size"])),
                sy(row["top1_accuracy"]),
            )
            for row in rows
        ]
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        color = colors[retriever]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_str}"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w + 28
    for i, retriever in enumerate(DEFAULT_RETRIEVERS):
        y = top + 18 + i * 24
        color = colors[retriever]
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Arial" font-size="13">{labels[retriever]}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Candidate pool size (log scale)</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="13">Top-1 accuracy</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def check_neural_embedding_availability(model_name: str) -> dict:
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name, local_files_only=True)
        return {"model": model_name, "local_files_available": True, "error": ""}
    except Exception as exc:
        return {
            "model": model_name,
            "local_files_available": False,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--output-dir", default="data/experiments/rq3_retriever_comparison")
    parser.add_argument("--pool-sizes", nargs="+", default=DEFAULT_POOL_SIZES)
    parser.add_argument("--retrievers", nargs="+", default=DEFAULT_RETRIEVERS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lsa-components", type=int, default=128)
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--neural-model-name", default="Qwen/Qwen3-Embedding-4B")
    args = parser.parse_args()

    root = Path(args.skill_usage_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_json(root / "data" / "task_queries.json")
    gt = normalize_gt(load_json(root / "data" / "task_skill_mapping.json"))
    docs = load_skill_docs(root / "skills-34k" / "skills_meta.jsonl")
    all_skill_ids = sorted(docs)
    skill_id_to_index = {sid: index for index, sid in enumerate(all_skill_ids)}
    texts = [skill_text(docs[sid]) for sid in all_skill_ids]
    tasks = sorted(set(queries) & set(gt))
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    neural_status = check_neural_embedding_availability(args.neural_model_name)

    build_times = {}
    start = time.perf_counter()
    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True)
    counts = count_vectorizer.fit_transform(texts).astype(np.float32)
    counts_csc = counts.tocsc()
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    doc_lengths[doc_lengths == 0] = 1.0
    doc_freq = np.diff(counts_csc.indptr).astype(np.float32)
    avg_doc_len = float(doc_lengths.mean())
    build_times["bm25"] = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True, norm="l2")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts).astype(np.float32)
    build_times["tfidf"] = time.perf_counter() - start

    start = time.perf_counter()
    n_components = min(args.lsa_components, min(tfidf_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    lsa_matrix = normalize(svd.fit_transform(tfidf_matrix).astype(np.float32))
    build_times["lsa_dense"] = time.perf_counter() - start
    build_times["hybrid_bm25_lsa"] = build_times["bm25"] + build_times["lsa_dense"]
    build_times["bm25_lsa_rerank"] = build_times["bm25"] + build_times["lsa_dense"]

    task_scores = {}
    score_times = Counter()
    for task in tasks:
        query = " ".join(queries[task])
        task_scores[task] = {}

        start = time.perf_counter()
        bm25_scores = bm25_full_scores(
            query,
            count_vectorizer,
            counts_csc,
            doc_lengths,
            doc_freq,
            avg_doc_len,
        )
        score_times["bm25"] += time.perf_counter() - start
        task_scores[task]["bm25"] = bm25_scores

        start = time.perf_counter()
        q_tfidf = tfidf_vectorizer.transform([query]).astype(np.float32)
        tfidf_scores = np.asarray((tfidf_matrix @ q_tfidf.T).todense()).ravel().astype(np.float32)
        score_times["tfidf"] += time.perf_counter() - start
        task_scores[task]["tfidf"] = tfidf_scores

        start = time.perf_counter()
        q_lsa = normalize(svd.transform(q_tfidf).astype(np.float32))
        lsa_scores = np.asarray(lsa_matrix @ q_lsa.T).ravel().astype(np.float32)
        score_times["lsa_dense"] += time.perf_counter() - start
        task_scores[task]["lsa_dense"] = lsa_scores

    per_query_rows = []
    summary_rows = []
    ranking_examples = {}

    for retriever in args.retrievers:
        for pool_label in args.pool_sizes:
            pool_size = parse_pool_size(pool_label, len(all_skill_ids))
            repeat_count = 1 if pool_label == "full" else args.repeats
            pooled_rows = []

            for repeat in range(repeat_count):
                rng = random.Random(args.seed + repeat + pool_size * 1009 + len(retriever) * 271)
                for task in tasks:
                    gold = gt[task]
                    candidates = sample_candidates(all_skill_ids, gold, pool_size, rng)
                    scores = task_scores[task]

                    if retriever == "hybrid_bm25_lsa":
                        ranked_ids = reciprocal_rank_fusion(
                            scores["bm25"],
                            scores["lsa_dense"],
                            candidates,
                            skill_id_to_index,
                            args.top_k,
                        )
                    elif retriever == "bm25_lsa_rerank":
                        ranked_ids = bm25_lsa_rerank(
                            scores["bm25"],
                            scores["lsa_dense"],
                            candidates,
                            skill_id_to_index,
                            args.top_k,
                            args.rerank_depth,
                        )
                    else:
                        ranked_ids = rank_candidates(
                            scores[retriever],
                            candidates,
                            skill_id_to_index,
                            args.top_k,
                        )

                    row = {
                        "task": task,
                        "retriever": retriever,
                        "pool_size": pool_label,
                        "actual_pool_size": len(candidates),
                        "repeat": repeat,
                        "gold_count": len(gold),
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

                    example_key = f"{retriever}:{pool_label}:{repeat}:{task}"
                    if repeat == 0 and len(ranking_examples) < 80:
                        ranking_examples[example_key] = {
                            "query": " ".join(queries[task]),
                            "gold": sorted(gold),
                            "top_results": ranked_ids,
                        }

            summary = summarize(pooled_rows)
            score_seconds_total = score_times.get(retriever, 0.0)
            if retriever in {"hybrid_bm25_lsa", "bm25_lsa_rerank"}:
                score_seconds_total = score_times["bm25"] + score_times["lsa_dense"]
            summary_row = {
                "retriever": retriever,
                "pool_size": pool_label,
                "tasks": len(tasks),
                "repeats": repeat_count,
                "build_seconds": build_times.get(retriever, 0.0),
                "score_seconds_total": score_seconds_total,
                "score_seconds_per_query": score_seconds_total / len(tasks),
                **summary,
            }
            for metric in SUMMARY_METRICS:
                summary_row[f"{metric}_std_across_queries"] = metric_stdev(pooled_rows, metric)
            summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: (row["retriever"], 34396 if row["pool_size"] == "full" else int(row["pool_size"])))

    metadata = {
        "retrievers": {
            "bm25": "Corpus-level BM25 over skill name + description.",
            "tfidf": "Sparse TF-IDF cosine similarity.",
            "lsa_dense": "Local dense LSA proxy built from TF-IDF + TruncatedSVD.",
            "hybrid_bm25_lsa": "Reciprocal-rank fusion of BM25 and LSA.",
            "bm25_lsa_rerank": f"BM25 top-{args.rerank_depth} reranked by LSA scores.",
        },
        "neural_embedding_status": neural_status,
        "note": (
            "The neural query embedding model is not required for this run. If it is "
            "not cached locally, lsa_dense should be interpreted as a local dense "
            "retrieval proxy rather than a neural embedding retriever."
        ),
        "lsa_components": n_components,
        "pool_sizes": args.pool_sizes,
    }

    (output_dir / "summary.json").write_text(json.dumps({"metadata": metadata, "summary": summary_rows}, indent=2) + "\n")
    (output_dir / "ranking_examples.json").write_text(json.dumps(ranking_examples, indent=2) + "\n")
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_query_metrics.csv", per_query_rows)
    write_metric_svg(summary_rows, output_dir / "top1_by_retriever.svg")

    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'per_query_metrics.csv'}")
    print(f"Wrote {output_dir / 'ranking_examples.json'}")
    print(f"Wrote {output_dir / 'top1_by_retriever.svg'}")
    print()
    print(f"{'retriever':>18} {'pool':>8} {'top1':>8} {'hit@10':>8} {'mrr':>8} {'ndcg':>8}")
    for row in summary_rows:
        print(
            f"{row['retriever']:>18} "
            f"{row['pool_size']:>8} "
            f"{row['top1_accuracy']:>8.3f} "
            f"{row['hit@10']:>8.3f} "
            f"{row['mrr@10']:>8.3f} "
            f"{row['ndcg@10']:>8.3f}"
        )


if __name__ == "__main__":
    main()
