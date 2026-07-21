#!/usr/bin/env python3
"""Question 8: relational skill graph retrieval experiment.

This script is intentionally self-contained and writes only below ``question_8``.
It reuses the read-only Skill-Usage corpus, the local MiniLM model, and the cached
RQ3 document embeddings.  The formal result is leakage-free five-fold CV; a
transductive co-required graph is reported only as an explicitly labelled upper
bound.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


SEED = 6002
FINAL_BUDGET = 10
RRF_K = 60
ARTIFACT_PATTERNS = {
    "pdf": ("pdf",),
    "csv": ("csv", "comma-separated"),
    "spreadsheet": ("xlsx", "excel", "spreadsheet"),
    "presentation": ("pptx", "powerpoint", "slide deck", "presentation"),
    "json": ("json",),
    "yaml": ("yaml", "yml"),
    "database": ("sql", "sqlite", "database"),
    "graph": ("graph", "network topology"),
    "image": ("image", "png", "jpeg", "jpg", "svg"),
    "video": ("video", "mp4"),
    "audio": ("audio", "wav", "speech"),
    "html": ("html", "web page", "webpage"),
    "text": ("text file", "markdown", "document"),
}
CONSUME_WORDS = (
    "read", "load", "parse", "extract", "analyze", "inspect", "import",
    "process", "convert from", "consume", "input",
)
PRODUCE_WORDS = (
    "write", "save", "export", "generate", "create", "render", "produce",
    "convert to", "output", "build",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_skill_docs(path: Path) -> dict[str, dict]:
    docs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("skill_id") or row.get("skillId") or row.get("name")
            if sid:
                docs[sid] = row
    return docs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_bm25(texts: list[str]):
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True)
    counts = vectorizer.fit_transform(texts).astype(np.float32)
    csc = counts.tocsc()
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    lengths[lengths == 0] = 1.0
    df = np.diff(csc.indptr).astype(np.float32)
    return vectorizer, csc, lengths, df, float(lengths.mean())


def bm25_scores(query: str, vectorizer, csc, lengths, df, avg_len: float) -> np.ndarray:
    scores = np.zeros(csc.shape[0], dtype=np.float32)
    vocab = vectorizer.vocabulary_
    n_docs = csc.shape[0]
    k1, b = 1.5, 0.75
    for term in tokenize(query):
        idx = vocab.get(term)
        if idx is None:
            continue
        col = csc.getcol(idx)
        if not col.nnz:
            continue
        idf = math.log(1.0 + (n_docs - df[idx] + 0.5) / (df[idx] + 0.5))
        rows = col.indices
        freq = col.data.astype(np.float32)
        denom = freq + k1 * (1.0 - b + b * lengths[rows] / avg_len)
        scores[rows] += idf * (freq * (k1 + 1.0)) / denom
    return scores


def rank_indices(scores: np.ndarray) -> np.ndarray:
    # Stable deterministic tie-breaking by candidate index.
    return np.lexsort((np.arange(len(scores)), -scores))


def rrf_ranking(bm25: np.ndarray, dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    br = rank_indices(bm25)
    dr = rank_indices(dense)
    bpos = np.empty(len(br), dtype=np.int32)
    dpos = np.empty(len(dr), dtype=np.int32)
    bpos[br] = np.arange(1, len(br) + 1)
    dpos[dr] = np.arange(1, len(dr) + 1)
    fused = 1.0 / (RRF_K + bpos) + 1.0 / (RRF_K + dpos)
    return rank_indices(fused), fused.astype(np.float32)


def choose_candidate_library(
    all_ids: list[str],
    gold_ids: set[str],
    task_bm25: dict[str, np.ndarray],
    task_dense: dict[str, np.ndarray],
    target_size: int,
) -> tuple[list[str], dict]:
    if target_size < len(gold_ids):
        raise ValueError("candidate library must be larger than the union of gold skills")
    selected = set(gold_ids)
    source = {sid: "gold_union" for sid in gold_ids}
    per_query_depth = max(10, min(40, target_size // 100))
    for task in sorted(task_bm25):
        for label, scores in (("bm25_hard", task_bm25[task]), ("dense_hard", task_dense[task])):
            for idx in rank_indices(scores)[:per_query_depth]:
                sid = all_ids[int(idx)]
                if sid not in selected:
                    selected.add(sid)
                    source[sid] = label
    rng = random.Random(SEED)
    remaining = [sid for sid in all_ids if sid not in selected]
    if len(selected) > target_size:
        # Preserve every gold skill, then deterministically balance both hard-negative sources.
        bm = sorted(s for s, label in source.items() if label == "bm25_hard")
        de = sorted(s for s, label in source.items() if label == "dense_hard")
        keep = set(gold_ids)
        pools = [bm, de]
        cursor = 0
        while len(keep) < target_size and any(pools):
            pool = pools[cursor % 2]
            if pool:
                keep.add(pool.pop(0))
            cursor += 1
        selected = keep
    elif len(selected) < target_size:
        fill = rng.sample(remaining, target_size - len(selected))
        for sid in fill:
            selected.add(sid)
            source[sid] = "random_fill"
    ids = sorted(selected)
    stats = dict(Counter(source[sid] for sid in ids))
    stats.update({"target_size": target_size, "actual_size": len(ids), "hard_depth_per_query": per_query_depth})
    return ids, stats


def add_edge(graph: dict[str, dict[str, dict[str, float]]], a: str, b: str, edge_type: str, weight: float) -> None:
    if a == b:
        return
    graph[a][b][edge_type] = max(graph[a][b].get(edge_type, 0.0), float(weight))
    graph[b][a][edge_type] = max(graph[b][a].get(edge_type, 0.0), float(weight))


def semantic_knn_edges(ids: list[str], emb: np.ndarray, k: int, threshold: float):
    graph = defaultdict(lambda: defaultdict(dict))
    rows = []
    n = len(ids)
    chunk = 256
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        sims = emb[start:stop] @ emb.T
        for local, idx in enumerate(range(start, stop)):
            sims[local, idx] = -1.0
        take = min(k, n - 1)
        part = np.argpartition(-sims, take - 1, axis=1)[:, :take]
        for local, idx in enumerate(range(start, stop)):
            ordered = part[local][np.argsort(-sims[local, part[local]])]
            for nbr in ordered:
                sim = float(sims[local, nbr])
                if sim < threshold:
                    continue
                a, b = ids[idx], ids[int(nbr)]
                add_edge(graph, a, b, "semantic_knn", sim)
    seen = set()
    for a in sorted(graph):
        for b, types in sorted(graph[a].items()):
            key = tuple(sorted((a, b))) + ("semantic_knn",)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"source": a, "target": b, "edge_type": "semantic_knn", "weight": types["semantic_knn"], "scope": "content_only"})
    return graph, rows


def artifact_roles(text: str) -> dict[str, str]:
    lowered = text.lower()
    has_consume = any(w in lowered for w in CONSUME_WORDS)
    has_produce = any(w in lowered for w in PRODUCE_WORDS)
    role = "both" if has_consume and has_produce else "consume" if has_consume else "produce" if has_produce else "mention"
    return {
        artifact: role
        for artifact, patterns in ARTIFACT_PATTERNS.items()
        if any(p in lowered for p in patterns)
    }


def add_metadata_edges(
    graph,
    edge_rows: list[dict],
    ids: list[str],
    docs: dict[str, dict],
    emb: np.ndarray,
    owner_k: int = 2,
    artifact_k: int = 3,
) -> tuple[dict[str, dict[str, str]], list[dict]]:
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    seen = {(r["source"], r["target"], r["edge_type"]) for r in edge_rows}
    owners = defaultdict(list)
    roles = {}
    by_artifact_role = defaultdict(lambda: defaultdict(list))
    for sid in ids:
        owner = str(docs[sid].get("owner") or "").strip()
        if owner:
            owners[owner].append(sid)
        text = f"{docs[sid].get('name','')} {docs[sid].get('description','')}"
        roles[sid] = artifact_roles(text)
        for artifact, role in roles[sid].items():
            by_artifact_role[artifact][role].append(sid)

    def connect_ranked(a: str, choices: list[str], edge_type: str, limit: int, scale: float):
        if not choices:
            return
        ai = id_to_idx[a]
        unique = [b for b in dict.fromkeys(choices) if b != a]
        if not unique:
            return
        scored = sorted(((float(emb[ai] @ emb[id_to_idx[b]]), b) for b in unique), reverse=True)[:limit]
        for sim, b in scored:
            weight = max(0.0, sim) * scale
            add_edge(graph, a, b, edge_type, weight)
            x, y = sorted((a, b))
            key = (x, y, edge_type)
            if key not in seen:
                seen.add(key)
                edge_rows.append({"source": x, "target": y, "edge_type": edge_type, "weight": weight, "scope": "content_only"})

    for group in owners.values():
        if len(group) < 2:
            continue
        for sid in group:
            connect_ranked(sid, group, "same_owner", owner_k, 0.30)

    for artifact, grouped in by_artifact_role.items():
        producers = grouped.get("produce", []) + grouped.get("both", [])
        consumers = grouped.get("consume", []) + grouped.get("both", [])
        for sid in producers:
            connect_ranked(sid, consumers, f"produces_consumes:{artifact}", artifact_k, 0.55)
    return roles, edge_rows


def build_corequired_graph(mapping: dict[str, set[str]], train_tasks: list[str]):
    graph = defaultdict(lambda: defaultdict(dict))
    pair_counts = Counter()
    freq = Counter()
    for task in train_tasks:
        skills = sorted(mapping[task])
        freq.update(skills)
        pair_counts.update(itertools.combinations(skills, 2))
    rows = []
    for (a, b), count in pair_counts.items():
        weight = count / math.sqrt(freq[a] * freq[b])
        add_edge(graph, a, b, "co_required", weight)
        rows.append({"source": a, "target": b, "edge_type": "co_required", "weight": weight, "count": count})
    return graph, rows


def merge_neighbors(*graphs):
    merged = defaultdict(lambda: defaultdict(dict))
    for graph in graphs:
        for a, nbrs in graph.items():
            for b, types in nbrs.items():
                for edge_type, weight in types.items():
                    merged[a][b][edge_type] = max(merged[a][b].get(edge_type, 0.0), weight)
    return merged


def edge_type_factor(edge_type: str) -> float:
    if edge_type == "co_required":
        return 1.0
    if edge_type == "semantic_knn":
        return 0.45
    if edge_type == "same_owner":
        return 0.20
    if edge_type.startswith("produces_consumes:"):
        return 0.70
    return 0.0


def graph_rank(
    base_rank: list[str],
    graph,
    seed_count: int,
    budget: int,
    beta: float,
    allowed_types: set[str] | None = None,
    expand: bool = True,
) -> list[str]:
    seeds = base_rank[:seed_count]
    candidate_set = set(base_rank[:30])
    if expand:
        for seed in seeds:
            candidate_set.update(graph.get(seed, {}))
    base_pos = {sid: i + 1 for i, sid in enumerate(base_rank)}
    scores = {}
    for sid in candidate_set:
        retrieval = 1.0 / math.log2(base_pos.get(sid, len(base_rank) + 1) + 1.0)
        graph_score = 0.0
        for seed in seeds:
            for edge_type, weight in graph.get(seed, {}).get(sid, {}).items():
                base_type = "produces_consumes" if edge_type.startswith("produces_consumes:") else edge_type
                if allowed_types is not None and base_type not in allowed_types:
                    continue
                graph_score += edge_type_factor(edge_type) * weight
        scores[sid] = retrieval + beta * graph_score
    ranked = sorted(candidate_set, key=lambda sid: (-scores[sid], base_pos.get(sid, 10**9), sid))
    # Keep seeds to model the standard retrieve-then-expand pipeline.
    output = list(seeds)
    for sid in ranked:
        if sid not in output:
            output.append(sid)
        if len(output) >= budget:
            break
    return output[:budget]


def make_folds(tasks: list[str], mapping: dict[str, set[str]], n_folds: int) -> dict[str, int]:
    by_count = defaultdict(list)
    for task in tasks:
        by_count[min(len(mapping[task]), 5)].append(task)
    assignment = {}
    for count in sorted(by_count):
        for i, task in enumerate(sorted(by_count[count])):
            assignment[task] = i % n_folds
    return assignment


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def evaluate(task: str, method: str, ranked: list[str], gold: set[str], fold: int, token_cost: dict[str, int]) -> dict:
    hits = [sid for sid in ranked if sid in gold]
    hit_count = len(hits)
    return {
        "task_id": task,
        "fold": fold,
        "method": method,
        "gold_count": len(gold),
        "is_multi_skill": int(len(gold) > 1),
        "retrieved_ids": "|".join(ranked),
        "gold_ids": "|".join(sorted(gold)),
        "hit_ids": "|".join(hits),
        "top1_is_gold": int(bool(ranked) and ranked[0] in gold),
        "any_gold_coverage": int(hit_count > 0),
        "complete_gold_coverage": int(gold.issubset(ranked)),
        "gold_recall": hit_count / len(gold),
        "skill_precision": hit_count / len(ranked),
        "extra_skill_count": len(ranked) - hit_count,
        "context_tokens_approx": sum(token_cost.get(sid, 1) for sid in ranked),
        "underload": int(not gold.issubset(ranked)),
    }


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def bootstrap_mean_ci(values: list[float], rng: np.random.Generator, repeats: int = 2000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    means = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        means[i] = arr[rng.integers(0, len(arr), len(arr))].mean()
    return percentile(means.tolist(), 2.5), percentile(means.tolist(), 97.5)


def summarize(rows: list[dict]) -> list[dict]:
    metrics = [
        "top1_is_gold", "any_gold_coverage", "complete_gold_coverage", "gold_recall",
        "skill_precision", "extra_skill_count", "context_tokens_approx", "underload",
    ]
    output = []
    rng = np.random.default_rng(SEED)
    methods = sorted({r["method"] for r in rows})
    for scope, scoped in (("all_tasks", rows), ("multi_skill_only", [r for r in rows if r["is_multi_skill"]])):
        for method in methods:
            part = [r for r in scoped if r["method"] == method]
            row = {"scope": scope, "method": method, "n_tasks": len(part)}
            for metric in metrics:
                vals = [float(r[metric]) for r in part]
                row[metric] = float(np.mean(vals))
                lo, hi = bootstrap_mean_ci(vals, rng)
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
            output.append(row)
    return output


def paired_comparisons(rows: list[dict], baseline: str = "hybrid_top10") -> list[dict]:
    metrics = ["complete_gold_coverage", "gold_recall", "skill_precision", "extra_skill_count", "context_tokens_approx"]
    multi = [r for r in rows if r["is_multi_skill"]]
    lookup = {(r["task_id"], r["method"]): r for r in multi}
    tasks = sorted({r["task_id"] for r in multi if r["method"] == baseline})
    methods = sorted({r["method"] for r in multi if r["method"] != baseline})
    rng = np.random.default_rng(SEED + 8)
    out = []
    for method in methods:
        row = {"baseline": baseline, "method": method, "n_tasks": len(tasks)}
        for metric in metrics:
            deltas = np.asarray([float(lookup[(t, method)][metric]) - float(lookup[(t, baseline)][metric]) for t in tasks])
            row[f"delta_{metric}"] = float(deltas.mean())
            lo, hi = bootstrap_mean_ci(deltas.tolist(), rng)
            row[f"delta_{metric}_ci_low"] = lo
            row[f"delta_{metric}_ci_high"] = hi
        recall_delta = [lookup[(t, method)]["gold_recall"] - lookup[(t, baseline)]["gold_recall"] for t in tasks]
        row["recall_wins"] = sum(x > 1e-12 for x in recall_delta)
        row["recall_ties"] = sum(abs(x) <= 1e-12 for x in recall_delta)
        row["recall_losses"] = sum(x < -1e-12 for x in recall_delta)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-size", type=int, default=5000)
    parser.add_argument("--semantic-k", type=int, default=8)
    parser.add_argument("--semantic-threshold", type=float, default=0.52)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    q8_root = Path(__file__).resolve().parents[1]
    workspace = q8_root.parent
    skill_root = workspace / "data" / "raw" / "Skill-Usage"
    results = q8_root / "results"
    data_dir = q8_root / "data"
    results.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    queries_raw = read_json(skill_root / "data" / "task_queries.json")
    # Skill-Usage stores benchmark mappings as short names, while the 34k
    # metadata index uses globally unique ``owner--skill`` identifiers.
    mapping = {
        task: {f"benchflow-ai--{sid}" for sid in ids}
        for task, ids in read_json(skill_root / "data" / "task_skill_mapping.json").items()
    }
    docs = load_skill_docs(skill_root / "skills-34k" / "skills_meta.jsonl")
    all_ids = sorted(docs)
    all_index = {sid: i for i, sid in enumerate(all_ids)}
    tasks = sorted(set(queries_raw) & set(mapping))
    queries = {task: " ".join(queries_raw[task]) for task in tasks}
    gold_union = set().union(*(mapping[t] for t in tasks))

    cache = workspace / "data" / "experiments" / "rerun_full_2026-07-17" / "rq3_retriever_enhanced" / "neural_doc_embeddings.npy"
    embeddings = np.load(cache).astype(np.float32)
    if embeddings.shape[0] != len(all_ids):
        raise ValueError(f"embedding cache rows {embeddings.shape[0]} != skill count {len(all_ids)}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(workspace / "model" / "all-MiniLM-L6-v2"), local_files_only=True)
    query_emb = model.encode([queries[t] for t in tasks], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    query_emb = np.asarray(query_emb, dtype=np.float32)

    desc_texts_all = [f"{docs[sid].get('name','')} {docs[sid].get('description','')}" for sid in all_ids]
    vectorizer_all, csc_all, lengths_all, df_all, avg_all = build_bm25(desc_texts_all)
    task_bm25_all = {}
    task_dense_all = {}
    for ti, task in enumerate(tasks):
        task_bm25_all[task] = bm25_scores(queries[task], vectorizer_all, csc_all, lengths_all, df_all, avg_all)
        task_dense_all[task] = embeddings @ query_emb[ti]

    candidate_ids, candidate_stats = choose_candidate_library(
        all_ids, gold_union, task_bm25_all, task_dense_all, min(args.library_size, len(all_ids))
    )
    candidate_global_idx = np.asarray([all_index[sid] for sid in candidate_ids], dtype=np.int32)
    cand_emb = embeddings[candidate_global_idx]
    cand_index = {sid: i for i, sid in enumerate(candidate_ids)}
    desc_texts = [desc_texts_all[all_index[sid]] for sid in candidate_ids]
    vectorizer, csc, lengths, df, avg_len = build_bm25(desc_texts)

    semantic_graph, edge_rows = semantic_knn_edges(candidate_ids, cand_emb, args.semantic_k, args.semantic_threshold)
    artifact_map, edge_rows = add_metadata_edges(semantic_graph, edge_rows, candidate_ids, docs, cand_emb)

    fold_of = make_folds(tasks, mapping, args.folds)
    write_json(data_dir / "splits.json", {"seed": SEED, "folds": args.folds, "task_to_fold": fold_of})
    write_csv(data_dir / "candidate_library.csv", [
        {
            "skill_id": sid,
            "name": docs[sid].get("name", ""),
            "owner": docs[sid].get("owner", ""),
            "description": docs[sid].get("description", ""),
            "is_gold_union": int(sid in gold_union),
            "artifact_roles": json.dumps(artifact_map.get(sid, {}), ensure_ascii=False, sort_keys=True),
        }
        for sid in candidate_ids
    ])

    # Token estimates use the actual full SKILL.md text for the reduced library.
    con = sqlite3.connect(skill_root / "search_server" / "index" / "skills_full.db")
    token_cost = {}
    full_text = {}
    for sid, name, desc, content in con.execute("SELECT skill_id, name, description, skill_md_content FROM skills"):
        if sid in cand_index:
            text = f"{name or ''} {desc or ''} {content or ''}"
            full_text[sid] = text
            token_cost[sid] = approx_tokens(text)
    con.close()

    base_rankings = {}
    score_cache = {}
    for ti, task in enumerate(tasks):
        bm = bm25_scores(queries[task], vectorizer, csc, lengths, df, avg_len)
        dense = cand_emb @ query_emb[ti]
        hybrid_idx, fused = rrf_ranking(bm, dense)
        base_rankings[task] = {
            "bm25": [candidate_ids[int(i)] for i in rank_indices(bm)],
            "hybrid": [candidate_ids[int(i)] for i in hybrid_idx],
        }
        score_cache[task] = {"bm25": bm, "dense": dense, "rrf": fused}

    trans_graph, trans_rows = build_corequired_graph(mapping, tasks)
    for r in trans_rows:
        r["scope"] = "transductive_upper_bound"
        edge_rows.append(r)

    per_task = []
    fold_graph_stats = []
    for fold in range(args.folds):
        test_tasks = [t for t in tasks if fold_of[t] == fold]
        train_tasks = [t for t in tasks if fold_of[t] != fold]
        cv_graph, cv_rows = build_corequired_graph(mapping, train_tasks)
        fold_graph_stats.append({
            "fold": fold,
            "train_tasks": len(train_tasks),
            "test_tasks": len(test_tasks),
            "cv_corequired_edges": len(cv_rows),
            "test_gold_skills_seen_in_train_fraction": float(np.mean([
                np.mean([any(s in mapping[x] for x in train_tasks) for s in mapping[t]]) for t in test_tasks
            ])),
        })
        leakage_free_all = merge_neighbors(semantic_graph, cv_graph)
        trans_all = merge_neighbors(semantic_graph, trans_graph)
        for task in test_tasks:
            gold = mapping[task]
            bm_rank = base_rankings[task]["bm25"]
            hybrid_rank = base_rankings[task]["hybrid"]
            outputs = {
                "bm25_top10": bm_rank[:FINAL_BUDGET],
                "hybrid_top10": hybrid_rank[:FINAL_BUDGET],
                "graph_semantic_seed3": graph_rank(hybrid_rank, semantic_graph, 3, FINAL_BUDGET, args.beta, {"semantic_knn"}),
                "graph_metadata_seed3": graph_rank(hybrid_rank, semantic_graph, 3, FINAL_BUDGET, args.beta, {"same_owner", "produces_consumes"}),
                "graph_all_content_seed3": graph_rank(hybrid_rank, semantic_graph, 3, FINAL_BUDGET, args.beta, {"semantic_knn", "same_owner", "produces_consumes"}),
                "graph_cv_corequired_seed3": graph_rank(hybrid_rank, cv_graph, 3, FINAL_BUDGET, args.beta, {"co_required"}),
                "graph_cv_all_seed3": graph_rank(hybrid_rank, leakage_free_all, 3, FINAL_BUDGET, args.beta),
                "graph_cv_all_seed5": graph_rank(hybrid_rank, leakage_free_all, 5, FINAL_BUDGET, args.beta),
                "graph_cv_prune_top20": graph_rank(hybrid_rank[:20], leakage_free_all, 3, FINAL_BUDGET, args.beta, expand=False),
                "graph_transductive_upper_bound": graph_rank(hybrid_rank, trans_all, 3, FINAL_BUDGET, args.beta),
            }
            for method, ranked in outputs.items():
                per_task.append(evaluate(task, method, ranked, gold, fold, token_cost))

    summary = summarize(per_task)
    paired = paired_comparisons(per_task)
    write_csv(results / "per_task_results.csv", per_task)
    write_csv(results / "summary.csv", summary)
    write_csv(results / "paired_comparisons.csv", paired)
    write_csv(data_dir / "graph_edges.csv", edge_rows)

    gold_counts = Counter(len(mapping[t]) for t in tasks)
    skill_freq = Counter(s for t in tasks for s in mapping[t])
    pair_freq = Counter(p for t in tasks for p in itertools.combinations(sorted(mapping[t]), 2))
    edge_counts = Counter(r["edge_type"].split(":", 1)[0] for r in edge_rows if r.get("scope") == "content_only")
    graph_stats = {
        "experiment": "Question 8 relational skill graph",
        "seed": SEED,
        "task_count": len(tasks),
        "multi_skill_task_count": sum(len(mapping[t]) > 1 for t in tasks),
        "candidate_library": candidate_stats,
        "full_library_size": len(all_ids),
        "unique_gold_skills": len(gold_union),
        "gold_assignments": sum(len(mapping[t]) for t in tasks),
        "gold_per_task_histogram": dict(sorted(gold_counts.items())),
        "gold_skill_frequency_histogram": dict(sorted(Counter(skill_freq.values()).items())),
        "singleton_gold_skills": sum(v == 1 for v in skill_freq.values()),
        "repeated_gold_skills": sum(v > 1 for v in skill_freq.values()),
        "corequired_pairs": len(pair_freq),
        "repeated_corequired_pairs": sum(v > 1 for v in pair_freq.values()),
        "content_edge_counts": dict(edge_counts),
        "transductive_corequired_edges": len(trans_rows),
        "fold_graph_stats": fold_graph_stats,
        "parameters": vars(args),
        "runtime_seconds": time.perf_counter() - started,
        "embedding_cache": str(cache),
        "token_estimation": "ceil(character_count / 4) over name + description + full SKILL.md",
    }
    write_json(results / "graph_stats.json", graph_stats)

    # Compact case studies: strongest recall gain and loss vs the baseline.
    lookup = {(r["task_id"], r["method"]): r for r in per_task}
    cases = []
    for method in ("graph_cv_all_seed3", "graph_transductive_upper_bound"):
        ranked_tasks = sorted(tasks, key=lambda t: (
            lookup[(t, method)]["gold_recall"] - lookup[(t, "hybrid_top10")]["gold_recall"], t
        ))
        for label, picked in (("largest_loss", ranked_tasks[:2]), ("largest_gain", ranked_tasks[-2:])):
            for task in picked:
                b = lookup[(task, "hybrid_top10")]
                g = lookup[(task, method)]
                cases.append({
                    "case_type": label,
                    "method": method,
                    "task_id": task,
                    "query": queries[task],
                    "gold_skills": sorted(mapping[task]),
                    "baseline_retrieved": b["retrieved_ids"].split("|"),
                    "graph_retrieved": g["retrieved_ids"].split("|"),
                    "baseline_recall": b["gold_recall"],
                    "graph_recall": g["gold_recall"],
                })
    write_json(results / "case_studies.json", cases)
    print(json.dumps({
        "status": "complete",
        "tasks": len(tasks),
        "library_size": len(candidate_ids),
        "methods": len({r['method'] for r in per_task}),
        "runtime_seconds": graph_stats["runtime_seconds"],
        "output": str(results),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
