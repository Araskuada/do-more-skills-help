#!/usr/bin/env python3
"""Run the RQ4 downstream task-performance experiment.

RQ4 asks: does retrieving the *correct* skill actually improve downstream
task performance, compared to (a) a wrong-but-plausible skill actually
returned by the retriever, (b) a maximally confusing hard-distractor skill,
(c) a random-wrong skill, or (d) no skill at all?

Design notes (kept consistent with RQ1-RQ3 wherever the underlying question
allows it):

- Uses the same 87 Skill-Usage tasks (intersection of task_queries.json and
  task_skill_mapping.json) that RQ1-RQ3 used.
- Reuses RQ1's BM25 retriever (`bm25_rank`) to determine what a full-library
  retrieval pipeline would *actually* hand to an agent for each task. This
  naturally reproduces RQ1's own top-1 correct/incorrect split, so RQ4 can
  ask "does that split matter downstream?"
- Reuses RQ2's hardest wrong-skill construction (`rank_by_query_overlap`) for
  a worst-case "confusing wrong skill" condition.
- Reuses RQ3-enhanced's full-`SKILL.md` content (from
  `search_server/index/skills_full.db`, already built while running RQ3) so
  the agent sees the same skill content a real system would inject, not just
  name+description. Falls back to name+description with a printed warning if
  that database is not present locally.

Key difference from RQ1-RQ3: those scripts only do free, local computation,
so they could afford 20 (RQ1/RQ2) or 3 (RQ3) repeats over all 87 tasks. RQ4
makes real Claude API calls, which cost money and wall-clock time. Defaults
here are deliberately smaller (a pilot-sized task sample, repeats=3 to match
RQ3) and everything is configurable via CLI flags so you can scale up once
the pilot looks sane. Use --dry-run first to sanity check prompts without
spending anything.

IMPORTANT LIMITATION: there is no Docker sandbox available in this
environment, so this script does NOT execute oracle/verifier scripts. It
reads them as reference text for an LLM judge instead. This is a
text-only proxy for task success, not a sandboxed pass/fail signal. Treat
RQ4's pass rate as "would a knowledgeable judge, comparing the model's
proposed solution against the oracle solution and verifier criteria, call
this a pass" -- not "did this actually run and exit 0". See the README
that ships alongside this script for the full discussion.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

from rq1_retrieval_scaling import (
    GT_OWNER,
    bm25_rank,
    load_json,
    load_skill_docs,
    normalize_gt,
)
from rq2_distractor_types import rank_by_query_overlap

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

TASK_FILE_CANDIDATES = ["task.md", "README.md", "instructions.md", "prompt.md", "TASK.md"]
ORACLE_DIR_CANDIDATES = ["oracle", "solution"]
ORACLE_FILE_CANDIDATES = ["solve.sh", "solution.sh", "solve.py", "SOLUTION.md", "solution.md"]
VERIFIER_DIR_CANDIDATES = ["verifier", "tests", "test"]
VERIFIER_FILE_CANDIDATES = ["test.sh", "test_outputs.py", "run_tests.sh", "tests.py", "test.py"]

DEFAULT_CONDITIONS = [
    "no_skill",
    "gold_skill",
    "retrieved_top1_full",
    "hard_distractor",
    "random_wrong",
]

MAX_TASK_CHARS = 6000
MAX_REFERENCE_CHARS = 4000
MAX_SKILL_CHARS = 6000


# ---------------------------------------------------------------------------
# Task bundle loading
# ---------------------------------------------------------------------------

def find_task_dir(tasks_root: Path, task_id: str) -> Path | None:
    direct = tasks_root / task_id
    if direct.is_dir():
        return direct
    lowered = task_id.lower()
    for child in tasks_root.iterdir():
        if child.is_dir() and child.name.lower() == lowered:
            return child
    return None


def read_first_existing(base: Path, dir_candidates: list[str], file_candidates: list[str]) -> tuple[str, str]:
    search_dirs = [base] + [base / d for d in dir_candidates]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for fname in file_candidates:
            fpath = d / fname
            if fpath.is_file():
                try:
                    return str(fpath), fpath.read_text(errors="replace")
                except OSError:
                    continue
    return "", ""


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}"


def load_task_bundle(task_dir: Path) -> dict:
    task_path, task_text = read_first_existing(task_dir, [], TASK_FILE_CANDIDATES)
    oracle_path, oracle_text = read_first_existing(task_dir, ORACLE_DIR_CANDIDATES, ORACLE_FILE_CANDIDATES)
    verifier_path, verifier_text = read_first_existing(task_dir, VERIFIER_DIR_CANDIDATES, VERIFIER_FILE_CANDIDATES)

    warnings = []
    if not task_text:
        warnings.append(f"no task description file found under {task_dir} (looked for {TASK_FILE_CANDIDATES})")
    if not oracle_text:
        warnings.append(f"no oracle solution found under {task_dir} (looked in {ORACLE_DIR_CANDIDATES})")
    if not verifier_text:
        warnings.append(f"no verifier found under {task_dir} (looked in {VERIFIER_DIR_CANDIDATES})")

    return {
        "task_dir": str(task_dir),
        "task_path": task_path,
        "task_text": truncate(task_text, MAX_TASK_CHARS),
        "oracle_path": oracle_path,
        "oracle_text": truncate(oracle_text, MAX_REFERENCE_CHARS),
        "verifier_path": verifier_path,
        "verifier_text": truncate(verifier_text, MAX_REFERENCE_CHARS),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Skill content loading
# ---------------------------------------------------------------------------

def load_full_skill_bodies(db_path: Path) -> dict[str, str]:
    if not db_path.is_file():
        return {}
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute("SELECT skill_id, skill_md_content FROM skills").fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return {row["skill_id"]: (row["skill_md_content"] or "") for row in rows}


def build_skill_text(skill_id: str, docs: dict, full_bodies: dict[str, str]) -> str:
    doc = docs.get(skill_id)
    if doc is None:
        return f"[skill metadata missing for {skill_id}]"
    header = f"# Skill: {doc['name']}\nSource: {doc['owner']}/{doc['repo']}\n\n"
    body = full_bodies.get(skill_id) or doc["description"] or "(no description available)"
    return truncate(header + body, MAX_SKILL_CHARS)


# ---------------------------------------------------------------------------
# Condition construction
# ---------------------------------------------------------------------------

def build_conditions_for_task(
    task: str,
    query: str,
    gold: set[str],
    all_skill_ids: list[str],
    docs: dict,
    top_k_full: int,
    rng: random.Random,
) -> dict[str, dict]:
    gold_sorted = sorted(gold & set(all_skill_ids))
    gold_id = gold_sorted[0] if gold_sorted else None

    ranked_full = bm25_rank(query, all_skill_ids, docs, top_k_full)
    retrieved_id = ranked_full[0]["skill_id"] if ranked_full else None
    retrieval_correct = bool(retrieved_id and retrieved_id in gold)

    non_gold_ids = [sid for sid in all_skill_ids if sid not in gold]
    hard_ranked = rank_by_query_overlap(query, non_gold_ids, docs)
    hard_id = hard_ranked[0] if hard_ranked else rng.choice(non_gold_ids)

    random_id = rng.choice(non_gold_ids)

    conditions = {
        "no_skill": {"skill_id": None, "retrieval_correct": None},
        "gold_skill": {"skill_id": gold_id, "retrieval_correct": True},
        "retrieved_top1_full": {"skill_id": retrieved_id, "retrieval_correct": retrieval_correct},
        "hard_distractor": {"skill_id": hard_id, "retrieval_correct": False},
        "random_wrong": {"skill_id": random_id, "retrieval_correct": False},
    }
    return conditions


# ---------------------------------------------------------------------------
# Claude API calls
# ---------------------------------------------------------------------------

SOLVER_SYSTEM = (
    "You are an AI agent attempting to complete a real task. If a skill "
    "guide is provided, use it as your primary reference for tools, "
    "commands, and procedure. If no skill guide is provided, solve the "
    "task using your own general knowledge. Produce a concrete, complete "
    "solution: the actual commands, code, or step-by-step actions you "
    "would run, not a plan to make a plan."
)

JUDGE_SYSTEM = (
    "You are grading whether a proposed solution would satisfy a task's "
    "success criteria. You will see the task description, the official "
    "oracle solution (reference only, may be partial or truncated), the "
    "verifier/test logic (reference only), and the candidate solution. "
    "Judge strictly against what the verifier appears to check, not "
    "general writing quality. Reply with ONLY a JSON object, no prose, "
    "no markdown fences: "
    '{"passed": 0 or 1, "quality_score": float 0.0-1.0, "rationale": '
    '"one or two sentences"}'
)


def call_solver(client, model: str, task_text: str, skill_text: str | None, max_tokens: int) -> str:
    if skill_text:
        user_content = (
            f"## Task\n{task_text}\n\n## Available skill guide\n{skill_text}\n\n"
            "Propose your full solution now."
        )
    else:
        user_content = f"## Task\n{task_text}\n\nNo skill guide is available. Propose your full solution now."

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SOLVER_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_judge(client, model: str, bundle: dict, solution_text: str, max_tokens: int) -> dict:
    user_content = (
        f"## Task\n{bundle['task_text']}\n\n"
        f"## Oracle solution (reference)\n{bundle['oracle_text'] or '(not available)'}\n\n"
        f"## Verifier (reference)\n{bundle['verifier_text'] or '(not available)'}\n\n"
        f"## Candidate solution to grade\n{solution_text}\n\n"
        "Return the JSON object now."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text").strip()
    match = JSON_OBJ_RE.search(raw)
    if not match:
        return {"passed": 0, "quality_score": 0.0, "rationale": f"unparseable judge output: {raw[:200]}"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"passed": 0, "quality_score": 0.0, "rationale": f"invalid JSON from judge: {raw[:200]}"}
    return {
        "passed": int(bool(parsed.get("passed", 0))),
        "quality_score": float(parsed.get("quality_score", 0.0)),
        "rationale": str(parsed.get("rationale", ""))[:500],
    }


def call_with_retry(fn, *args, retries: int = 4, base_delay: float = 2.0, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as err:  # noqa: BLE001 - deliberately broad for a retry wrapper
            last_err = err
            delay = base_delay * (2**attempt)
            print(f"  [retry {attempt + 1}/{retries}] {type(err).__name__}: {err} -> sleeping {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
    raise last_err


# ---------------------------------------------------------------------------
# Resumable JSONL log
# ---------------------------------------------------------------------------

def load_done_keys(log_path: Path) -> set[tuple[str, str, int]]:
    done = set()
    if not log_path.is_file():
        return done
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["task"], row["condition"], row["repeat"]))
    return done


def append_jsonl(log_path: Path, row: dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def read_jsonl(log_path: Path) -> list[dict]:
    rows = []
    if not log_path.is_file():
        return rows
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_condition(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "pass_rate": 0.0, "pass_rate_std": 0.0, "quality_score": 0.0, "quality_score_std": 0.0}
    passed = [r["passed"] for r in rows]
    quality = [r["quality_score"] for r in rows]
    return {
        "n": n,
        "pass_rate": mean(passed),
        "pass_rate_std": pstdev(passed) if n > 1 else 0.0,
        "quality_score": mean(quality),
        "quality_score_std": pstdev(quality) if n > 1 else 0.0,
    }


def write_bar_svg(summary_rows: list[dict], path: Path) -> None:
    width, height = 900, 420
    left, right, top, bottom = 70, 30, 40, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(summary_rows)
    bar_w = plot_w / max(n, 1) * 0.5
    gap = plot_w / max(n, 1)

    def sy(v: float) -> float:
        return top + (1.0 - v) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-family="Arial" font-size="18" font-weight="700">RQ4 downstream pass rate and quality score by condition</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')

    for i, row in enumerate(summary_rows):
        cx = left + gap * i + gap / 2
        x_pass = cx - bar_w * 0.55
        x_qual = cx + bar_w * 0.05
        pass_h = plot_h * row["pass_rate"]
        qual_h = plot_h * row["quality_score"]
        lines.append(f'<rect x="{x_pass:.2f}" y="{top + plot_h - pass_h:.2f}" width="{bar_w * 0.5:.2f}" height="{pass_h:.2f}" fill="#1f77b4"/>')
        lines.append(f'<rect x="{x_qual:.2f}" y="{top + plot_h - qual_h:.2f}" width="{bar_w * 0.5:.2f}" height="{qual_h:.2f}" fill="#ff7f0e"/>')
        lines.append(
            f'<text x="{cx:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="Arial" '
            f'font-size="11" transform="rotate(28 {cx:.2f},{top + plot_h + 18})">{row["condition"]} (n={row["n"]})</text>'
        )

    legend_x = left + plot_w - 170
    lines.append(f'<rect x="{legend_x}" y="16" width="14" height="14" fill="#1f77b4"/>')
    lines.append(f'<text x="{legend_x + 20}" y="27" font-family="Arial" font-size="12">pass_rate (judge)</text>')
    lines.append(f'<rect x="{legend_x + 130}" y="16" width="14" height="14" fill="#ff7f0e"/>')
    lines.append(f'<text x="{legend_x + 150}" y="27" font-family="Arial" font-size="12">quality_score</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(all_rows: list[dict]) -> list[dict]:
    summary = []
    conditions_seen = []
    for row in all_rows:
        if row["condition"] not in conditions_seen:
            conditions_seen.append(row["condition"])

    for cond in conditions_seen:
        rows = [r for r in all_rows if r["condition"] == cond]
        stats = summarize_condition(rows)
        summary.append({"condition": cond, **stats})

    # Retrieval-conditioned split of retrieved_top1_full: this is the row
    # that most directly answers RQ4.
    retrieved_rows = [r for r in all_rows if r["condition"] == "retrieved_top1_full"]
    for label, flag in [("retrieved_top1_full__correct", True), ("retrieved_top1_full__wrong", False)]:
        subset = [r for r in retrieved_rows if r.get("retrieval_correct") is flag]
        stats = summarize_condition(subset)
        summary.append({"condition": label, **stats})

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--tasks-root", default="", help="Defaults to <skill-usage-root>/tasks")
    parser.add_argument("--output-dir", default="data/experiments/rq4_downstream_task_performance")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--repeats", type=int, default=3, help="Matches RQ3's repeat count.")
    parser.add_argument("--seed", type=int, default=6002, help="Matches RQ1-RQ3's seed.")
    parser.add_argument("--limit-tasks", type=int, default=15, help="0 means all 87 tasks. Start small: API calls cost money.")
    parser.add_argument("--top-k-full", type=int, default=1, help="How many full-library BM25 ranks to compute (only rank 1 is used).")
    parser.add_argument("--solver-model", default="claude-sonnet-5")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-tokens-solver", type=int, default=1200)
    parser.add_argument("--max-tokens-judge", type=int, default=400)
    parser.add_argument("--sleep-between-calls", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true", help="Build conditions and print prompt sizes; make no API calls.")
    args = parser.parse_args()

    root = Path(args.skill_usage_root)
    tasks_root = Path(args.tasks_root) if args.tasks_root else root / "tasks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "raw_log.jsonl"
    transcripts_path = output_dir / "transcript_examples.json"

    queries = load_json(root / "data" / "task_queries.json")
    gt = normalize_gt(load_json(root / "data" / "task_skill_mapping.json"))
    docs = load_skill_docs(root / "skills-34k" / "skills_meta.jsonl")
    all_skill_ids = sorted(docs)

    full_db_path = root / "search_server" / "index" / "skills_full.db"
    full_bodies = load_full_skill_bodies(full_db_path)
    if full_bodies:
        print(f"Loaded full SKILL.md content for {len(full_bodies)} skills from {full_db_path}")
    else:
        print(
            f"WARNING: {full_db_path} not found or empty. Falling back to name+description only. "
            "Full SKILL.md content makes the agent's context far more realistic; if you have "
            "time, unzip skills-34k/skills.zip and rebuild skills_full.db the way RQ3-enhanced did, "
            "then rerun this script."
        )

    tasks = sorted(set(queries) & set(gt))
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    print(f"Running RQ4 over {len(tasks)} tasks x {len(args.conditions)} conditions x {args.repeats} repeats")

    if not args.dry_run:
        if anthropic is None:
            raise SystemExit("The `anthropic` package is not installed. Run: pip install anthropic")
        client = anthropic.Anthropic()
    else:
        client = None

    done_keys = load_done_keys(log_path)
    if done_keys:
        print(f"Resuming: {len(done_keys)} (task, condition, repeat) rows already logged in {log_path}")

    transcript_examples = {}
    missing_task_dirs = []

    for task in tasks:
        task_dir = find_task_dir(tasks_root, task)
        if task_dir is None:
            missing_task_dirs.append(task)
            continue
        bundle = load_task_bundle(task_dir)
        for w in bundle["warnings"]:
            print(f"  [warn] {task}: {w}")

        gold = gt[task]
        query = " ".join(queries[task])
        rng = random.Random(args.seed + hash(task) % 100000)
        conditions = build_conditions_for_task(task, query, gold, all_skill_ids, docs, args.top_k_full, rng)

        for cond_name in args.conditions:
            if cond_name not in conditions:
                print(f"  [skip] unknown condition '{cond_name}'")
                continue
            cond = conditions[cond_name]
            skill_id = cond["skill_id"]
            skill_text = build_skill_text(skill_id, docs, full_bodies) if skill_id else None

            for repeat in range(args.repeats):
                key = (task, cond_name, repeat)
                if key in done_keys:
                    continue

                if args.dry_run:
                    prompt_len = len(bundle["task_text"]) + len(skill_text or "")
                    print(f"  [dry-run] {task} / {cond_name} / repeat {repeat}: skill={skill_id} prompt_chars~={prompt_len}")
                    continue

                solution = call_with_retry(
                    call_solver, client, args.solver_model, bundle["task_text"], skill_text, args.max_tokens_solver
                )
                time.sleep(args.sleep_between_calls)
                verdict = call_with_retry(
                    call_judge, client, args.judge_model, bundle, solution, args.max_tokens_judge
                )
                time.sleep(args.sleep_between_calls)

                row = {
                    "task": task,
                    "condition": cond_name,
                    "repeat": repeat,
                    "skill_id": skill_id,
                    "retrieval_correct": cond["retrieval_correct"],
                    "passed": verdict["passed"],
                    "quality_score": verdict["quality_score"],
                    "rationale": verdict["rationale"],
                }
                append_jsonl(log_path, row)
                done_keys.add(key)

                if repeat == 0 and len(transcript_examples) < 40:
                    transcript_examples[f"{task}:{cond_name}"] = {
                        "skill_id": skill_id,
                        "solution": solution[:2000],
                        "verdict": verdict,
                    }
                print(f"  {task} / {cond_name} / repeat {repeat}: passed={verdict['passed']} quality={verdict['quality_score']:.2f}")

    if missing_task_dirs:
        print(f"\nWARNING: could not find a task directory under {tasks_root} for {len(missing_task_dirs)} tasks:")
        print(", ".join(missing_task_dirs[:20]) + (" ..." if len(missing_task_dirs) > 20 else ""))
        print("Run inspect_tasks_dir.py against your tasks-root to check the actual folder naming.")

    if args.dry_run:
        print("\nDry run complete. No API calls made, nothing written to raw_log.jsonl.")
        return

    all_rows = read_jsonl(log_path)
    summary_rows = build_summary(all_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    write_csv(output_dir / "per_query_metrics.csv", all_rows)
    transcripts_path.write_text(json.dumps(transcript_examples, indent=2) + "\n")
    write_bar_svg([r for r in summary_rows if r["condition"] in args.conditions], output_dir / "condition_bar.svg")

    print(f"\nWrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'per_query_metrics.csv'}")
    print(f"Wrote {transcripts_path}")
    print(f"Wrote {output_dir / 'condition_bar.svg'}")
    print()
    print(f"{'condition':>28} {'n':>5} {'pass_rate':>10} {'quality':>10}")
    for row in summary_rows:
        print(f"{row['condition']:>28} {row['n']:>5} {row['pass_rate']:>10.3f} {row['quality_score']:>10.3f}")


if __name__ == "__main__":
    main()
