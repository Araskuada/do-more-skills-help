#!/usr/bin/env python3
"""Adaptive, budget-aware RQ4 Qwen experiment.

RQ4 asks whether better skill exposure improves downstream task performance.
This script avoids the old "one solver + one judge per row" design. It uses a
paired task-level design instead:

1. Select high-information SkillsBench tasks from the exposure proxy results.
2. Ask Qwen to produce a concise execution plan under each exposure condition.
3. Judge all condition plans for the same task in one calibrated comparison.

The result is still an LLM-judged proxy, not a sandbox verifier pass rate, but
it is cheaper and more informative than a binary-only judged pass run.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import requests

from rq4_downstream_skill_exposure import load_skillsbench_tasks, parse_front_matter


DASHSCOPE_COMPATIBLE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_CONDITIONS = [
    "no_skill",
    "oracle_gold_all",
    "bm25_top1",
    "hybrid_bm25_neural_top10",
    "oracle_gold_plus_5_noise",
]
TEXT_SUFFIXES = {".md", ".txt", ".sh", ".py", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml"}

SOLVER_SYSTEM = (
    "You are an autonomous technical agent solving a SkillsBench task. "
    "Use visible skill guides when they are relevant, ignore irrelevant guides, "
    "and produce a concise execution plan that could actually be implemented. "
    "Do not ask clarifying questions. Do not claim you executed commands."
)

JUDGE_SYSTEM = (
    "You are a calibrated SkillsBench evaluator. You compare multiple proposed "
    "execution plans for the same task. Use the oracle and verifier as ground "
    "truth, but grade readiness rather than requiring proof of execution. "
    "Return STRICT JSON ONLY."
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def read_text(path: Path, cap: int) -> str:
    try:
        return path.read_text(errors="replace")[:cap]
    except Exception:
        return ""


def concat_dir_text(dir_path: Path, cap: int, suffixes: set[str] = TEXT_SUFFIXES) -> str:
    if not dir_path.exists():
        return ""
    chunks: list[str] = []
    total = 0
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = read_text(path, min(cap, 6000))
        if not text.strip():
            continue
        rel = path.relative_to(dir_path)
        chunks.append(f"## {rel}\n{text}")
        total += len(text)
        if total >= cap:
            break
    return "\n\n".join(chunks)[:cap]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return int(non_ascii * 0.75 + (len(text) - non_ascii) / 3.0)


def load_exposure_rows(path: Path) -> dict[str, dict[str, dict]]:
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in read_csv(path):
        for key in [
            "gold_skill_count",
            "exposed_skill_count",
            "covered_gold_count",
            "extra_skill_count",
            "context_tokens",
        ]:
            row[key] = int(float(row[key]))
        for key in ["complete_gold_coverage", "gold_recall", "skill_precision", "top1_is_gold"]:
            row[key] = float(row[key])
        by_task[row["task_id"]][row["condition"]] = row
    return by_task


def load_task_bundle(task_dir: Path, instructions_cap: int, oracle_cap: int, verifier_cap: int) -> dict:
    raw_task = read_text(task_dir / "task.md", instructions_cap * 2)
    _, body = parse_front_matter(raw_task)
    return {
        "instructions": (body or raw_task)[:instructions_cap],
        "oracle": concat_dir_text(task_dir / "oracle", oracle_cap),
        "verifier": concat_dir_text(task_dir / "verifier", verifier_cap),
    }


def skill_brief(skill: dict, cap: int) -> str:
    text = skill["text"]
    meta, body = parse_front_matter(text)
    name = skill.get("canonical_name", "")
    desc = skill.get("description", "")
    if isinstance(meta, dict):
        name = meta.get("name") or name
        desc = meta.get("description") or desc
    body = body or text
    return f"# Skill: {name}\nDescription: {desc}\n\n{body[:cap]}"


def exposure_skill_context(row: dict, skill_by_id: dict[str, dict], per_skill_cap: int, total_cap: int) -> str:
    skill_ids = [sid for sid in row.get("exposed_skill_ids", "").split(";") if sid]
    if not skill_ids:
        return ""
    chunks = []
    total = 0
    for sid in skill_ids:
        skill = skill_by_id.get(sid)
        if not skill:
            continue
        chunk = skill_brief(skill, per_skill_cap)
        chunks.append(chunk)
        total += estimate_tokens(chunk)
        if total >= total_cap:
            break
    return "\n\n---\n\n".join(chunks)[: total_cap * 4]


def condition_metadata(row: dict | None) -> dict:
    if not row:
        return {}
    return {
        "gold_skill_count": row["gold_skill_count"],
        "exposed_skill_count": row["exposed_skill_count"],
        "covered_gold_count": row["covered_gold_count"],
        "complete_gold_coverage": row["complete_gold_coverage"],
        "extra_skill_count": row["extra_skill_count"],
        "context_tokens": row["context_tokens"],
    }


def select_tasks(
    exposure: dict[str, dict[str, dict]],
    requested_tasks: list[str],
    conditions: list[str],
    max_tasks: int,
) -> list[dict]:
    if requested_tasks:
        selected = []
        for task_id in requested_tasks:
            rows = exposure.get(task_id, {})
            missing = [condition for condition in conditions if condition not in rows]
            if missing:
                raise SystemExit(f"Task {task_id} is missing conditions: {', '.join(missing)}")
            selected.append({"task_id": task_id, "selection_score": 999.0, "selection_reason": "user requested"})
        return selected[:max_tasks]

    candidates = []
    for task_id, rows in exposure.items():
        if any(condition not in rows for condition in conditions):
            continue
        oracle = rows["oracle_gold_all"]
        retrieved = [rows[c] for c in conditions if c not in {"no_skill", "oracle_gold_all"}]
        gold_count = oracle["gold_skill_count"]
        coverage_values = [r["complete_gold_coverage"] for r in retrieved]
        recall_values = [r["gold_recall"] for r in retrieved]
        has_retrieval_gap = max(recall_values or [0.0]) - min(recall_values or [0.0])
        has_noise = rows.get("oracle_gold_plus_5_noise", {}).get("extra_skill_count", 0) > 0
        top1_underloads = rows.get("bm25_top1", {}).get("complete_gold_coverage", 0.0) == 0.0
        score = (
            gold_count * 2.0
            + has_retrieval_gap * 3.0
            + (1.5 if top1_underloads else 0.0)
            + (1.0 if 0.0 in coverage_values and 1.0 in coverage_values else 0.0)
            + (0.5 if has_noise else 0.0)
        )
        reason_bits = []
        if gold_count >= 2:
            reason_bits.append(f"{gold_count} gold skills")
        if top1_underloads:
            reason_bits.append("top1 underloads")
        if has_retrieval_gap:
            reason_bits.append("retrieval coverage varies")
        if has_noise:
            reason_bits.append("tests noisy context")
        candidates.append((score, task_id, "; ".join(reason_bits) or "coverage control"))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"task_id": task_id, "selection_score": round(score, 3), "selection_reason": reason}
        for score, task_id, reason in candidates[:max_tasks]
    ]


def build_solver_messages(task_id: str, condition: str, bundle: dict, skill_context: str) -> list[dict]:
    if skill_context.strip():
        skill_block = f"Visible skill guides for `{condition}`:\n\n{skill_context}"
    else:
        skill_block = f"No skill guide is visible for `{condition}`."
    user = (
        f"Task id: {task_id}\n"
        f"Condition: {condition}\n\n"
        f"Task instructions:\n{bundle['instructions']}\n\n"
        f"{skill_block}\n\n"
        "Return a compact JSON object with these keys:\n"
        "- approach: 2-4 sentence strategy\n"
        "- files_to_create_or_modify: array of concrete paths\n"
        "- commands: array of shell commands you would run\n"
        "- validation: array of checks matching the verifier\n"
        "- risks: array of likely failure risks\n"
        "JSON only."
    )
    return [{"role": "system", "content": SOLVER_SYSTEM}, {"role": "user", "content": user}]


def build_judge_messages(task_id: str, bundle: dict, condition_rows: list[dict], solutions: dict[str, str]) -> list[dict]:
    solution_blocks = []
    for row in condition_rows:
        condition = row["condition"]
        solution_blocks.append(
            "## Condition: {condition}\n"
            "Exposure metadata: {metadata}\n"
            "Proposed plan:\n{solution}".format(
                condition=condition,
                metadata=json.dumps(condition_metadata(row), ensure_ascii=False),
                solution=solutions[condition],
            )
        )
    user = (
        f"Task id: {task_id}\n\n"
        f"Task instructions:\n{bundle['instructions']}\n\n"
        f"Oracle/reference implementation excerpts:\n{bundle['oracle'] or '(not available)'}\n\n"
        f"Verifier/pass criteria excerpts:\n{bundle['verifier'] or '(not available)'}\n\n"
        "Candidate plans:\n"
        + "\n\n".join(solution_blocks)
        + "\n\nGrade each condition. A score of 8-10 means the plan is likely verifier-ready; "
        "5-7 means partially actionable but missing important verifier details; "
        "0-4 means unlikely to pass. Also compare each condition to no_skill.\n"
        "Return JSON with this exact shape:\n"
        "{"
        '"conditions":[{"condition":"name","readiness_score":0-10,"likely_pass":true/false,'
        '"uses_relevant_skill_evidence":true/false,"major_gaps":["..."],"reason":"..."}],'
        '"ranking":["best_condition", "..."],'
        '"best_condition":"name",'
        '"skill_helped_summary":"one sentence"'
        "}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def call_qwen(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> tuple[str, dict]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
            continue
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            time.sleep(min(2**attempt, 20))
            continue
        raise RuntimeError(f"Qwen API error {response.status_code}: {response.text[:500]}")
    raise RuntimeError(f"Qwen API call failed after {max_retries} retries: {last_error}")


def is_budget_or_access_error(exc: Exception) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in [
            "AccessDenied.Unpurchased",
            "free quota has been exhausted",
            "insufficient",
            "quota",
            "403",
        ]
    )


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(per_condition_rows: list[dict]) -> list[dict]:
    summary = []
    conditions = []
    for row in per_condition_rows:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])
    for condition in conditions:
        group = [row for row in per_condition_rows if row["condition"] == condition]
        summary.append(
            {
                "condition": condition,
                "n": len(group),
                "mean_readiness_score": round(mean(float(row["readiness_score"]) for row in group), 3),
                "likely_pass_rate": round(mean(1.0 if row["likely_pass"] else 0.0 for row in group), 3),
                "mean_delta_vs_no_skill": round(mean(float(row["delta_vs_no_skill"]) for row in group), 3),
                "best_condition_rate": round(mean(1.0 if row["is_best_condition"] else 0.0 for row in group), 3),
                "mean_solver_tokens": round(mean(float(row["solver_tokens"]) for row in group), 1),
                "mean_exposed_skill_count": round(mean(float(row["exposed_skill_count"]) for row in group), 2),
                "mean_complete_gold_coverage": round(mean(float(row["complete_gold_coverage"]) for row in group), 3),
            }
        )
    return summary


def write_score_svg(summary_rows: list[dict], path: Path) -> None:
    width, height = 920, 420
    left, right, top, bottom = 72, 24, 44, 125
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = plot_w / max(len(summary_rows), 1)
    bar_w = gap * 0.48

    def sy(value: float) -> float:
        return top + (1.0 - value / 10.0) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-family="Arial" font-size="18" font-weight="700">RQ4 adaptive Qwen judged readiness</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 2, 4, 6, 8, 10]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick}</text>')
    for i, row in enumerate(summary_rows):
        value = min(float(row["mean_readiness_score"]), 10.0)
        cx = left + gap * i + gap / 2
        h = plot_h * value / 10.0
        x = cx - bar_w / 2
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#0f766e"/>')
        lines.append(f'<text x="{cx:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
        label = f'{row["condition"]} (n={row["n"]})'
        lines.append(
            f'<text x="{cx:.2f}" y="{top + plot_h + 18}" text-anchor="start" font-family="Arial" '
            f'font-size="11" transform="rotate(35 {cx:.2f},{top + plot_h + 18})">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillsbench-root", default="data/raw/skillsbench/tasks")
    parser.add_argument("--exposure-csv", default="data/experiments/rq4_downstream_skill_exposure/per_task_exposure.csv")
    parser.add_argument("--output-dir", default="data/experiments/rq4_adaptive_qwen")
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--max-tasks", type=int, default=6)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--api-key-prompt", action="store_true", help="Read API key with hidden terminal input.")
    parser.add_argument("--base-url", default=DASHSCOPE_COMPATIBLE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens-solver", type=int, default=750)
    parser.add_argument("--max-tokens-judge", type=int, default=1400)
    parser.add_argument("--instructions-cap", type=int, default=3500)
    parser.add_argument("--oracle-cap", type=int, default=4200)
    parser.add_argument("--verifier-cap", type=int, default=4200)
    parser.add_argument("--per-skill-cap", type=int, default=1400)
    parser.add_argument("--skills-total-cap", type=int, default=6500)
    parser.add_argument("--token-budget", type=int, default=250000)
    parser.add_argument("--sleep-between-calls", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = output_dir / "solutions.jsonl"
    judgments_path = output_dir / "judged_tasks.jsonl"
    per_condition_path = output_dir / "per_condition_results.csv"

    tasks, skills = load_skillsbench_tasks(Path(args.skillsbench_root))
    task_by_id = {task["task_id"]: task for task in tasks}
    skill_by_id = {skill["skill_id"]: skill for skill in skills}
    exposure = load_exposure_rows(Path(args.exposure_csv))
    selected_tasks = select_tasks(exposure, args.tasks, args.conditions, args.max_tasks)

    plan_rows = []
    estimated_tokens = 0
    for selected in selected_tasks:
        task_id = selected["task_id"]
        task_dir = Path(args.skillsbench_root) / task_id
        bundle = load_task_bundle(task_dir, args.instructions_cap, args.oracle_cap, args.verifier_cap)
        condition_rows = [exposure[task_id][condition] for condition in args.conditions]
        solver_estimates = []
        for row in condition_rows:
            skill_context = exposure_skill_context(row, skill_by_id, args.per_skill_cap, args.skills_total_cap)
            messages = build_solver_messages(task_id, row["condition"], bundle, skill_context)
            estimate = estimate_tokens(json.dumps(messages, ensure_ascii=False)) + args.max_tokens_solver
            solver_estimates.append(estimate)
        placeholder_solutions = {row["condition"]: '{"approach":"placeholder"}' for row in condition_rows}
        judge_messages = build_judge_messages(task_id, bundle, condition_rows, placeholder_solutions)
        judge_estimate = estimate_tokens(json.dumps(judge_messages, ensure_ascii=False)) + args.max_tokens_judge
        estimated_tokens += sum(solver_estimates) + judge_estimate
        plan_rows.append(
            {
                "task_id": task_id,
                "selection_score": selected["selection_score"],
                "selection_reason": selected["selection_reason"],
                "gold_skill_count": task_by_id[task_id]["gold_skill_count"],
                "conditions": ";".join(args.conditions),
                "estimated_solver_tokens": sum(solver_estimates),
                "estimated_judge_tokens": judge_estimate,
                "estimated_total_tokens": sum(solver_estimates) + judge_estimate,
            }
        )

    write_csv(output_dir / "experiment_plan.csv", plan_rows)
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "model": args.model,
                    "max_tasks": args.max_tasks,
                    "conditions": args.conditions,
                    "estimated_total_tokens": estimated_tokens,
                    "token_budget": args.token_budget,
                    "boundary": "Qwen-judged task-level readiness proxy, not execution-based verifier pass rate.",
                },
                "tasks": plan_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    if args.dry_run:
        print(json.dumps({"selected_tasks": len(selected_tasks), "estimated_total_tokens": estimated_tokens}, indent=2))
        print(f"Wrote {output_dir / 'experiment_plan.csv'}")
        print(f"Wrote {output_dir / 'experiment_plan.json'}")
        return

    api_key = os.environ.get(args.api_key_env) or os.environ.get("ALIYUN_API_KEY", "")
    if args.api_key_prompt and not api_key:
        api_key = getpass.getpass("DashScope API key: ").strip()
    if args.api_key_stdin and not api_key:
        api_key = sys.stdin.readline().strip()
    if not api_key:
        raise SystemExit(
            f"No API key found. Set {args.api_key_env}, ALIYUN_API_KEY, or pass --api-key-stdin."
        )

    completed_solutions = {
        (row["task_id"], row["condition"]): row for row in load_jsonl(solutions_path)
    } if args.resume else {}
    completed_judgments = {row["task_id"]: row for row in load_jsonl(judgments_path)} if args.resume else {}
    spent_tokens = sum(int(row.get("total_tokens", 0)) for row in completed_solutions.values())
    spent_tokens += sum(int(row.get("judge_tokens", 0)) for row in completed_judgments.values())

    for selected in selected_tasks:
        task_id = selected["task_id"]
        if task_id not in task_by_id:
            raise SystemExit(f"Task {task_id} not found under {args.skillsbench_root}")
        task_dir = Path(args.skillsbench_root) / task_id
        bundle = load_task_bundle(task_dir, args.instructions_cap, args.oracle_cap, args.verifier_cap)
        condition_rows = [exposure[task_id][condition] for condition in args.conditions]
        solutions: dict[str, str] = {}

        for row in condition_rows:
            key = (task_id, row["condition"])
            if key in completed_solutions:
                solutions[row["condition"]] = completed_solutions[key]["solution_text"]
                continue
            skill_context = exposure_skill_context(row, skill_by_id, args.per_skill_cap, args.skills_total_cap)
            messages = build_solver_messages(task_id, row["condition"], bundle, skill_context)
            rough_needed = estimate_tokens(json.dumps(messages, ensure_ascii=False)) + args.max_tokens_solver
            if spent_tokens + rough_needed > args.token_budget:
                print(f"Token budget reached before solver {task_id}/{row['condition']}.")
                break
            try:
                solution_text, usage = call_qwen(
                    messages,
                    args.model,
                    api_key,
                    args.base_url,
                    args.temperature,
                    args.max_tokens_solver,
                    args.timeout,
                    args.max_retries,
                )
            except RuntimeError as exc:
                if is_budget_or_access_error(exc):
                    print(f"Stopping after API budget/access error at solver {task_id}/{row['condition']}: {exc}")
                    break
                raise
            total_tokens = int(usage.get("total_tokens", rough_needed))
            spent_tokens += total_tokens
            solution_row = {
                "task_id": task_id,
                "condition": row["condition"],
                "model": args.model,
                "solution_text": solution_text,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": total_tokens,
            }
            append_jsonl(solutions_path, solution_row)
            completed_solutions[key] = solution_row
            solutions[row["condition"]] = solution_text
            print(f"solver {task_id}/{row['condition']}: tokens={total_tokens} spent={spent_tokens}/{args.token_budget}")
            time.sleep(args.sleep_between_calls)

        if len(solutions) != len(condition_rows):
            break
        if task_id in completed_judgments:
            continue

        judge_messages = build_judge_messages(task_id, bundle, condition_rows, solutions)
        rough_needed = estimate_tokens(json.dumps(judge_messages, ensure_ascii=False)) + args.max_tokens_judge
        if spent_tokens + rough_needed > args.token_budget:
            print(f"Token budget reached before judge {task_id}.")
            break
        try:
            judge_raw, judge_usage = call_qwen(
                judge_messages,
                args.model,
                api_key,
                args.base_url,
                0.0,
                args.max_tokens_judge,
                args.timeout,
                args.max_retries,
            )
        except RuntimeError as exc:
            if is_budget_or_access_error(exc):
                print(f"Stopping after API budget/access error at judge {task_id}: {exc}")
                break
            raise
        judge_tokens = int(judge_usage.get("total_tokens", rough_needed))
        spent_tokens += judge_tokens
        judgment = parse_json_object(judge_raw)
        judgment_row = {
            "task_id": task_id,
            "model": args.model,
            "judge_raw": judge_raw,
            "judgment": judgment,
            "judge_tokens": judge_tokens,
            "spent_tokens_after_task": spent_tokens,
        }
        append_jsonl(judgments_path, judgment_row)
        completed_judgments[task_id] = judgment_row
        print(f"judge {task_id}: tokens={judge_tokens} spent={spent_tokens}/{args.token_budget}")
        time.sleep(args.sleep_between_calls)

    per_condition_rows = []
    for task_id, judgment_row in completed_judgments.items():
        judgment = judgment_row.get("judgment") or {}
        conditions = judgment.get("conditions") or []
        if not isinstance(conditions, list):
            continue
        score_by_condition = {}
        for item in conditions:
            if not isinstance(item, dict) or "condition" not in item:
                continue
            try:
                score = float(item.get("readiness_score", 0))
            except Exception:
                score = 0.0
            score_by_condition[str(item["condition"])] = max(0.0, min(score, 10.0))
        no_skill_score = score_by_condition.get("no_skill", 0.0)
        best_condition = str(judgment.get("best_condition") or "")
        for item in conditions:
            if not isinstance(item, dict) or "condition" not in item:
                continue
            condition = str(item["condition"])
            row = exposure[task_id].get(condition, {})
            readiness_score = score_by_condition.get(condition, 0.0)
            per_condition_rows.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "readiness_score": readiness_score,
                    "likely_pass": bool(item.get("likely_pass", False)),
                    "delta_vs_no_skill": readiness_score - no_skill_score,
                    "is_best_condition": condition == best_condition,
                    "uses_relevant_skill_evidence": bool(item.get("uses_relevant_skill_evidence", False)),
                    "major_gaps": "; ".join(str(x) for x in item.get("major_gaps", [])[:4])
                    if isinstance(item.get("major_gaps", []), list)
                    else str(item.get("major_gaps", "")),
                    "reason": str(item.get("reason", ""))[:500],
                    "solver_tokens": completed_solutions.get((task_id, condition), {}).get("total_tokens", 0),
                    "judge_tokens_task_level": judgment_row.get("judge_tokens", 0),
                    "exposed_skill_count": row.get("exposed_skill_count", 0),
                    "covered_gold_count": row.get("covered_gold_count", 0),
                    "gold_skill_count": row.get("gold_skill_count", 0),
                    "complete_gold_coverage": row.get("complete_gold_coverage", 0.0),
                    "extra_skill_count": row.get("extra_skill_count", 0),
                    "context_tokens": row.get("context_tokens", 0),
                }
            )

    if per_condition_rows:
        write_csv(per_condition_path, per_condition_rows)
        summary_rows = summarize(per_condition_rows)
        write_csv(output_dir / "summary.csv", summary_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "model": args.model,
                        "tasks_judged": len(completed_judgments),
                        "conditions": args.conditions,
                        "token_budget": args.token_budget,
                        "tokens_spent_observed": spent_tokens,
                        "boundary": "Qwen-judged readiness proxy, not execution-based verifier pass rate.",
                    },
                    "summary": summary_rows,
                    "best_condition_counts": Counter(
                        str((row.get("judgment") or {}).get("best_condition", ""))
                        for row in completed_judgments.values()
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        write_score_svg(summary_rows, output_dir / "readiness_score_bar.svg")
        print(f"Wrote {per_condition_path}")
        print(f"Wrote {output_dir / 'summary.csv'}")
        print(f"Wrote {output_dir / 'summary.json'}")
        print(f"Wrote {output_dir / 'readiness_score_bar.svg'}")
    else:
        print("No judged rows yet; resume when more budget/API time is available.")


if __name__ == "__main__":
    main()
