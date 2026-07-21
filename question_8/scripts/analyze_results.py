#!/usr/bin/env python3
"""Create publication-ready SVG figures from Question 8 CSV/JSON outputs."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


COLORS = {
    "hybrid_top10": "#64748b",
    "graph_semantic_seed3": "#2563eb",
    "graph_metadata_seed3": "#f59e0b",
    "graph_cv_corequired_seed3": "#8b5cf6",
    "graph_cv_all_seed3": "#0f766e",
    "graph_transductive_upper_bound": "#dc2626",
}
LABELS = {
    "hybrid_top10": "Hybrid Top-10",
    "graph_semantic_seed3": "Semantic graph",
    "graph_metadata_seed3": "Metadata graph",
    "graph_cv_corequired_seed3": "CV co-required",
    "graph_cv_all_seed3": "CV all edges",
    "graph_transductive_upper_bound": "Transductive upper bound",
}
METHODS = list(COLORS)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_metric_chart(rows: list[dict], path: Path) -> None:
    by_method = {r["method"]: r for r in rows if r["scope"] == "multi_skill_only"}
    w, h = 1120, 620
    left, top, plot_w, plot_h = 90, 80, 900, 410
    metrics = [("complete_gold_coverage", "Complete coverage"), ("gold_recall", "Gold recall")]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="90" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#0f172a">Multi-skill retrieval: coverage and recall</text>',
        '<text x="90" y="63" font-family="Arial" font-size="13" fill="#475569">Mean over 61 multi-skill tasks; whiskers are task-bootstrap 95% CIs; red is a leakage-labelled upper bound</text>',
    ]
    for tick in (0, .2, .4, .6, .8, 1.0):
        y = top + plot_h * (1 - tick)
        lines += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#cbd5e1" stroke-width="1"/>',
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#475569">{tick:.1f}</text>',
        ]
    group_w = plot_w / len(METHODS)
    bar_w = 31
    for i, method in enumerate(METHODS):
        row = by_method[method]
        cx = left + group_w * (i + .5)
        for j, (metric, _) in enumerate(metrics):
            value = float(row[metric])
            x = cx + (j - .5) * (bar_w + 5) - bar_w / 2
            y = top + plot_h * (1 - value)
            fill = COLORS[method] if j == 0 else "#ffffff"
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top+plot_h-y:.1f}" rx="3" fill="{fill}" stroke="{COLORS[method]}" stroke-width="3"/>')
            lo = float(row[f"{metric}_ci_low"])
            hi = float(row[f"{metric}_ci_high"])
            ey0 = top + plot_h * (1 - max(0.0, lo))
            ey1 = top + plot_h * (1 - min(1.0, hi))
            ex = x + bar_w / 2
            lines += [
                f'<line x1="{ex:.1f}" y1="{ey0:.1f}" x2="{ex:.1f}" y2="{ey1:.1f}" stroke="#0f172a" stroke-width="2"/>',
                f'<line x1="{ex-5:.1f}" y1="{ey0:.1f}" x2="{ex+5:.1f}" y2="{ey0:.1f}" stroke="#0f172a" stroke-width="2"/>',
                f'<line x1="{ex-5:.1f}" y1="{ey1:.1f}" x2="{ex+5:.1f}" y2="{ey1:.1f}" stroke="#0f172a" stroke-width="2"/>',
            ]
            lines.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#0f172a">{value:.3f}</text>')
        label = LABELS[method]
        lines.append(f'<text x="{cx:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="11" fill="#334155" transform="rotate(24 {cx:.1f},{top+plot_h+25})">{html.escape(label)}</text>')
    lines += [
        '<rect x="840" y="25" width="14" height="14" fill="#334155"/><text x="861" y="37" font-family="Arial" font-size="12">Complete coverage</text>',
        '<rect x="840" y="48" width="14" height="14" fill="white" stroke="#334155" stroke-width="2"/><text x="861" y="60" font-family="Arial" font-size="12">Gold recall</text>',
        '<line x1="840" y1="78" x2="854" y2="78" stroke="#0f172a" stroke-width="2"/><text x="861" y="82" font-family="Arial" font-size="12">95% task-bootstrap CI</text>',
        '</svg>',
    ]
    save(path, lines)


def tradeoff_chart(rows: list[dict], path: Path) -> None:
    by_method = {r["method"]: r for r in rows if r["scope"] == "multi_skill_only"}
    w, h = 900, 600
    left, top, plot_w, plot_h = 100, 75, 680, 410
    xmin, xmax, ymin, ymax = .34, .75, .08, .28
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * plot_w
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * plot_h
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="100" y="35" font-family="Arial" font-size="23" font-weight="700" fill="#0f172a">Coverage–pollution trade-off</text>',
        '<text x="100" y="58" font-family="Arial" font-size="13" fill="#64748b">Means over 61 multi-skill tasks; horizontal/vertical whiskers are 95% task-bootstrap CIs</text>',
    ]
    for x in (.35, .40, .45, .50, .55, .60, .65, .70, .75):
        xx = sx(x)
        lines += [f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#e2e8f0"/>', f'<text x="{xx:.1f}" y="{top+plot_h+21}" text-anchor="middle" font-family="Arial" font-size="12">{x:.2f}</text>']
    for y in (.08, .12, .16, .20, .24, .28):
        yy = sy(y)
        lines += [f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e2e8f0"/>', f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.2f}</text>']
    for i, method in enumerate(METHODS):
        row = by_method[method]
        x, y = float(row["gold_recall"]), float(row["skill_precision"])
        xlo, xhi = float(row["gold_recall_ci_low"]), float(row["gold_recall_ci_high"])
        ylo, yhi = float(row["skill_precision_ci_low"]), float(row["skill_precision_ci_high"])
        lines += [
            f'<line x1="{sx(xlo):.1f}" y1="{sy(y):.1f}" x2="{sx(xhi):.1f}" y2="{sy(y):.1f}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<line x1="{sx(xlo):.1f}" y1="{sy(y)-5:.1f}" x2="{sx(xlo):.1f}" y2="{sy(y)+5:.1f}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<line x1="{sx(xhi):.1f}" y1="{sy(y)-5:.1f}" x2="{sx(xhi):.1f}" y2="{sy(y)+5:.1f}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<line x1="{sx(x):.1f}" y1="{sy(ylo):.1f}" x2="{sx(x):.1f}" y2="{sy(yhi):.1f}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<line x1="{sx(x)-5:.1f}" y1="{sy(ylo):.1f}" x2="{sx(x)+5:.1f}" y2="{sy(ylo):.1f}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<line x1="{sx(x)-5:.1f}" y1="{sy(yhi):.1f}" x2="{sx(x)+5:.1f}" y2="{sy(yhi):.1f}" stroke="{COLORS[method]}" stroke-width="2"/>',
        ]
        lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="8" fill="{COLORS[method]}" stroke="white" stroke-width="2"/>')
        dy = -12 if i % 2 == 0 else 20
        lines.append(f'<text x="{sx(x)+10:.1f}" y="{sy(y)+dy:.1f}" font-family="Arial" font-size="11" fill="#0f172a">{html.escape(LABELS[method])}</text>')
    lines += [
        f'<text x="{left+plot_w/2}" y="{h-35}" text-anchor="middle" font-family="Arial" font-size="14">Gold recall</text>',
        f'<text x="25" y="{top+plot_h/2}" transform="rotate(-90 25,{top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="14">Skill precision</text>',
        '</svg>',
    ]
    save(path, lines)


def edge_chart(stats: dict, path: Path) -> None:
    values = list(stats["content_edge_counts"].items()) + [("co_required (upper)", stats["transductive_corequired_edges"])]
    w, h = 900, 520
    left, top, plot_w, plot_h = 110, 75, 650, 330
    maxv = max(v for _, v in values)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="90" y="35" font-family="Arial" font-size="23" font-weight="700">Graph edge inventory</text>',
        '<text x="90" y="58" font-family="Arial" font-size="13" fill="#64748b">Content-derived graph is dense; reusable supervised relations are sparse</text>',
    ]
    bar_w = 95
    gap = plot_w / len(values)
    colors = ["#2563eb", "#f59e0b", "#0f766e", "#dc2626"]
    for i, ((label, value), color) in enumerate(zip(values, colors)):
        x = left + gap * (i + .5) - bar_w / 2
        bar_h = plot_h * value / maxv
        y = top + plot_h - bar_h
        lines += [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="5" fill="{color}"/>',
            f'<text x="{x+bar_w/2:.1f}" y="{y-9:.1f}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{value:,}</text>',
            f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(label)}</text>',
        ]
    lines.append('</svg>')
    save(path, lines)


def delta_chart(paired: list[dict], path: Path) -> None:
    wanted = ["graph_semantic_seed3", "graph_cv_corequired_seed3", "graph_cv_all_seed3", "graph_transductive_upper_bound"]
    by_method = {r["method"]: r for r in paired}
    w, h = 980, 520
    left, top, plot_w, plot_h = 270, 70, 590, 340
    xmin, xmax = -.06, .25
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * plot_w
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="75" y="35" font-family="Arial" font-size="23" font-weight="700">Paired change in gold recall vs Hybrid Top-10</text>',
        '<text x="75" y="58" font-family="Arial" font-size="13" fill="#64748b">Paired task means; intervals are paired task-bootstrap 95% CIs (61 multi-skill tasks)</text>',
        f'<line x1="{sx(0):.1f}" y1="{top-10}" x2="{sx(0):.1f}" y2="{top+plot_h}" stroke="#0f172a" stroke-dasharray="5,5"/>',
    ]
    for i, method in enumerate(wanted):
        row = by_method[method]
        mean = float(row["delta_gold_recall"])
        lo = float(row["delta_gold_recall_ci_low"])
        hi = float(row["delta_gold_recall_ci_high"])
        y = top + 48 + i * 74
        lines += [
            f'<text x="{left-18}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="13">{html.escape(LABELS[method])}</text>',
            f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{COLORS[method]}" stroke-width="5"/>',
            f'<line x1="{sx(lo):.1f}" y1="{y-7}" x2="{sx(lo):.1f}" y2="{y+7}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<line x1="{sx(hi):.1f}" y1="{y-7}" x2="{sx(hi):.1f}" y2="{y+7}" stroke="{COLORS[method]}" stroke-width="2"/>',
            f'<circle cx="{sx(mean):.1f}" cy="{y}" r="7" fill="{COLORS[method]}" stroke="white" stroke-width="2"/>',
            f'<text x="{sx(hi)+10:.1f}" y="{y+4}" font-family="Arial" font-size="11">{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]</text>',
        ]
    for x in (-.05, 0, .05, .10, .15, .20, .25):
        lines.append(f'<text x="{sx(x):.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="12">{x:+.2f}</text>')
    lines.append('</svg>')
    save(path, lines)


def main() -> None:
    q8 = Path(__file__).resolve().parents[1]
    results = q8 / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = read_csv(results / "summary.csv")
    paired = read_csv(results / "paired_comparisons.csv")
    stats = json.loads((results / "graph_stats.json").read_text(encoding="utf-8"))
    grouped_metric_chart(summary, figures / "coverage_recall_comparison.svg")
    tradeoff_chart(summary, figures / "recall_precision_tradeoff.svg")
    edge_chart(stats, figures / "graph_edge_inventory.svg")
    delta_chart(paired, figures / "paired_recall_delta.svg")
    data_dictionary = """# Question 8 Figure Data Dictionary

All figures are generated by `scripts/analyze_results.py` from the frozen CSV/JSON files in the same results directory. No values are manually entered into the SVGs.

| Figure | Source | Scope / n | Statistic | Formula / interpretation |
|---|---|---:|---|---|
| `coverage_recall_comparison.svg` | `summary.csv` | `scope=multi_skill_only`, `n_tasks=61` | Mean + 95% CI | Bars use `complete_gold_coverage` and `gold_recall`; whiskers use the corresponding `_ci_low`/`_ci_high` columns from 2,000 task-level bootstrap resamples. |
| `recall_precision_tradeoff.svg` | `summary.csv` | `scope=multi_skill_only`, `n_tasks=61` | Mean + marginal 95% CIs | x=`gold_recall`, y=`skill_precision`; horizontal/vertical intervals use the matching CI columns. This is descriptive, not a significance test. |
| `paired_recall_delta.svg` | `paired_comparisons.csv` | 61 paired multi-skill tasks | Paired mean delta + 95% CI | `method - hybrid_top10` for `gold_recall`; intervals are bootstrap CIs over task-level paired differences. Zero is the no-change reference. |
| `graph_edge_inventory.svg` | `graph_stats.json` | 5,000-node candidate graph | Edge-record count | Counts are `content_edge_counts` for the reduced graph and `transductive_corequired_edges` for the explicitly labelled upper-bound relation set. These are graph-construction diagnostics, not retrieval performance. |

The complete per-task observations are in `per_task_results.csv`; the graph records are in `data/graph_edges.csv`. A chart should not be interpreted as evidence beyond the scope and statistic stated above.
"""
    (results / "figure_data_dictionary.md").write_text(data_dictionary, encoding="utf-8")
    print(f"Wrote four figures to {figures}")


if __name__ == "__main__":
    main()
