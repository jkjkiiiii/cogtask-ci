"""Portable static reports from run outputs, without access to source text."""
from collections import Counter
from pathlib import Path
import csv
import html
import json


def render_report(run_dir):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    root = Path(run_dir)
    with (root / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with (root / "selections.csv").open(encoding="utf-8-sig", newline="") as f:
        selections = list(csv.DictReader(f))
    coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    with (root / "contrasts.csv").open(encoding="utf-8-sig", newline="") as f:
        contrasts = list(csv.DictReader(f))
    has_matched = any(r["arm"].endswith("matched") for r in rows)
    nrows = 2 if has_matched else 1
    fig = Figure(figsize=(11, 4.3 * nrows), constrained_layout=True)
    FigureCanvasAgg(fig)
    panels = fig.subplots(nrows, 2, squeeze=False)
    names = {"text": "Full text", "quantity": "Quantity only", "text_matched": "Matched full text", "quantity_matched": "Matched quantity", "fixed_text": "Per-task prefix crop"}
    for arm in sorted({r["arm"] for r in rows}):
        axes = panels[1 if has_matched and arm not in {"text", "quantity"} else 0]
        group = sorted((r for r in rows if r["arm"] == arm), key=lambda r: int(r["k"]))
        x = [int(r["k"]) for r in group]
        for ax, val, low, high in [(axes[0], "auroc", "ci_low", "ci_high"), (axes[1], "delta_vs_single", "delta_ci_low", "delta_ci_high")]:
            y = [float(r[val]) for r in group]
            line, = ax.plot(x, y, marker="o", label=names.get(arm, arm))
            ax.fill_between(x, [float(r[low]) for r in group], [float(r[high]) for r in group], color=line.get_color(), alpha=.12)
            ax.set_xlabel("Number of tasks")
            ax.set_xticks(x)
    for i, axes in enumerate(panels):
        population = "Primary population" if i == 0 else "Crop-eligible matched population"
        axes[0].set(ylabel="AUROC", ylim=(0, 1), title=population)
        axes[1].set(ylabel="Paired AUROC difference", title="Gain over inner-selected single task")
        axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
        axes[0].legend(fontsize=8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(root / ("task_budget." + ext), dpi=160)
    fig.clear()
    counts = Counter((r["arm"], int(r["k"]), r["selected_tasks"]) for r in selections)
    frequency = [dict(arm=a, k=k, selected_tasks=s, count=n,
                      fraction=n / sum(r["arm"] == a and int(r["k"]) == k for r in selections))
                 for (a, k, s), n in sorted(counts.items())]
    with (root / "selection_frequency.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frequency[0]))
        writer.writeheader()
        writer.writerows(frequency)
    def table(records):
        if not records:
            return ""
        keys = list(records[0])
        return "<table><thead><tr>" + "".join("<th>" + html.escape(k) + "</th>" for k in keys) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(str(r[k])) + "</td>" for k in keys) + "</tr>" for r in records) + "</tbody></table>"
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>CogTask-CI report</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem;color:#203044}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:6px;border:1px solid #ccd}img{width:100%}pre{white-space:pre-wrap}</style>
<h1>CogTask-CI evaluation</h1><p>Research procedure evaluation. Labels are user-defined; this report does not establish clinical validity.</p>
<p>Intervals: 95% class-stratified paired participant bootstrap, conditional on realized models and folds. They do not include model-development or task-protocol selection uncertainty. Task selection frequencies describe stability within this run, not causal importance.</p>
<p>Task counts share the complete-task evaluation set within each arm. Matched arms use the same crop-eligible subjects; compare their summaries separately from the primary population. Incomplete subjects can contribute task-eligible training records.</p>
<p>The optional fixed_text arm crops N tokens PER TASK: a k-task combination uses k*N tokens, rather than a constant total across task counts. Probabilities are uncalibrated class-weighted model scores.</p>
<h2>Coverage</h2><pre>''' + html.escape(json.dumps(coverage, indent=2)) + "</pre><h2>Task budget</h2><img src='task_budget.png' alt='Task budget curve'><h2>Metrics</h2>" + table(rows) + "<h2>Selected combinations</h2>" + table(frequency) + "<p>Local outputs include participant-derived records; review before sharing.</p></html>"
    page = page.replace("<h2>Selected combinations</h2>", "<h2>Paired arm differences</h2>" + table(contrasts) + "<h2>Selected combinations</h2>")
    (root / "report.html").write_text(page, encoding="utf-8")
    return root / "report.html"
