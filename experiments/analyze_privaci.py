"""Generate reproducible LaTeX macros + a table from the PrivaCI compliance matrix.

Reads the three committed compliance-matrix JSONs
(``experiments/results/privaci_matrix_<teacher>.json`` for teacher in
gemini/grok/claude) and emits two generated LaTeX files:

- ``paper/results/macros_privaci.tex`` -- ``\newcommand`` definitions for every
  per-teacher amortisation/accuracy number plus two global scalars.
- ``paper/results/tab_privaci.tex`` -- a self-contained booktabs table float.

Both are pure functions of the committed JSON, so the paper's compliance
numbers are regenerable with::

    uv run python experiments/analyze_privaci.py
"""

import argparse
import json
from pathlib import Path

# teacher key in the JSON -> LaTeX-safe token (letters only, capitalised) and
# the literal model-name string used in the table.
TEACHERS = {
    "gemini": ("Gemini", "gemini-3.5-flash"),
    "claude": ("Claude", "claude-sonnet-4.6"),
    "grok": ("Grok", "grok-4.3"),
}


def llm_only_article_f1_mean(runs):
    """Mean over seeds of the LLM-only baseline's article micro-F1."""
    f1s = [run["stream"]["llm_only"]["article_micro_f1"] for run in runs]
    return sum(f1s) / len(f1s)


def load_matrices(results_dir):
    """Load the three teacher matrices keyed by teacher name."""
    matrices = {}
    for teacher in TEACHERS:
        path = results_dir / f"privaci_matrix_{teacher}.json"
        matrices[teacher] = json.loads(path.read_text(encoding="utf-8"))
    return matrices


def build_macros(matrices):
    """Return the macros_privaci.tex body as a string."""
    lines = []
    distinct_max = 0
    spend_total = 0.0
    for teacher, (token, _model) in TEACHERS.items():
        data = matrices[teacher]
        agg = data["aggregate"][teacher]
        full, cache, nl = agg["full"], agg["cache"], agg["nl_strategy"]
        runs = data["runs"]
        lines += [
            f"\\newcommand{{\\privFull{token}}}{{{full['amortisation_mean']:.2f}}}",
            f"\\newcommand{{\\privFullStd{token}}}{{{full['amortisation_std']:.2f}}}",
            f"\\newcommand{{\\privCache{token}}}{{{cache['amortisation_mean']:.2f}}}",
            f"\\newcommand{{\\privNl{token}}}{{{nl['amortisation_mean']:.2f}}}",
            f"\\newcommand{{\\privAccFull{token}}}{{{full['verdict_acc_mean']:.2f}}}",
            f"\\newcommand{{\\privArtFOneFull{token}}}{{{full['article_f1_mean']:.2f}}}",
            f"\\newcommand{{\\privArtFOneLlm{token}}}{{{llm_only_article_f1_mean(runs):.2f}}}",
        ]
        distinct_max = max(distinct_max, max(r["distinct_patterns"] for r in runs))
        spend_total += sum(r["total_measured_spend_usd"] for r in runs)
    lines.append(f"\\newcommand{{\\privDistinct}}{{{distinct_max}}}")
    lines.append(f"\\newcommand{{\\privSpendTotal}}{{{spend_total:.1f}}}")
    return "\n".join(lines) + "\n"


def build_table(matrices):
    """Return the tab_privaci.tex body as a self-contained table float."""
    rows = []
    for teacher, (_token, model) in TEACHERS.items():
        agg = matrices[teacher]["aggregate"][teacher]
        runs = matrices[teacher]["runs"]
        full, cache, nl = agg["full"], agg["cache"], agg["nl_strategy"]
        art_full = full["article_f1_mean"]
        art_llm = llm_only_article_f1_mean(runs)
        rows.append(
            f"{model} & "
            f"${full['amortisation_mean']:.2f}\\pm{full['amortisation_std']:.2f}$ & "
            f"${cache['amortisation_mean']:.2f}$ & "
            f"${nl['amortisation_mean']:.2f}$ & "
            f"${full['verdict_acc_mean']:.2f}$ & "
            f"${art_full:.2f}$ / ${art_llm:.2f}$ \\\\"
        )
    caption = (
        "Compliance-domain cost amortisation on PrivaCI-Bench GDPR "
        "($n{=}300$, 3 seeds, iso-accuracy). Full online distillation beats an "
        "exact-match cache across three frontier teachers; the "
        "in-context-strategy baseline (nl\\_strategy) does not amortise."
    )
    header = (
        "Teacher & full $\\times$ & cache $\\times$ & nl\\_strategy $\\times$ & "
        "acc (full) & article-F1 (full / LLM-only) \\\\"
    )
    body = "\n".join(
        [
            "\\begin{table}[t]\\centering",
            f"\\caption{{{caption}}}",
            "\\label{tab:privaci}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            header,
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return body + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/results"))
    parser.add_argument("--out", type=Path, default=Path("paper/results"))
    args = parser.parse_args()

    matrices = load_matrices(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "macros_privaci.tex").write_text(build_macros(matrices), encoding="utf-8")
    (args.out / "tab_privaci.tex").write_text(build_table(matrices), encoding="utf-8")


if __name__ == "__main__":
    main()
