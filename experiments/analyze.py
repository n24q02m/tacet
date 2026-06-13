"""Turn experiment results into the paper's figures and LaTeX tables.

    python experiments/analyze.py --results experiments/results --out paper

Reads `summary.json` (from `--results`) and writes vector PDF figures to
`<out>/figures/` and `\\input`-able LaTeX tables + the merged `macros.tex` to
`<out>/results/`. The default `--out paper` targets the directories the paper
actually `\\input`s (`paper/figures`, `paper/results`); the macro write merges
into the shared `macros.tex` so the other runners' macros are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SYSTEMS = ["llm_only", "symbolic_only", "cache_cascade", "static_cascade", "tacet"]
LABEL = {
    "llm_only": "LLM-only",
    "symbolic_only": "Symbolic-only",
    "cache_cascade": "Cache cascade",
    "static_cascade": "Static cascade",
    "tacet": "TACET",
}
COLOR = {
    "llm_only": "#c0392b",
    "symbolic_only": "#7f8c8d",
    "cache_cascade": "#e67e22",
    "static_cascade": "#2980b9",
    "tacet": "#27ae60",
}


def _merge_macros(path: Path, macros: dict[str, str]) -> None:
    """Write ``\\renewcommand`` lines into ``path``, replacing any existing
    definition of the same macro while preserving every other line.

    The paper's ``results/macros.tex`` is shared by several result runners
    (this script plus ``run_rule_precision``/``run_audit_eval``/
    ``run_proofwriter``); merging rather than overwriting lets each runner
    update only its own macros.
    """
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in existing if not any(f"{{{name}}}" in ln for name in macros)]
    for name, value in macros.items():
        kept.append(f"\\renewcommand{{{name}}}{{{value}}}")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def fig_cost_trajectory(summary: dict, out: Path) -> None:
    e2 = summary["E2"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for sysname in SYSTEMS:
        if sysname not in e2:
            continue
        traj = e2[sysname]
        ax.plot(
            range(1, len(traj) + 1), traj, label=LABEL[sysname], color=COLOR[sysname], linewidth=2
        )
    ax.set_xlabel("queries processed")
    ax.set_ylabel("cumulative cost (USD)")
    ax.set_title("Cumulative cost over a streamed workload")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_cost_trajectory.pdf")
    plt.close(fig)


def fig_pareto(summary: dict, out: Path) -> None:
    e3, e1 = summary["E3"], summary["E1"]
    pts = sorted(
        ((v["accuracy"]["mean"], v["cost"]["mean"], tag) for tag, v in e3.items()),
        key=lambda x: x[0],
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(
        [a for a, _, _ in pts],
        [c for _, c, _ in pts],
        "-o",
        color=COLOR["tacet"],
        label="TACET (threshold sweep)",
        linewidth=2,
    )
    for sysname in ("llm_only", "cache_cascade", "symbolic_only"):
        if sysname in e1:
            ax.scatter(
                [e1[sysname]["accuracy"]["mean"]],
                [e1[sysname]["cost"]["mean"]],
                color=COLOR[sysname],
                s=60,
                zorder=5,
                label=LABEL[sysname],
            )
    ax.set_xlabel("accuracy")
    ax.set_ylabel("blended cost (USD)")
    ax.set_title("Cost–accuracy frontier")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_pareto.pdf")
    plt.close(fig)


def fig_sensitivity(summary: dict, out: Path) -> None:
    e5 = summary["E5"]

    def series(system: str) -> tuple[list[float], list[float]]:
        rows = sorted(
            (float(tag.split("=")[1]), v["cost"]["mean"])
            for tag, v in e5.items()
            if tag.startswith(system)
        )
        return [r for r, _ in rows], [c for _, c in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for system, color in (("tacet", COLOR["tacet"]), ("cache_cascade", COLOR["cache_cascade"])):
        xs, ys = series(system)
        if xs:
            ax.plot(xs, ys, "-o", color=color, linewidth=2, label=LABEL[system])
    ax.set_xlabel("workload repeat rate")
    ax.set_ylabel("blended cost (USD)")
    ax.set_title("Sensitivity to workload repetition")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_sensitivity.pdf")
    plt.close(fig)


def fig_noise(summary: dict, out: Path) -> None:
    e6 = summary["E6"]

    def series(system: str, metric: str) -> tuple[list[float], list[float]]:
        rows = sorted(
            (float(tag.split("=")[1]), v[metric]["mean"])
            for tag, v in e6.items()
            if tag.startswith(system)
        )
        return [r for r, _ in rows], [c for _, c in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for system, color in (("tacet", COLOR["tacet"]), ("llm_only", COLOR["llm_only"])):
        xs, ys = series(system, "accuracy")
        if xs:
            ax.plot(xs, ys, "-o", color=color, linewidth=2, label=LABEL[system])
    ax.set_xlabel("teacher error rate")
    ax.set_ylabel("end-to-end accuracy")
    ax.set_title("Robustness to a noisy teacher")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_noise.pdf")
    plt.close(fig)


def table_main(summary: dict) -> str:
    e1 = summary["E1"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"System & Cost (USD) & Accuracy & Latency (ms) & \% no-LLM \\",
        r"\midrule",
    ]
    for s in SYSTEMS:
        if s not in e1:
            continue
        d = e1[s]
        tf = d["tier_fraction"]
        no_llm = 100 * (tf.get("1", 0) + tf.get("2", 0))
        cost = f"{d['cost']['mean']:.3f}\\,$\\pm$\\,{d['cost']['ci95']:.3f}"
        acc = f"{d['accuracy']['mean']:.3f}"
        lat = f"{d['latency_ms']['mean']:.0f}"
        row = f"{LABEL[s]} & {cost} & {acc} & {lat} & {no_llm:.0f} \\\\"
        if s == "tacet":
            row = r"\textbf{" + LABEL[s] + r"}" + row[len(LABEL[s]) :]
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_ablation(summary: dict) -> str:
    e4 = summary["E4"]
    order = ["full", "no_writeback", "no_kge_aug", "no_rule_synth", "no_distill"]
    name = {
        "full": "TACET (full)",
        "no_writeback": "-- fact write-back",
        "no_kge_aug": "-- KGE augmentation",
        "no_rule_synth": "-- rule synthesis",
        "no_distill": "-- all distillation (static)",
    }
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Configuration & Cost (USD) & Accuracy \\",
        r"\midrule",
    ]
    for tag in order:
        if tag not in e4:
            continue
        d = e4[tag]
        lines.append(f"{name[tag]} & {d['cost']['mean']:.3f} & {d['accuracy']['mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()

    summary = json.loads((Path(args.results) / "summary.json").read_text())
    figures = Path(args.out) / "figures"
    results = Path(args.out) / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    fig_cost_trajectory(summary, figures)
    fig_pareto(summary, figures)
    fig_sensitivity(summary, figures)
    fig_noise(summary, figures)
    (results / "tab_main.tex").write_text(table_main(summary), encoding="utf-8")
    (results / "tab_ablation.tex").write_text(table_ablation(summary), encoding="utf-8")

    e1 = summary["E1"]
    ak, llm, cache = e1["tacet"], e1["llm_only"], e1["cache_cascade"]
    seeds = summary.get("_meta", {}).get("seeds", 12)
    digest = {
        "tacet_cost": ak["cost"]["mean"],
        "tacet_accuracy": ak["accuracy"]["mean"],
        "llm_cost": llm["cost"]["mean"],
        "cache_cost": cache["cost"]["mean"],
        "cost_reduction_x": llm["cost"]["mean"] / ak["cost"]["mean"],
        "seeds": seeds,
    }
    (results / "digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")

    # LaTeX result macros consumed by paper/main.tex. Merge (not overwrite) into
    # the shared macros file so the macros written by the other result runners
    # (run_rule_precision, run_audit_eval, run_proofwriter) are preserved.
    _merge_macros(
        results / "macros.tex",
        {
            r"\akCost": f"{ak['cost']['mean']:.2f}",
            r"\akAcc": f"{ak['accuracy']['mean']:.3f}",
            r"\llmCost": f"{llm['cost']['mean']:.1f}",
            r"\cacheCost": f"{cache['cost']['mean']:.1f}",
            r"\costFactor": f"{llm['cost']['mean'] / ak['cost']['mean']:.1f}",
            r"\seeds": str(seeds),
        },
    )
    print("figures + tables written:")
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
