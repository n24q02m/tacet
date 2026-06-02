"""Gamma (confidence-threshold) sensitivity sweep for the rule miner.

A reviewer asked for the precision/recall tradeoff of the confidence filter. We
sweep the miner's confidence threshold ``gamma`` and, at each value, run the
full distillation cascade on the deterministic synthetic KGQA benchmark (oracle
teacher, no API calls) and measure:

* **#installed**         -- how many rules clear ``gamma`` and get added to
                            Tier~1 (recall side: a higher bar installs fewer
                            rules).
* **mean world-precision** -- the mean precision of the installed rules on the
                            FULL ground-truth graph (precision side: a higher
                            bar should keep cleaner rules).

This is the classic precision/recall knob: as ``gamma`` rises fewer rules
install but the survivors are more world-correct, because the last rules to be
filtered out are the noisy near-functional (homophily) relations whose
answered-head confidence sits just under the bar.

Results are averaged over a small seed set (mirroring
``run_rule_precision.py``) and rendered to ``paper/figures/fig_gamma_sensitivity.pdf``.

    uv run python experiments/run_gamma_sensitivity.py [--seeds 5] \
        [--gammas 0.5 0.7 0.8 0.9 0.95 0.99] [--workload-size 300]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tacet.cascade.router import TACET
from tacet.core.graph import WorldGraph
from tacet.distill.distill import mine_rules_with_stats
from tacet.eval import benchmark
from tacet.eval.benchmark import BenchmarkConfig
from tacet.eval.rule_precision import rule_world_precision
from tacet.llm.teacher import OracleTeacher
from tacet.serve.config import CascadeConfig, KGEConfig

_RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
_FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"

_DEFAULT_GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def _ground_truth_graph(bench: benchmark.Benchmark) -> WorldGraph:
    """Complete truth table over every entity (see run_rule_precision.py)."""
    g = bench.graph.copy()
    for (head, relation), tails in bench.truth.items():
        for tail in tails:
            g.add_edge(head, relation, tail)
    return g


def _eval_gamma_seed(gamma: float, seed: int, workload_size: int) -> tuple[int, float]:
    """Run the cascade with min_confidence=gamma; return (#installed, mean world-prec).

    Mirrors ``run_rule_precision._evaluate_synthetic`` but parametrises the miner
    confidence threshold and reports only the two sweep quantities.
    """
    cfg = BenchmarkConfig(seed=seed, workload_size=workload_size)
    bench = benchmark.generate(cfg)
    gt_graph = _ground_truth_graph(bench)

    teacher = OracleTeacher(bench.oracle, error_rate=0.0, entity_pool=bench.entity_pool, seed=seed)
    cascade = CascadeConfig(min_confidence=gamma, kge=KGEConfig(epochs=55, dim=64))
    ak = TACET(
        bench.graph.copy(), bench.ontology, teacher, rules=list(bench.given_rules), config=cascade
    )
    ak.warmup(calibration=bench.calibration)

    for idx, (h, r) in enumerate(bench.workload):
        ak.ask(h, r)
        if (idx + 1) % 100 == 0:
            ak.consolidate()

    distiller = ak.distiller
    synth_relations = sorted(
        {n[len("syn:") :].split("<=")[0] for n in ak.synthesised_rules if n.startswith("syn:")}
    )

    installed_rules = []
    for relation in synth_relations:
        rules, _ = mine_rules_with_stats(
            ak.graph,
            distiller.teacher_facts,
            relation,
            min_confidence=gamma,
            min_support=distiller.min_support,
            complete_heads=distiller._complete_heads.get(relation, set()),
            allowed_body=distiller.base_relations or None,
        )
        installed_rules.extend(rules)

    precisions = [rule_world_precision(m.rule, gt_graph) for m in installed_rules]
    n_installed = len(installed_rules)
    mean_prec = statistics.fmean(precisions) if precisions else 0.0
    return n_installed, mean_prec


def _sweep(gammas: list[float], seeds: list[int], workload_size: int) -> list[dict]:
    rows = []
    for gamma in gammas:
        per_seed_installed = []
        per_seed_prec = []
        for seed in seeds:
            n_installed, mean_prec = _eval_gamma_seed(gamma, seed, workload_size)
            per_seed_installed.append(n_installed)
            # only count a mean precision when at least one rule installed
            if n_installed > 0:
                per_seed_prec.append(mean_prec)
        installed_mean = statistics.fmean(per_seed_installed)
        prec_mean = statistics.fmean(per_seed_prec) if per_seed_prec else 0.0
        rows.append(
            {
                "gamma": gamma,
                "installed_mean": installed_mean,
                "installed_per_seed": per_seed_installed,
                "mean_world_precision": prec_mean,
                "precision_per_seed": per_seed_prec,
            }
        )
        print(
            f"gamma={gamma:<5} installed={installed_mean:>5.1f} "
            f"mean_world_precision={prec_mean:.4f} "
            f"(installed/seed={per_seed_installed})"
        )
    return rows


def _plot(rows: list[dict], path: Path) -> None:
    gammas = [r["gamma"] for r in rows]
    installed = [r["installed_mean"] for r in rows]
    precision = [r["mean_world_precision"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
    color1 = "#1f77b4"
    color2 = "#d62728"

    ax1.set_xlabel(r"confidence threshold $\gamma$")
    ax1.set_ylabel("rules installed", color=color1)
    line1 = ax1.plot(gammas, installed, "o-", color=color1, label="rules installed")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("mean world-precision", color=color2)
    line2 = ax2.plot(gammas, precision, "s--", color=color2, label="mean world-precision")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.7, 1.02)

    lines = line1 + line2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..seeds-1)")
    ap.add_argument("--gammas", type=float, nargs="+", default=_DEFAULT_GAMMAS)
    ap.add_argument("--workload-size", type=int, default=300)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    rows = _sweep(args.gammas, seeds, args.workload_size)

    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "gamma_sensitivity.json").write_text(
        json.dumps({"seeds": seeds, "sweep": rows}, indent=2), encoding="utf-8"
    )
    _plot(rows, _FIGURES / "fig_gamma_sensitivity.pdf")
    print(f"  wrote {_RESULTS / 'gamma_sensitivity.json'}")
    print(f"  wrote {_FIGURES / 'fig_gamma_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
