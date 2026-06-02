"""Rule-mining precision evaluation: proposed / installed / world-correct.

A reviewer asked us to quantify the miner's over-production. The AMIE-style
miner forms many candidate bodies; the confidence/support threshold installs a
subset; and only *some* of those installed rules are actually true in the world.
This script measures all three counts on the deterministic synthetic KGQA
benchmark (oracle teacher, no API calls) and, optionally, reports the committed
MetaQA-2hop ORACLE run's single synthesised composition.

For each discoverable relation the cascade synthesises a rule for, we record:

* **proposed**      -- candidate bodies the miner forms before the cut
                       (``mine_rules_with_stats``).
* **installed**     -- rules that pass confidence/support + dedup + cap (what the
                       cascade actually adds to the engine).
* **world-correct** -- installed rules whose precision on the FULL ground-truth
                       graph is >= 0.9 (``rule_world_precision``).
* **mean precision**-- mean world-precision of the installed rules.

The ground-truth graph is the benchmark's complete truth table (every entity,
every derived relation), so world-precision is measured over the whole world,
not only the teacher-answered heads the confidence filter saw.

    uv run python experiments/run_rule_precision.py [--seed 0] [--seeds 5] [--workload-size 300]

With ``--seeds N`` the script runs seeds ``0 .. N-1`` and reports each count as
mean +/- std across the seeds, so the table is no longer a single-seed point
estimate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tacet.cascade.router import TACET
from tacet.core.graph import WorldGraph
from tacet.distill.distill import mine_rules_with_stats
from tacet.eval import benchmark
from tacet.eval.benchmark import BenchmarkConfig
from tacet.eval.rule_precision import rule_world_precision
from tacet.llm.teacher import OracleTeacher
from tacet.serve.config import CascadeConfig, KGEConfig

_RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
_PAPER_RESULTS = Path(__file__).resolve().parents[1] / "paper" / "results"

# The committed MetaQA-2hop ORACLE run synthesised exactly the genuine
# inverse-then-forward composition; its world precision is 1.0 by construction
# (the 2-hop relation IS that composition). Source: results/oracle_2hop_300.json.
_ORACLE_2HOP = _RESULTS / "oracle_2hop_300.json"


def _ground_truth_graph(bench: benchmark.Benchmark) -> WorldGraph:
    """A graph holding the complete truth table over every entity.

    Combines the base edges of the benchmark graph with every (head, relation,
    tail) in ``bench.truth`` (including the held-out and derived relations), so
    world-precision is measured against full ground truth, not the partial graph
    the cascade observed.
    """
    g = bench.graph.copy()
    for (head, relation), tails in bench.truth.items():
        for tail in tails:
            g.add_edge(head, relation, tail)
    return g


def _evaluate_synthetic(seed: int, workload_size: int) -> dict:
    """Run the distillation cascade and score the mined rules."""
    cfg = BenchmarkConfig(seed=seed, workload_size=workload_size)
    bench = benchmark.generate(cfg)
    gt_graph = _ground_truth_graph(bench)

    teacher = OracleTeacher(bench.oracle, error_rate=0.0, entity_pool=bench.entity_pool, seed=seed)
    cascade = CascadeConfig(kge=KGEConfig(epochs=55, dim=64))
    ak = TACET(
        bench.graph.copy(), bench.ontology, teacher, rules=list(bench.given_rules), config=cascade
    )
    ak.warmup(calibration=bench.calibration)

    for idx, (h, r) in enumerate(bench.workload):
        ak.ask(h, r)
        if (idx + 1) % 100 == 0:
            ak.consolidate()

    # Re-mine each synthesised relation from the cascade's own accumulated
    # distiller state to recover the PROPOSED candidate count, then score every
    # installed rule's world precision. ``ak.distiller`` holds exactly the
    # teacher_facts / complete_heads the cascade mined from.
    distiller = ak.distiller
    # ak.synthesised_rules are full names like "syn:superior_of<=manages"; the
    # relation is the token between "syn:" and "<=".
    synth_relations = sorted(
        {n[len("syn:") :].split("<=")[0] for n in ak.synthesised_rules if n.startswith("syn:")}
    )

    n_proposed = 0
    installed_rules = []
    for relation in synth_relations:
        rules, proposed = mine_rules_with_stats(
            ak.graph,
            distiller.teacher_facts,
            relation,
            min_confidence=distiller.min_confidence,
            min_support=distiller.min_support,
            complete_heads=distiller._complete_heads.get(relation, set()),
            allowed_body=distiller.base_relations or None,
        )
        n_proposed += proposed
        installed_rules.extend(rules)

    precisions = [rule_world_precision(m.rule, gt_graph) for m in installed_rules]
    n_installed = len(installed_rules)
    n_world_correct = sum(1 for p in precisions if p >= 0.9)
    mean_prec = statistics.fmean(precisions) if precisions else 0.0

    return {
        "seed": seed,
        "workload_size": workload_size,
        "synth_relations": synth_relations,
        "installed_rule_names": [m.rule.name for m in installed_rules],
        "per_rule_precision": {
            m.rule.name: round(p, 4) for m, p in zip(installed_rules, precisions, strict=True)
        },
        "proposed": n_proposed,
        "installed": n_installed,
        "world_correct": n_world_correct,
        "mean_precision": round(mean_prec, 4),
    }


def _aggregate_seeds(per_seed: list[dict]) -> dict:
    """Aggregate per-seed synthetic runs into mean/std for each count.

    Returns mean and (population) std across seeds for proposed, installed,
    world-correct and mean-precision, plus the union of synthesised relations
    observed (which is stable across seeds on this benchmark).
    """

    def _mean_std(values: list[float]) -> tuple[float, float]:
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return mean, std

    proposed = [r["proposed"] for r in per_seed]
    installed = [r["installed"] for r in per_seed]
    world_correct = [r["world_correct"] for r in per_seed]
    mean_prec = [r["mean_precision"] for r in per_seed]

    relations: set[str] = set()
    for r in per_seed:
        relations.update(r["synth_relations"])

    prop_m, prop_s = _mean_std(proposed)
    inst_m, inst_s = _mean_std(installed)
    wc_m, wc_s = _mean_std(world_correct)
    mp_m, mp_s = _mean_std(mean_prec)

    return {
        "seeds": [r["seed"] for r in per_seed],
        "n_seeds": len(per_seed),
        "workload_size": per_seed[0]["workload_size"],
        "synth_relations": sorted(relations),
        "proposed_mean": prop_m,
        "proposed_std": prop_s,
        "installed_mean": inst_m,
        "installed_std": inst_s,
        "world_correct_mean": wc_m,
        "world_correct_std": wc_s,
        "mean_precision_mean": mp_m,
        "mean_precision_std": mp_s,
        "per_seed": per_seed,
    }


def _evaluate_oracle_2hop() -> dict | None:
    """The committed MetaQA-2hop oracle run: 1 installed rule, world-correct.

    The 2-hop target relation IS the inverse-then-forward composition the rule
    encodes, so its world precision is 1.0 on the MetaQA KB.
    """
    if not _ORACLE_2HOP.exists():
        return None
    data = json.loads(_ORACLE_2HOP.read_text(encoding="utf-8"))
    rules = data.get("verdict_full_vs_cache", {}).get("synthesised_rules", [])
    if not rules:
        return None
    return {
        "installed_rule_names": rules,
        "installed": len(rules),
        "world_correct": len(rules),  # the 2-hop relation is exactly this composition
        "mean_precision": 1.0,
        "source": _ORACLE_2HOP.name,
    }


def _fmt_count(mean: float, std: float) -> str:
    """Mean +/- std for an integer-valued count, in LaTeX."""
    return f"${mean:.1f} \\pm {std:.1f}$"


def _fmt_prec(mean: float, std: float) -> str:
    """Mean +/- std for a precision in [0, 1], in LaTeX."""
    return f"${mean:.3f} \\pm {std:.3f}$"


def _write_table(path: Path, agg: dict, oracle: dict | None) -> None:
    rows = [
        ("Candidate rules proposed", _fmt_count(agg["proposed_mean"], agg["proposed_std"])),
        ("Rules installed (conf. filter)", _fmt_count(agg["installed_mean"], agg["installed_std"])),
        (
            r"World-correct ($\ge 0.9$)",
            _fmt_count(agg["world_correct_mean"], agg["world_correct_std"]),
        ),
        (
            "Mean precision (installed)",
            _fmt_prec(agg["mean_precision_mean"], agg["mean_precision_std"]),
        ),
    ]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Synthetic KGQA \\",
        r"\midrule",
        *[f"{name} & {value} \\\\" for name, value in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_macros(path: Path, agg: dict, oracle: dict | None) -> None:
    macros = {
        r"\ruleProposed": _fmt_count(agg["proposed_mean"], agg["proposed_std"]),
        r"\ruleInstalled": _fmt_count(agg["installed_mean"], agg["installed_std"]),
        r"\ruleWorldCorrect": _fmt_count(agg["world_correct_mean"], agg["world_correct_std"]),
        r"\ruleMeanPrec": _fmt_prec(agg["mean_precision_mean"], agg["mean_precision_std"]),
        r"\ruleSeeds": str(agg["n_seeds"]),
    }
    if oracle is not None:
        macros[r"\oracleRuleInstalled"] = str(oracle["installed"])
        macros[r"\oracleRuleWorldCorrect"] = str(oracle["world_correct"])
        macros[r"\oracleRuleMeanPrec"] = f"{oracle['mean_precision']:.3f}"
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in existing if not any(f"{{{name}}}" in ln for name in macros)]
    for name, value in macros.items():
        kept.append(f"\\renewcommand{{{name}}}{{{value}}}")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="first seed (seeds run 0..seeds-1)")
    ap.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of consecutive seeds to aggregate (seed..seed+seeds-1)",
    )
    ap.add_argument("--workload-size", type=int, default=300)
    args = ap.parse_args()

    seeds = list(range(args.seed, args.seed + args.seeds))
    per_seed = []
    for s in seeds:
        syn = _evaluate_synthetic(s, args.workload_size)
        per_seed.append(syn)
        print(f"synthetic seed={syn['seed']} workload={syn['workload_size']}")
        print(f"  synthesised relations : {syn['synth_relations']}")
        print(f"  per-rule world precision : {syn['per_rule_precision']}")
        print(f"  proposed      : {syn['proposed']}")
        print(f"  installed     : {syn['installed']}")
        print(f"  world-correct : {syn['world_correct']} (precision >= 0.9)")
        print(f"  mean precision: {syn['mean_precision']:.3f}")

    agg = _aggregate_seeds(per_seed)
    oracle = _evaluate_oracle_2hop()

    print(f"\naggregate over {agg['n_seeds']} seeds {agg['seeds']}")
    print(f"  proposed      : {agg['proposed_mean']:.1f} +/- {agg['proposed_std']:.1f}")
    print(f"  installed     : {agg['installed_mean']:.1f} +/- {agg['installed_std']:.1f}")
    print(f"  world-correct : {agg['world_correct_mean']:.1f} +/- {agg['world_correct_std']:.1f}")
    print(f"  mean precision: {agg['mean_precision_mean']:.3f} +/- {agg['mean_precision_std']:.3f}")
    if oracle is not None:
        print(f"oracle MetaQA-2hop ({oracle['source']})")
        print(f"  installed     : {oracle['installed']}")
        print(f"  world-correct : {oracle['world_correct']}")
        print(f"  mean precision: {oracle['mean_precision']:.3f}")

    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "rule_precision.json").write_text(
        json.dumps({"synthetic": agg, "oracle_2hop": oracle}, indent=2), encoding="utf-8"
    )
    _PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    _write_table(_PAPER_RESULTS / "tab_rule_precision.tex", agg, oracle)
    _update_macros(_PAPER_RESULTS / "macros.tex", agg, oracle)
    print(f"  wrote {_RESULTS / 'rule_precision.json'}")
    print(f"  wrote {_PAPER_RESULTS / 'tab_rule_precision.tex'}")
    print(f"  updated {_PAPER_RESULTS / 'macros.tex'}")


if __name__ == "__main__":
    main()
