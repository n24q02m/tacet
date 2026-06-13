"""Proof-tree validity and auditability evaluation over the synthetic KGQA benchmark.

    python experiments/run_audit_eval.py [--seed 0] [--workload-size 300]

What it does
------------
1. Generates the synthetic "organisation & society" benchmark (``benchmark.py``).
2. Runs the Tier-1 symbolic engine (ontology axioms + the rules shipped with the
   system) over every workload query.
3. For each *answered* query, scores the proof with the auditability metrics from
   ``tacet.eval.eval_audit``: ``proof_validity`` (grounding) and ``proof_coverage``.
4. Aggregates mean validity, mean coverage and the answered count, writes a LaTeX
   table to ``experiments/results/tab_audit.tex`` and updates the macros in
   ``experiments/results/macros.tex``.

Honesty note: ``proof_validity`` audits that every proof step reduces to a base
fact or a known rule (provenance integrity / soundness), not that the answer is
correct against the oracle. The symbolic tier abstains rather than guess, so its
*answered* set is the rule-covered fraction of the workload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tacet.core.symbolic import RuleEngine
from tacet.eval import benchmark
from tacet.eval.benchmark import BenchmarkConfig
from tacet.eval.eval_audit import proof_coverage, proof_validity

_RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"
# Paper-consumed artifacts (\input-ed tables + the shared macros file) live under
# paper/results so a documented run regenerates exactly what the paper reads.
_PAPER_RESULTS = Path(__file__).resolve().parents[1] / "paper" / "results"


def _write_table(path: Path, coverage: float, validity: float, answered: int, total: int) -> None:
    rows = [
        ("Proof coverage", f"{coverage:.3f}"),
        ("Proof validity", f"{validity:.3f}"),
        ("Answered queries", f"{answered}/{total}"),
    ]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        *[f"{name} & {value} \\\\" for name, value in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_macros(path: Path, coverage: float, validity: float) -> None:
    macros = {
        r"\proofCoverage": f"{coverage:.3f}",
        r"\proofValidity": f"{validity:.3f}",
    }
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in existing if not any(f"{{{name}}}" in ln for name in macros)]
    for name, value in macros.items():
        kept.append(f"\\renewcommand{{{name}}}{{{value}}}")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workload-size", type=int, default=300)
    args = ap.parse_args()

    cfg = BenchmarkConfig(seed=args.seed, workload_size=args.workload_size)
    bench = benchmark.generate(cfg)

    engine = RuleEngine(bench.ontology, list(bench.given_rules))
    engine.materialise(bench.graph)

    n_total = len(bench.workload)
    answered = 0
    validity_sum = 0.0
    coverage_sum = 0.0
    for head, relation in bench.workload:
        result = engine.query(head, relation)
        coverage_sum += proof_coverage(engine, result)
        if result.answered:
            answered += 1
            validity_sum += proof_validity(engine, result)

    mean_coverage = coverage_sum / n_total if n_total else 0.0
    mean_validity = validity_sum / answered if answered else 0.0

    print(f"seed={args.seed}  workload={n_total}")
    print(f"  answered          : {answered}/{n_total}")
    print(f"  mean proof_coverage (over workload) : {mean_coverage:.3f}")
    print(f"  mean proof_validity (over answered) : {mean_validity:.3f}")

    _PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    _write_table(_PAPER_RESULTS / "tab_audit.tex", mean_coverage, mean_validity, answered, n_total)
    _update_macros(_PAPER_RESULTS / "macros.tex", mean_coverage, mean_validity)
    print(f"  wrote {_PAPER_RESULTS / 'tab_audit.tex'}")
    print(f"  updated {_PAPER_RESULTS / 'macros.tex'}")


if __name__ == "__main__":
    main()
