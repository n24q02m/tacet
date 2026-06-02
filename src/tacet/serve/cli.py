"""Command-line interface — `python -m tacet.serve.cli ...` (or `tacet ...`)."""

from __future__ import annotations

import argparse

from tacet.eval import baselines
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher
from tacet.serve.config import CascadeConfig, KGEConfig


def _demo(args: argparse.Namespace) -> None:
    """Run all five systems on one synthetic benchmark and print the comparison."""
    bench = generate(BenchmarkConfig(seed=args.seed))
    teacher = OracleTeacher(bench.oracle, entity_pool=bench.entity_pool)
    cascade = CascadeConfig(kge=KGEConfig(epochs=args.epochs))

    print(f"\nTACET demo — synthetic KGQA benchmark (seed {args.seed})")
    print(f"  graph: {bench.graph.stats()}")
    print(f"  workload: {len(bench.workload)} queries\n")

    runs = [
        baselines.run_llm_only(bench, teacher),
        baselines.run_symbolic_only(bench),
        baselines.run_cache_cascade(bench, teacher),
        baselines.run_cascade(
            bench,
            teacher,
            CascadeConfig(distillation=False, kge=KGEConfig(epochs=args.epochs)),
            system_name="static_cascade",
        ),
        baselines.run_cascade(bench, teacher, cascade, system_name="tacet"),
    ]
    head = f"  {'system':16s} {'cost ($)':>10s} {'accuracy':>10s} {'avg ms':>9s}  routing T1/T2/T3"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in runs:
        tc = r.tier_counts()
        print(
            f"  {r.system:16s} {r.total_cost:10.3f} {r.accuracy:10.3f} "
            f"{r.avg_latency_ms:9.1f}  {tc[1]:>4d}/{tc[2]:>3d}/{tc[3]:>3d}"
        )

    tacet = runs[-1]
    llm = runs[0]
    saving = (1 - tacet.total_cost / llm.total_cost) * 100
    print(
        f"\n  TACET vs LLM-only: {saving:.1f}% cheaper "
        f"({llm.total_cost / tacet.total_cost:.1f}x), "
        f"accuracy {tacet.accuracy:.1%}"
    )
    print(f"  synthesised rules: {len(tacet.meta.get('synthesised_rules', []))}\n")


def _experiment(args: argparse.Namespace) -> None:
    """Delegate to the full experiment grid."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from tacet.eval.experiment import _run_job, aggregate, build_jobs

    jobs = build_jobs(args.seeds, args.fast)
    print(f"dispatching {len(jobs)} jobs ...")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        for i, row in enumerate(pool.map(_run_job, jobs), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.json").write_text(json.dumps(rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(aggregate(rows), indent=2), encoding="utf-8")
    print(f"wrote results to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tacet",
        description="TACET — an auditable causal-temporal neuro-symbolic "
        "reasoning engine with proof trees and online rule distillation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run all systems on a synthetic benchmark")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--epochs", type=int, default=80)
    d.set_defaults(func=_demo)

    e = sub.add_parser("experiment", help="run the full experiment grid")
    e.add_argument("--out", default="experiments/results")
    e.add_argument("--seeds", type=int, default=16)
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--fast", action="store_true")
    e.set_defaults(func=_experiment)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
