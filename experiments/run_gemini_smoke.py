"""End-to-end TACET cascade with a *real* Gemini Tier-3 teacher.

This is the **real-LLM validation** of the cost story: instead of the
``OracleTeacher`` (perfect, free), the cascade routes its Tier-3 queries
to Gemini and we measure the actual cost / accuracy / latency.

Usage::

    export GEMINI_API_KEY=...   # or TACET_GEMINI_API_KEY
    python experiments/run_gemini_smoke.py --limit 30 --seed 0

The benchmark is the same controlled synthetic-org graph used by the
main experiment grid, so the Tier-1 / Tier-2 behaviour is identical and
only the Tier-3 teacher differs. Output: ``experiments/results/gemini_smoke.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from tacet.eval import baselines
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher
from tacet.llm.teachers import GeminiRestTeacher
from tacet.serve.config import CascadeConfig, KGEConfig
from tacet.serve.settings import load_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=30, help="number of workload queries to process")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", default="experiments/results/gemini_smoke.json")
    args = ap.parse_args()

    settings = load_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or TACET_GEMINI_API_KEY).")

    print(f"generating benchmark (seed {args.seed}) ...")
    bench = generate(BenchmarkConfig(seed=args.seed))
    # subset the workload so a smoke run is bounded by `--limit` queries.
    bench.workload = bench.workload[: args.limit]
    bench.classes = bench.classes[: args.limit]
    print(f"  graph: {bench.graph.stats()}, workload: {len(bench.workload)}")

    cfg = CascadeConfig(kge=KGEConfig(epochs=60))

    # --- 1. Oracle baseline (cost ceiling for tier 3 = 0.05 per call) ----
    oracle = OracleTeacher(bench.oracle, entity_pool=bench.entity_pool)
    print("\nrun A — OracleTeacher (frictionless, free):")
    t0 = time.time()
    r_oracle = baselines.run_cascade(bench, oracle, cfg, system_name="oracle")
    print(
        f"  cost=${r_oracle.total_cost:.4f}  acc={r_oracle.accuracy:.3f}  "
        f"tiers={r_oracle.tier_counts()}  wallclock={time.time() - t0:.1f}s"
    )

    # --- 2. Gemini Tier-3 (real frontier LLM, real $ per call) ----------
    gemini = GeminiRestTeacher(api_key, model=args.model, cost=0.005)
    print(f"\nrun B — GeminiRestTeacher ({args.model}, real API):")
    t0 = time.time()
    r_gemini = baselines.run_cascade(bench, gemini, cfg, system_name="gemini")
    print(
        f"  cost=${r_gemini.total_cost:.4f}  acc={r_gemini.accuracy:.3f}  "
        f"tiers={r_gemini.tier_counts()}  wallclock={time.time() - t0:.1f}s"
    )

    # --- 3. LLM-only baseline w/ Gemini (cost ceiling) ------------------
    print("\nrun C — LLM-only (Gemini for every query):")
    t0 = time.time()
    r_llm = baselines.run_llm_only(bench, gemini)
    print(
        f"  cost=${r_llm.total_cost:.4f}  acc={r_llm.accuracy:.3f}  "
        f"wallclock={time.time() - t0:.1f}s"
    )

    report = {
        "model": args.model,
        "seed": args.seed,
        "workload_size": args.limit,
        "graph_stats": bench.graph.stats(),
        "runs": {
            "oracle_cascade": {
                "cost": r_oracle.total_cost,
                "accuracy": r_oracle.accuracy,
                "tier_counts": r_oracle.tier_counts(),
                "accuracy_by_class": r_oracle.accuracy_by_class(),
            },
            "gemini_cascade": {
                "cost": r_gemini.total_cost,
                "accuracy": r_gemini.accuracy,
                "tier_counts": r_gemini.tier_counts(),
                "accuracy_by_class": r_gemini.accuracy_by_class(),
            },
            "gemini_llm_only": {
                "cost": r_llm.total_cost,
                "accuracy": r_llm.accuracy,
            },
        },
        "cost_reduction_x": (
            r_llm.total_cost / r_gemini.total_cost if r_gemini.total_cost > 0 else None
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"cost reduction x (cascade vs LLM-only): {report['cost_reduction_x']:.2f}")


if __name__ == "__main__":
    main()
