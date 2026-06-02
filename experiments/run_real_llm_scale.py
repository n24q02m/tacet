"""Scale-up real-LLM validation for the TACET cascade — 500+ queries × multi-provider.

Compared to ``run_gemini_smoke.py`` (12-30 queries, a single Gemini
model), this script:

* Scales the workload to at least 500 queries (paper §13 needs a large
  sample for the cost / accuracy figures to be statistically meaningful).
* Runs two cascades in parallel: ``RotatingTeacher`` (Gemini + Gemma
  free tier, 9 models per default) and ``GrokTeacher`` if an xAI key is
  available — so the paper can claim TACET is not tied to a single
  provider.
* Records latency p50/p95/p99 + cost / correctness per-query, not just
  the aggregate.  Paper reviewers see the actual distribution, not just
  the total figures.
* Breaks the results out into a confusion matrix by tier (1 / 2 / 3) —
  showing that Tier 3 (LLM) genuinely only serves the novel tail.

Usage::

    export GEMINI_API_KEY=...
    export XAI_API_KEY=...     # optional
    python experiments/run_real_llm_scale.py --limit 500 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, median

from tacet.eval import baselines
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher
from tacet.llm.teachers.llm import (
    DEFAULT_ROTATING_MODELS,
    GeminiRestTeacher,
    GrokTeacher,
    RotatingTeacher,
)
from tacet.serve.config import CascadeConfig, KGEConfig
from tacet.serve.settings import load_settings


def _percentiles(xs: list[float]) -> dict[str, float]:
    """p50 / p95 / p99 for a list of floats (no numpy dependency)."""
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    s = sorted(xs)
    n = len(s)
    return {
        "p50": float(median(s)),
        "p95": s[min(n - 1, int(0.95 * n))],
        "p99": s[min(n - 1, int(0.99 * n))],
        "mean": float(mean(s)),
    }


def _per_query_metrics(run) -> dict:
    """Extract cost / latency / correctness per-query to compute percentiles.

    ``baselines.QueryRecord`` is a dataclass with ``cost``, ``correct``,
    ``latency_ms`` — not a dict.  Converted to seconds for readability in
    paper §13 (LLM round-trips range from hundreds of ms to a few seconds).
    """
    costs = [float(r.cost) for r in run.records]
    lats_s = [float(r.latency_ms) / 1000.0 for r in run.records]
    correct = [bool(r.correct) for r in run.records]
    return {
        "n": len(run.records),
        "cost": _percentiles(costs),
        "latency_s": _percentiles(lats_s) if lats_s else None,
        "accuracy_overall": (sum(correct) / len(correct)) if correct else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit",
        type=int,
        default=500,
        help="number of queries in the workload (paper §13: ≥500 for statistical significance)",
    )
    ap.add_argument(
        "--rotating-cost", type=float, default=0.005, help="cost / Gemini Flash call (USD)"
    )
    ap.add_argument(
        "--grok-cost", type=float, default=0.020, help="cost / Grok call (USD); set 0 if no key"
    )
    ap.add_argument("--out", default="experiments/results/real_llm_scale.json")
    args = ap.parse_args()

    settings = load_settings()
    xai_key = settings.xai_api_key or os.environ.get("XAI_API_KEY")
    gem_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not xai_key and not gem_key:
        raise SystemExit("Set XAI_API_KEY (Grok) or GEMINI_API_KEY.")

    print(f"generating benchmark (seed {args.seed}, limit {args.limit}) ...")
    bench = generate(BenchmarkConfig(seed=args.seed))
    bench.workload = bench.workload[: args.limit]
    bench.classes = bench.classes[: args.limit]
    print(f"  graph: {bench.graph.stats()}, workload: {len(bench.workload)}")

    cfg = CascadeConfig(kge=KGEConfig(epochs=60))

    # --- 1. Oracle baseline (cost floor for comparison) ----------------
    oracle = OracleTeacher(bench.oracle, entity_pool=bench.entity_pool)
    print("\nrun A — OracleTeacher (cost floor, free):")
    t0 = time.time()
    r_oracle = baselines.run_cascade(bench, oracle, cfg, system_name="oracle")
    print(
        f"  cost=${r_oracle.total_cost:.4f}  acc={r_oracle.accuracy:.3f}  "
        f"tiers={r_oracle.tier_counts()}  wallclock={time.time() - t0:.1f}s"
    )

    # --- 2. Grok cascade (PRIMARY teacher per 2026-05-28 switch) -------
    r_grok = None
    grok_wall = 0.0
    if xai_key:
        print(f"\nrun B — GrokTeacher cascade ({settings.xai_model}):")
        grok = GrokTeacher(
            xai_key, model=settings.xai_model, base_url=settings.xai_base_url, cost=args.grok_cost
        )
        t0 = time.time()
        r_grok = baselines.run_cascade(bench, grok, cfg, system_name="grok")
        grok_wall = time.time() - t0
        print(
            f"  cost=${r_grok.total_cost:.4f}  acc={r_grok.accuracy:.3f}  "
            f"tiers={r_grok.tier_counts()}  wallclock={grok_wall:.1f}s"
        )
    else:
        print("\nrun B — skipped (no xAI key in env)")

    # --- 3. RotatingTeacher (Gemini + Gemma free-tier — optional) -----
    r_rot = None
    rot_wall = 0.0
    rotating_models: list[str] = []
    rotating = None
    if gem_key:
        print("\nrun C — RotatingTeacher (Gemini + Gemma, optional cross-check):")
        rotating_models = list(DEFAULT_ROTATING_MODELS)
        rotating = RotatingTeacher(
            [GeminiRestTeacher(gem_key, model=m, cost=args.rotating_cost) for m in rotating_models],
            cooldown_s=settings.rotating_cooldown_s,
        )
        t0 = time.time()
        r_rot = baselines.run_cascade(bench, rotating, cfg, system_name="rotating")
        rot_wall = time.time() - t0
        print(
            f"  cost=${r_rot.total_cost:.4f}  acc={r_rot.accuracy:.3f}  "
            f"tiers={r_rot.tier_counts()}  wallclock={rot_wall:.1f}s"
        )
    else:
        print("\nrun C — skipped (no Gemini key in env)")

    # --- 4. LLM-only ceiling — uses the primary teacher (Grok if set, ---
    #       else rotating Gemini).  This is the apples-to-apples cost
    #       ceiling for the cascade in run B / C.
    llm_only_teacher = grok if xai_key else rotating
    if llm_only_teacher is None:
        raise SystemExit("no real teacher available for LLM-only baseline")
    print("\nrun D — LLM-only (cost ceiling):")
    t0 = time.time()
    r_llm = baselines.run_llm_only(bench, llm_only_teacher)
    llm_wall = time.time() - t0
    print(f"  cost=${r_llm.total_cost:.4f}  acc={r_llm.accuracy:.3f}  wallclock={llm_wall:.1f}s")

    # --- Build the report -----------------------------------------------
    primary_cost = (
        r_grok.total_cost
        if r_grok is not None
        else (r_rot.total_cost if r_rot is not None else None)
    )
    report = {
        "seed": args.seed,
        "workload_size": args.limit,
        "graph_stats": bench.graph.stats(),
        "primary_teacher": "grok" if xai_key else "rotating",
        "primary_model": settings.xai_model if xai_key else None,
        "rotating_models": rotating_models,
        "runs": {
            "oracle_cascade": {
                "cost": r_oracle.total_cost,
                "accuracy": r_oracle.accuracy,
                "tier_counts": r_oracle.tier_counts(),
                "accuracy_by_class": r_oracle.accuracy_by_class(),
            },
            "llm_only": {
                "cost": r_llm.total_cost,
                "accuracy": r_llm.accuracy,
                "wallclock_s": round(llm_wall, 1),
            },
        },
        "cost_reduction_primary_x": (
            r_llm.total_cost / primary_cost if primary_cost and primary_cost > 0 else None
        ),
    }
    if r_grok is not None:
        report["runs"]["grok_cascade"] = {
            "cost": r_grok.total_cost,
            "accuracy": r_grok.accuracy,
            "tier_counts": r_grok.tier_counts(),
            "accuracy_by_class": r_grok.accuracy_by_class(),
            "wallclock_s": round(grok_wall, 1),
            "per_query": _per_query_metrics(r_grok),
        }
    if r_rot is not None:
        report["runs"]["rotating_cascade"] = {
            "cost": r_rot.total_cost,
            "accuracy": r_rot.accuracy,
            "tier_counts": r_rot.tier_counts(),
            "accuracy_by_class": r_rot.accuracy_by_class(),
            "wallclock_s": round(rot_wall, 1),
            "per_query": _per_query_metrics(r_rot),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if report["cost_reduction_primary_x"]:
        print(
            f"cost reduction x ({report['primary_teacher']} cascade vs "
            f"LLM-only): {report['cost_reduction_primary_x']:.2f}"
        )


if __name__ == "__main__":
    main()
