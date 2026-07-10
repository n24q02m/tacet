"""E11 oracle-noise sweep — the teacher-accuracy -> rule-advantage curve ($0).

Research entry E11 asks how the rule arm's saving over the cache arm holds up as
the teacher gets less accurate. The free part of that study sweeps
``OracleTeacher.error_rate`` under the SAME controlled design that measures the
real Grok teacher (Tier-2 disabled + one shared teacher answer per distinct
``(head, relation)`` replayed to every arm), so both live under one design.

For each ``(error_rate, seed)`` cell it runs the controlled pipeline
(:func:`run_real_kg_controlled.run_controlled`) and records the per-arm teacher
call counts, the rule-vs-cache ``calls_saved_pct`` at matched accuracy, and the
teacher's OWN answer accuracy (the curve's x-axis, measured from the shared
cache; see :func:`run_real_kg_controlled._teacher_answer_accuracy`). Across seeds
it reports the mean saving plus a bootstrap 95% CI and a PRE-REGISTERED verdict.

The heavy work is :func:`sweep`, importable so a Modal job can call it directly:
it takes no ``sys.argv``, needs no ``__main__``, and accepts the MetaQA root as a
parameter. ``main()`` is a thin argparse wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from run_real_kg_controlled import run_controlled  # noqa: E402

from tacet.data.metaqa import load_metaqa  # noqa: E402
from tacet.serve.settings import load_settings  # noqa: E402

#: Default corruption grid for the sweep's x-axis (0% .. 50% in 5-point steps).
DEFAULT_ERROR_RATES: tuple[float, ...] = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
)
#: Default number of seeds per error_rate (used by ``main`` to build ``range(N)``).
DEFAULT_N_SEEDS = 3

# ---------------------------------------------------------------------------
# PRE-REGISTERED verdict thresholds. These were fixed BEFORE seeing any result
# and MUST NOT be tuned to the data — moving them to fit a curve would turn the
# pre-registered test into a post-hoc one. Change them only with a documented,
# pre-registered revision of the study, never to make a result read better.
# ---------------------------------------------------------------------------
#: A POSITIVE cell must save at least this many percent of teacher calls (mean).
POSITIVE_MIN_CALLS_SAVED_PCT = 20.0  # PRE-REGISTERED
#: A NEGATIVE cell must have a mean saving strictly below this (i.e. < 0).
NEGATIVE_MAX_CALLS_SAVED_PCT = 0.0  # PRE-REGISTERED
#: Bootstrap replicate count and a FIXED rng seed so every reported CI is
#: reproducible (resampling the per-seed savings).
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_RNG_SEED = 12345
BOOTSTRAP_CI = 0.95


def classify_verdict(mean_calls_saved_pct: float, ci_low: float, ci_high: float) -> str:
    """The PRE-REGISTERED verdict for one error_rate, using EXACTLY the rule:

        POSITIVE := mean_calls_saved_pct >= 20.0  AND  CI excludes 0
        NEUTRAL  := CI contains 0
        NEGATIVE := mean_calls_saved_pct < 0      AND  CI excludes 0

    ``NEUTRAL`` (CI contains 0) is tested first, so a large mean with a CI that
    still straddles 0 is NEUTRAL, not POSITIVE. A cell whose CI excludes 0 with a
    mean in ``[0, 20)`` matches none of the three named classes — a real but
    sub-threshold positive effect — and is reported as ``INCONCLUSIVE`` (a residual
    label, not a fourth tuned threshold).
    """
    ci_excludes_zero = not (ci_low <= 0.0 <= ci_high)
    if not ci_excludes_zero:
        return "NEUTRAL"
    if mean_calls_saved_pct >= POSITIVE_MIN_CALLS_SAVED_PCT:
        return "POSITIVE"
    if mean_calls_saved_pct < NEGATIVE_MAX_CALLS_SAVED_PCT:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over ``values`` (resampling with
    replacement). Deterministic given ``values`` and ``rng``'s seed, so a caller
    that passes a freshly seeded generator gets a reproducible interval.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (1.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 + ci) / 2.0))
    return (lo, hi)


def aggregate_error_rate(calls_saved_pct: Sequence[float], rng: np.random.Generator) -> dict:
    """Mean + bootstrap CI + PRE-REGISTERED verdict for one error_rate's per-seed
    savings. Kept separate from :func:`sweep` so tests can feed synthetic per-seed
    arrays straight in.
    """
    vals = [v for v in calls_saved_pct if v is not None]
    if not vals:
        return {
            "mean_calls_saved_pct": None,
            "bootstrap95_ci": [None, None],
            "ci_excludes_zero": False,
            "verdict": "NEUTRAL",
        }
    mean = float(np.mean(vals))
    lo, hi = bootstrap_ci(vals, rng)
    return {
        "mean_calls_saved_pct": round(mean, 4),
        "bootstrap95_ci": [round(lo, 4), round(hi, 4)],
        "ci_excludes_zero": not (lo <= 0.0 <= hi),
        "verdict": classify_verdict(mean, lo, hi),
    }


def sweep(
    hop: int,
    seeds: Sequence[int],
    error_rates: Sequence[float],
    limit: int,
    zipf_a: float,
    metaqa_root: str = "data/MetaQA",
    split: str = "test",
    budget_usd: float = 1.5,
    bench=None,  # noqa: ANN001
    settings=None,  # noqa: ANN001
) -> dict:
    """Draw the ``teacher-accuracy -> rule-advantage`` curve.

    Runs the controlled pipeline for every ``(error_rate, seed)`` cell (loading the
    MetaQA bench ONCE and reusing it), records each cell, aggregates across seeds
    per error_rate with a bootstrap 95% CI, and emits the PRE-REGISTERED verdict.
    This sweep is free by construction: it forces the ground-truth oracle teacher,
    so measured USD is 0 and the decisive metric is the per-arm teacher call count.

    ``bench`` / ``settings`` are injectable (the Modal wrapper and the tests pass
    their own); when absent they are loaded from ``metaqa_root`` / the environment.
    """
    seeds = list(seeds)
    error_rates = list(error_rates)
    if bench is None:
        bench = load_metaqa(metaqa_root, hop=1, split=split)
    if settings is None:
        settings = load_settings()
    # This sweep is the FREE oracle study ($0, no API calls): force the oracle
    # teacher regardless of any provider key the environment happens to expose.
    if getattr(settings, "teacher", None) != "oracle":
        settings.teacher = "oracle"

    cells: list[dict] = []
    for er in error_rates:
        for sd in seeds:
            rep = run_controlled(
                metaqa_root=metaqa_root,
                hop=hop,
                split=split,
                limit=limit,
                zipf_a=zipf_a,
                budget_usd=budget_usd,
                seed=sd,
                oracle_error_rate=er,
                settings=settings,
                bench=bench,
                verbose=False,
            )
            v = rep["verdict"]
            cells.append(
                {
                    "error_rate": er,
                    "seed": sd,
                    "cache_teacher_calls": v.get("cache_teacher_calls"),
                    "full_teacher_calls": v.get("full_teacher_calls"),
                    "calls_saved_pct": v.get("calls_saved_pct"),
                    "accuracy_matched": v.get("accuracy_matched"),
                    "cache_accuracy": v.get("accuracy_cache"),
                    "full_accuracy": v.get("accuracy_full"),
                    "synthesised_rules": v.get("synthesised_rules", []),
                    "teacher_answer_accuracy": rep["teacher_answer_accuracy"],
                    "distinct_queries": rep["distinct_queries"],
                    "stream_len": rep["stream_len"],
                }
            )

    aggregate: list[dict] = []
    for er in error_rates:
        rows = [c for c in cells if c["error_rate"] == er]
        saved = [c["calls_saved_pct"] for c in rows]
        teacher_acc = [c["teacher_answer_accuracy"] for c in rows]
        # A fresh, fixed-seed generator per error_rate so each CI is reproducible
        # and independent of the grid iteration order.
        rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
        agg = aggregate_error_rate(saved, rng)
        aggregate.append(
            {
                "error_rate": er,
                "n_seeds": len(rows),
                "per_seed_calls_saved_pct": saved,
                "mean_teacher_answer_accuracy": (
                    round(float(np.mean(teacher_acc)), 6) if teacher_acc else None
                ),
                **agg,
            }
        )

    return {
        "provenance": {
            "hop": hop,
            "seeds": seeds,
            "error_rates": error_rates,
            "workload_cap": limit,
            "zipf_a": zipf_a,
            "split": split,
            "metaqa_root": metaqa_root,
            "budget_usd": budget_usd,
            "design": "controlled: Tier-2 disabled + shared teacher answers across arms",
            "teacher_kind": "oracle",
            "tier2_disabled": True,
            "shared_teacher_answers": True,
            "noise_mode": "per_key",
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "rng_seed": BOOTSTRAP_RNG_SEED,
                "ci": BOOTSTRAP_CI,
            },
            "verdict_thresholds": {
                "positive_min_calls_saved_pct": POSITIVE_MIN_CALLS_SAVED_PCT,
                "negative_max_calls_saved_pct": NEGATIVE_MAX_CALLS_SAVED_PCT,
                "note": "PRE-REGISTERED; MUST NOT be tuned to the data",
            },
        },
        "grid": {"error_rates": error_rates, "seeds": seeds},
        "cells": cells,
        "aggregate": aggregate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metaqa-root", default="data/MetaQA")
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--zipf-a", type=float, default=1.5)
    ap.add_argument("--budget-usd", type=float, default=1.5)
    ap.add_argument(
        "--seeds", type=int, default=DEFAULT_N_SEEDS, help="number of seeds (runs 0..N-1)"
    )
    ap.add_argument(
        "--error-rates",
        default=None,
        help="comma-separated override for the error_rate grid (default: 0.0..0.5 by 0.05)",
    )
    ap.add_argument("--out", default="experiments/results/oracle_noise_sweep.json")
    args = ap.parse_args()

    error_rates = (
        [float(x) for x in args.error_rates.split(",")]
        if args.error_rates
        else list(DEFAULT_ERROR_RATES)
    )
    seeds = list(range(args.seeds))
    result = sweep(
        hop=args.hop,
        seeds=seeds,
        error_rates=error_rates,
        limit=args.limit,
        zipf_a=args.zipf_a,
        metaqa_root=args.metaqa_root,
        split=args.split,
        budget_usd=args.budget_usd,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for agg in result["aggregate"]:
        print(
            f"  p={agg['error_rate']:.2f} teacher_acc={agg['mean_teacher_answer_accuracy']} "
            f"mean_saved={agg['mean_calls_saved_pct']}% ci={agg['bootstrap95_ci']} "
            f"-> {agg['verdict']}"
        )


if __name__ == "__main__":
    main()
