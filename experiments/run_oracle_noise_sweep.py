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

#: Two-sided standard-normal quantiles z_{1-a/2}, tabulated so the Wilson interval
#: needs NO scipy dependency. Only the pre-registered 0.95 level is used by the
#: sweep; the neighbours let a caller pass a different ``confidence``.
_NORMAL_QUANTILE: dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}
#: Confidence level for the E11 secondary estimand P(rule installed | teacher acc).
WILSON_CONFIDENCE = 0.95  # PRE-REGISTERED
#: Rule-miner confidence threshold used by the 1-D sweep (the library default).
DEFAULT_GAMMA = 0.95
#: Default gamma grid for the 2-D sweep (matches run_gamma_sensitivity.py).
DEFAULT_GAMMAS: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


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


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``successes / n``.

    Pure function, no scipy: the normal quantile ``z`` is read from the module
    tabulation :data:`_NORMAL_QUANTILE`, so only the pre-registered 0.95 level (and
    a couple of neighbours) are supported; an unknown ``confidence`` raises.

    Used for the E11 secondary estimand ``P(rule installed | teacher accuracy)``.
    A Wilson interval — not a bootstrap-on-a-mean — is correct here because the
    per-cell outcome is Bernoulli (a rule installs or does not), and Wilson stays
    inside ``[0, 1]`` and well-behaved at the 0 / 1 boundaries and small ``n``.

    ``n == 0`` contract: with no trials there is no evidence, so the interval
    degenerates to the point ``(0.0, 0.0)`` (in range, never raises) — a caller
    aggregating an empty cell still gets a usable bound. Both endpoints are
    clamped to ``[0, 1]``.
    """
    if n <= 0:
        return (0.0, 0.0)
    try:
        z = _NORMAL_QUANTILE[confidence]
    except KeyError as exc:
        raise ValueError(
            f"unsupported confidence {confidence!r}; add its normal quantile to "
            f"_NORMAL_QUANTILE (no scipy). Available: {sorted(_NORMAL_QUANTILE)}"
        ) from exc
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def cliff_teacher_accuracy(
    teacher_accuracies: Sequence[float | None],
    p_installed: Sequence[float | None],
    threshold: float = 0.5,
) -> float | None:
    """Teacher accuracy at which ``p_rule_installed`` crosses ``threshold`` (0.5).

    ``teacher_accuracies`` and ``p_installed`` are PARALLEL arrays giving the
    measured curve, one point per ``error_rate`` and ORDERED BY ``error_rate``
    ascending (so teacher accuracy runs high->low as noise rises). The crossing is
    found by scanning consecutive points for the first pair that brackets
    ``threshold`` and LINEARLY INTERPOLATING the teacher accuracy at
    ``p_installed == threshold``::

        frac  = (threshold - p_i) / (p_{i+1} - p_i)
        cliff = acc_i + frac * (acc_{i+1} - acc_i)

    Returns ``None`` when the curve never crosses ``threshold`` within the swept
    grid (all points on one side) or when fewer than two usable points exist;
    points with a ``None`` coordinate are dropped first. This is the direct,
    pre-registered test of H2: the crossing should fall as ``gamma`` falls.
    """
    pts = [
        (a, p)
        for a, p in zip(teacher_accuracies, p_installed, strict=True)
        if a is not None and p is not None
    ]
    if len(pts) < 2:
        return None
    for (a0, p0), (a1, p1) in zip(pts, pts[1:], strict=False):
        if p0 == threshold:
            return float(a0)
        if (p0 - threshold) * (p1 - threshold) < 0.0:
            frac = (threshold - p0) / (p1 - p0)
            return float(a0 + frac * (a1 - a0))
    if pts[-1][1] == threshold:
        return float(pts[-1][0])
    return None


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


def _run_cell(
    *,
    hop: int,
    split: str,
    limit: int,
    zipf_a: float,
    seed: int,
    error_rate: float,
    gamma: float,
    metaqa_root: str,
    budget_usd: float,
    settings,  # noqa: ANN001
    bench,  # noqa: ANN001
) -> dict:
    """Run ONE controlled cell and flatten its report into the sweep's cell dict.

    Shared by :func:`sweep` (always ``gamma=DEFAULT_GAMMA``) and :func:`sweep_2d`
    (gamma swept), so a single-gamma 2-D sweep reproduces the 1-D cells exactly.
    """
    rep = run_controlled(
        metaqa_root=metaqa_root,
        hop=hop,
        split=split,
        limit=limit,
        zipf_a=zipf_a,
        budget_usd=budget_usd,
        seed=seed,
        oracle_error_rate=error_rate,
        gamma=gamma,
        settings=settings,
        bench=bench,
        verbose=False,
    )
    v = rep["verdict"]
    return {
        "gamma": gamma,
        "error_rate": error_rate,
        "seed": seed,
        "cache_teacher_calls": v.get("cache_teacher_calls"),
        "full_teacher_calls": v.get("full_teacher_calls"),
        "calls_saved_pct": v.get("calls_saved_pct"),
        "accuracy_matched": v.get("accuracy_matched"),
        "cache_accuracy": v.get("accuracy_cache"),
        "full_accuracy": v.get("accuracy_full"),
        "accuracy_delta": v.get("accuracy_delta"),
        "cell_valid": v.get("cell_valid"),
        "rule_installed": v.get("rule_installed"),
        "rule_world_precision": v.get("rule_world_precision"),
        "synthesised_rules": v.get("synthesised_rules", []),
        "teacher_answer_accuracy": rep["teacher_answer_accuracy"],
        "distinct_queries": rep["distinct_queries"],
        "stream_len": rep["stream_len"],
    }


def _aggregate_cells(rows: Sequence[dict], rng: np.random.Generator) -> dict:
    """Aggregate one group of cells (a fixed error_rate, and gamma in the 2-D sweep).

    Wraps :func:`aggregate_error_rate` (mean calls-saved + bootstrap CI +
    pre-registered verdict — untouched) and appends the E11 install-probability
    estimand (Wilson 95% interval) and the one-sided validity summary. ``rows`` is
    the group's per-seed cells from :func:`_run_cell`.
    """
    rows = list(rows)
    saved = [c["calls_saved_pct"] for c in rows]
    teacher_acc = [c["teacher_answer_accuracy"] for c in rows]
    agg = aggregate_error_rate(saved, rng)

    n = len(rows)
    n_rule_installed = sum(1 for c in rows if c["rule_installed"])
    p_installed = n_rule_installed / n if n else 0.0
    lo, hi = wilson_ci(n_rule_installed, n, WILSON_CONFIDENCE)
    deltas = [c["accuracy_delta"] for c in rows if c["accuracy_delta"] is not None]
    n_invalid = sum(1 for c in rows if c["cell_valid"] is False)
    return {
        "n_seeds": n,
        "per_seed_calls_saved_pct": saved,
        "mean_teacher_answer_accuracy": (
            round(float(np.mean(teacher_acc)), 6) if teacher_acc else None
        ),
        **agg,
        "n_rule_installed": n_rule_installed,
        "p_rule_installed": round(p_installed, 6),
        "p_rule_installed_wilson95": [round(lo, 6), round(hi, 6)],
        "mean_accuracy_delta": (round(float(np.mean(deltas)), 6) if deltas else None),
        "n_invalid_cells": n_invalid,
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
            cells.append(
                _run_cell(
                    hop=hop,
                    split=split,
                    limit=limit,
                    zipf_a=zipf_a,
                    seed=sd,
                    error_rate=er,
                    gamma=DEFAULT_GAMMA,
                    metaqa_root=metaqa_root,
                    budget_usd=budget_usd,
                    settings=settings,
                    bench=bench,
                )
            )

    aggregate: list[dict] = []
    for er in error_rates:
        rows = [c for c in cells if c["error_rate"] == er]
        # A fresh, fixed-seed generator per error_rate so each CI is reproducible
        # and independent of the grid iteration order.
        rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
        aggregate.append({"error_rate": er, **_aggregate_cells(rows, rng)})

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


def sweep_2d(
    hop: int,
    seeds: Sequence[int],
    error_rates: Sequence[float],
    gammas: Sequence[float],
    limit: int,
    zipf_a: float,
    metaqa_root: str = "data/MetaQA",
    split: str = "test",
    budget_usd: float = 1.5,
    bench=None,  # noqa: ANN001
    settings=None,  # noqa: ANN001
) -> dict:
    """2-D ``gamma`` x ``error_rate`` sweep: does the imperfect-teacher cliff track gamma? (H2)

    Cells are the cross product ``(gamma, error_rate, seed)``; each runs the
    controlled pipeline with the full-distillation arm's rule-mining confidence set
    to ``gamma``. Aggregation per ``(gamma, error_rate)`` is IDENTICAL to the 1-D
    :func:`sweep`'s per-``error_rate`` aggregation (mean calls-saved + bootstrap
    95% CI + pre-registered verdict, plus the install-probability Wilson interval
    and one-sided validity counts) — with ``gammas=[DEFAULT_GAMMA]`` the cells are
    byte-for-byte the 1-D sweep's cells.

    Per ``gamma`` it additionally reports ``cliff_teacher_accuracy``: the teacher
    accuracy at which ``p_rule_installed`` crosses 0.5, by LINEAR INTERPOLATION
    between the two bracketing ``error_rate`` points on the measured
    ``(mean_teacher_answer_accuracy, p_rule_installed)`` curve (``None`` if it never
    crosses within the grid). H2 predicts this number falls as ``gamma`` falls.

    Importable and ``sys.argv``-free (a Modal wrapper calls it); ``bench`` /
    ``settings`` are injectable and loaded from ``metaqa_root`` / the environment
    when absent, exactly as :func:`sweep`.
    """
    seeds = list(seeds)
    error_rates = list(error_rates)
    gammas = list(gammas)
    if bench is None:
        bench = load_metaqa(metaqa_root, hop=1, split=split)
    if settings is None:
        settings = load_settings()
    # Free oracle study ($0): force the oracle teacher regardless of the environment.
    if getattr(settings, "teacher", None) != "oracle":
        settings.teacher = "oracle"

    cells: list[dict] = []
    for g in gammas:
        for er in error_rates:
            for sd in seeds:
                cells.append(
                    _run_cell(
                        hop=hop,
                        split=split,
                        limit=limit,
                        zipf_a=zipf_a,
                        seed=sd,
                        error_rate=er,
                        gamma=g,
                        metaqa_root=metaqa_root,
                        budget_usd=budget_usd,
                        settings=settings,
                        bench=bench,
                    )
                )

    aggregate: list[dict] = []
    for g in gammas:
        for er in error_rates:
            rows = [c for c in cells if c["gamma"] == g and c["error_rate"] == er]
            rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
            aggregate.append({"gamma": g, "error_rate": er, **_aggregate_cells(rows, rng)})

    cliff: list[dict] = []
    for g in gammas:
        rows = sorted((a for a in aggregate if a["gamma"] == g), key=lambda a: a["error_rate"])
        accs = [a["mean_teacher_answer_accuracy"] for a in rows]
        ps = [a["p_rule_installed"] for a in rows]
        cliff.append({"gamma": g, "cliff_teacher_accuracy": cliff_teacher_accuracy(accs, ps)})

    return {
        "provenance": {
            "hop": hop,
            "seeds": seeds,
            "error_rates": error_rates,
            "gammas": gammas,
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
            "wilson": {
                "confidence": WILSON_CONFIDENCE,
                "z": _NORMAL_QUANTILE[WILSON_CONFIDENCE],
            },
            "verdict_thresholds": {
                "positive_min_calls_saved_pct": POSITIVE_MIN_CALLS_SAVED_PCT,
                "negative_max_calls_saved_pct": NEGATIVE_MAX_CALLS_SAVED_PCT,
                "note": "PRE-REGISTERED; MUST NOT be tuned to the data",
            },
            "pre_registration": {
                "primary": (
                    "POSITIVE := mean calls_saved_pct >= 20.0 AND the bootstrap 95% CI "
                    "excludes 0 (threshold pre-registered, unchanged)."
                ),
                "validity_one_sided": (
                    "A cell is INVALID iff full_accuracy < cache_accuracy - 1e-9 (the rule "
                    "arm made accuracy worse); cells where the rule arm is more accurate are "
                    "VALID and reported with the accuracy delta."
                ),
                "h2": (
                    "The cliff location tracks gamma: the teacher accuracy at which "
                    "P(rule installed) crosses 0.5 falls as gamma falls."
                ),
            },
        },
        "grid": {"gammas": gammas, "error_rates": error_rates, "seeds": seeds},
        "cells": cells,
        "aggregate": aggregate,
        "cliff": cliff,
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
    ap.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "rule-mining confidence gammas; omit for the 1-D sweep (gamma=0.95). Provide "
            "values to run the 2-D noise x gamma sweep (H2). Default grid: "
            f"{list(DEFAULT_GAMMAS)}"
        ),
    )
    ap.add_argument("--out", default="experiments/results/oracle_noise_sweep.json")
    args = ap.parse_args()

    error_rates = (
        [float(x) for x in args.error_rates.split(",")]
        if args.error_rates
        else list(DEFAULT_ERROR_RATES)
    )
    seeds = list(range(args.seeds))
    if args.gammas is None:
        # 1-D sweep — the existing path, unchanged.
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
    else:
        # 2-D noise x gamma sweep (H2). A single gamma of 0.95 reproduces the 1-D cells.
        result = sweep_2d(
            hop=args.hop,
            seeds=seeds,
            error_rates=error_rates,
            gammas=args.gammas,
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
    if args.gammas is None:
        for agg in result["aggregate"]:
            print(
                f"  p={agg['error_rate']:.2f} teacher_acc={agg['mean_teacher_answer_accuracy']} "
                f"mean_saved={agg['mean_calls_saved_pct']}% ci={agg['bootstrap95_ci']} "
                f"-> {agg['verdict']}"
            )
    else:
        for c in result["cliff"]:
            print(
                f"  gamma={c['gamma']} cliff_teacher_accuracy={c['cliff_teacher_accuracy']} "
                "(P(install) crosses 0.5)"
            )


if __name__ == "__main__":
    main()
