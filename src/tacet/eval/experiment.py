"""Experiment harness — the controlled study reported in the paper.

Each experiment runs many independent benchmark instances (one per seed) and
aggregates the results. Jobs are dispatched to a process pool; every worker
regenerates its benchmark from a `BenchmarkConfig` (picklable) so the oracle
closure never has to cross a process boundary.

Run::

    python -m tacet.eval.experiment --out experiments/results --seeds 20
"""

from __future__ import annotations

import os

# single-thread BLAS: with a process pool, multi-threaded numpy would
# oversubscribe the cores. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from tacet.eval import baselines
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher
from tacet.serve.config import CascadeConfig, KGEConfig


@dataclass
class Job:
    """A single (system, configuration, seed) run."""

    experiment: str
    system: str
    bench: BenchmarkConfig
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    teacher_error: float = 0.0
    consolidate_every: int = 100
    tag: str = ""


def _run_job(job: Job) -> dict:
    """Worker: regenerate the benchmark and run one system."""
    bench = generate(job.bench)
    teacher = OracleTeacher(
        bench.oracle,
        error_rate=job.teacher_error,
        entity_pool=bench.entity_pool,
        seed=job.bench.seed,
    )
    if job.system == "llm_only":
        res = baselines.run_llm_only(bench, teacher)
    elif job.system == "symbolic_only":
        res = baselines.run_symbolic_only(bench)
    elif job.system == "cache_cascade":
        res = baselines.run_cache_cascade(bench, teacher)
    else:
        res = baselines.run_cascade(
            bench,
            teacher,
            job.cascade,
            consolidate_every=job.consolidate_every,
            system_name=job.system,
        )
    return {
        "experiment": job.experiment,
        "system": job.system,
        "tag": job.tag,
        "seed": job.bench.seed,
        "total_cost": res.total_cost,
        "accuracy": res.accuracy,
        "avg_latency_ms": res.avg_latency_ms,
        "tier_counts": res.tier_counts(),
        "cost_trajectory": res.cost_trajectory(),
        "cost_by_class": res.cost_by_class(),
        "accuracy_by_class": res.accuracy_by_class(),
        "n_queries": res.n,
        "synthesised_rules": res.meta.get("synthesised_rules", []),
    }


def _t_critical(df: int) -> float:
    """Two-sided 95% Student-t critical value for ``df`` degrees of freedom.

    With few seeds the normal approximation (1.96) understates the interval:
    for the 8-seed grid here ``df=7`` gives ``t=2.365``, a 1.21x wider bar. We
    use the exact t value deliberately. A silent fall-back to 1.96 would
    misreport a (narrower) normal interval as a Student-t one, so a missing
    SciPy is raised rather than swallowed -- the harness's ``experiments`` extra
    pins SciPy precisely so this never degrades a reported error bar unnoticed.
    """
    if df < 1:
        return 0.0
    try:
        from scipy.stats import t as _t
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for the Student-t confidence interval; install "
            "the 'experiments' extra. Refusing to silently use the narrower "
            "normal z=1.96 in its place."
        ) from exc
    return float(_t.ppf(0.975, df))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    n = len(values)
    mean = statistics.fmean(values)
    # sample statistics (ddof=1) reported consistently: a single sample has no
    # spread, so std/sem/CI are zero at n=1.
    std = statistics.stdev(values) if n > 1 else 0.0
    sem = (std / (n**0.5)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": _t_critical(n - 1) * sem, "n": n}


# --------------------------------------------------------------------------
def build_jobs(seeds: int, fast: bool) -> list[Job]:
    """The full job list for every experiment in the paper."""
    base_bench = BenchmarkConfig()
    kge = KGEConfig(epochs=45 if fast else 55, dim=64)
    base_cascade = CascadeConfig(kge=kge)
    seed_range = range(seeds)
    jobs: list[Job] = []

    # E1: main comparison (also feeds E2 cost trajectory)
    for s in seed_range:
        bench = replace(base_bench, seed=s)
        for system in ("llm_only", "symbolic_only", "cache_cascade"):
            jobs.append(Job("E1", system, bench))
        jobs.append(
            Job("E1", "static_cascade", bench, cascade=replace(base_cascade, distillation=False))
        )
        jobs.append(Job("E1", "tacet", bench, cascade=base_cascade))

    # E3: Tier-2 threshold sweep (Pareto frontier)
    for s in seed_range:
        bench = replace(base_bench, seed=s)
        for thr in (0.0, 0.6, 0.8, 0.95, 1.01):
            jobs.append(
                Job(
                    "E3",
                    "tacet",
                    bench,
                    cascade=replace(base_cascade, l2_threshold=thr),
                    tag=f"thr={thr}",
                )
            )

    # E4: ablations
    for s in seed_range:
        bench = replace(base_bench, seed=s)
        ablations = {
            "full": base_cascade,
            "no_writeback": replace(base_cascade, write_back=False),
            "no_kge_aug": replace(base_cascade, kge_augment=False),
            "no_rule_synth": replace(base_cascade, rule_synthesis=False),
            "no_distill": replace(base_cascade, distillation=False),
        }
        for tag, cc in ablations.items():
            jobs.append(Job("E4", "tacet", bench, cascade=cc, tag=tag))

    # E5: sensitivity to workload repeat rate
    for s in seed_range:
        for rep in (0.0, 0.15, 0.30, 0.45, 0.60):
            bench = replace(base_bench, seed=s, repeat_rate=rep)
            jobs.append(Job("E5", "tacet", bench, cascade=base_cascade, tag=f"repeat={rep}"))
            jobs.append(Job("E5", "cache_cascade", bench, tag=f"repeat={rep}"))

    # E6: robustness to a noisy teacher
    for s in seed_range:
        bench = replace(base_bench, seed=s)
        for err in (0.0, 0.05, 0.10, 0.20):
            jobs.append(
                Job("E6", "tacet", bench, cascade=base_cascade, teacher_error=err, tag=f"err={err}")
            )
            jobs.append(Job("E6", "llm_only", bench, teacher_error=err, tag=f"err={err}"))
    return jobs


def aggregate(rows: list[dict]) -> dict:
    """Aggregate raw per-run rows into per-experiment summary statistics."""
    out: dict = {}

    def group(exp: str, key) -> dict:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            if r["experiment"] != exp:
                continue
            groups.setdefault(key(r), []).append(r)
        return groups

    # E1
    e1: dict = {}
    for name, rs in group("E1", lambda r: r["system"]).items():
        e1[name] = {
            "cost": _summary([r["total_cost"] for r in rs]),
            "accuracy": _summary([r["accuracy"] for r in rs]),
            "latency_ms": _summary([r["avg_latency_ms"] for r in rs]),
            "tier_fraction": _avg_tier_fraction(rs),
        }
    out["E1"] = e1

    # E2: mean per-query cost trajectory (cumulative)
    e2: dict = {}
    for name, rs in group("E1", lambda r: r["system"]).items():
        trajs = [r["cost_trajectory"] for r in rs if r["cost_trajectory"]]
        if trajs:
            length = min(len(t) for t in trajs)
            e2[name] = [statistics.fmean(t[i] for t in trajs) for i in range(length)]
    out["E2"] = e2

    # E3: Pareto frontier
    e3: dict = {}
    for tag, rs in group("E3", lambda r: r["tag"]).items():
        e3[tag] = {
            "cost": _summary([r["total_cost"] for r in rs]),
            "accuracy": _summary([r["accuracy"] for r in rs]),
        }
    out["E3"] = e3

    # E4: ablations
    e4: dict = {}
    for tag, rs in group("E4", lambda r: r["tag"]).items():
        e4[tag] = {
            "cost": _summary([r["total_cost"] for r in rs]),
            "accuracy": _summary([r["accuracy"] for r in rs]),
        }
    out["E4"] = e4

    # E5: sensitivity
    e5: dict = {}
    for tag, rs in group("E5", lambda r: f"{r['system']}|{r['tag']}").items():
        e5[tag] = {
            "cost": _summary([r["total_cost"] for r in rs]),
            "accuracy": _summary([r["accuracy"] for r in rs]),
        }
    out["E5"] = e5

    # E6: noisy teacher
    e6: dict = {}
    for tag, rs in group("E6", lambda r: f"{r['system']}|{r['tag']}").items():
        e6[tag] = {
            "cost": _summary([r["total_cost"] for r in rs]),
            "accuracy": _summary([r["accuracy"] for r in rs]),
        }
    out["E6"] = e6
    return out


def _avg_tier_fraction(rows: list[dict]) -> dict[str, float]:
    acc = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for r in rows:
        total = sum(r["tier_counts"].values()) or 1
        for tier, count in r["tier_counts"].items():
            acc[int(tier)] += count / total
    n = len(rows) or 1
    return {str(k): v / n for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TACET experiment grid.")
    ap.add_argument("--out", default="experiments/results", help="output directory")
    ap.add_argument("--seeds", type=int, default=20, help="benchmark seeds per cell")
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--fast", action="store_true", help="smaller KGE (quick run)")
    args = ap.parse_args()

    jobs = build_jobs(args.seeds, args.fast)
    print(f"dispatching {len(jobs)} jobs ...")
    workers = args.workers or None
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_run_job, jobs), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} done", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw.json").write_text(json.dumps(rows), encoding="utf-8")
    summary = aggregate(rows)
    summary["_meta"] = {"seeds": args.seeds, "jobs": len(jobs), "fast": args.fast}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'raw.json'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
