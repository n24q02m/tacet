"""Aggregate the controlled multi-seed real-LLM runs into a paper-ready summary.

The camera-ready condition raised in review was to move the real-cost claim from
a single seed to >=3 seeds. This reads the controlled per-seed runs (Tier-2
disabled + a single teacher answer shared across arms, so accuracy is matched by
construction) and reports the amortisation ratio as mean +/- sample standard
deviation across seeds, so the headline rests on a multi-seed estimate.

Run after the per-seed controlled runs exist::

    uv run python experiments/aggregate_multiseed_real_kg.py

Writes ``experiments/results/real_kg_controlled_summary.json`` and prints a table.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

R = Path("experiments/results")

#: hop label -> per-seed controlled-run result files. The controlled design
#: (Tier-2 disabled + a single teacher answer shared across arms) makes the
#: cross-arm accuracy comparison fair, so the amortisation ratio is a genuine
#: cost-at-matched-accuracy figure.
HOPS: dict[str, list[str]] = {
    "1hop": [
        "real_kg_controlled_1hop_seed0.json",
        "real_kg_controlled_1hop_seed1.json",
        "real_kg_controlled_1hop_seed2.json",
    ],
    "2hop": [
        "real_kg_controlled_2hop_seed0.json",
        "real_kg_controlled_2hop_seed1.json",
        "real_kg_controlled_2hop_seed2.json",
    ],
}
ARMS = ["llm_only", "cache_cascade", "full_distillation"]


def _load(fn: str) -> dict:
    return json.loads((R / fn).read_text(encoding="utf-8"))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main() -> None:
    summary: dict[str, dict] = {}
    for hop, files in HOPS.items():
        reports = [_load(f) for f in files]
        seeds = [r["seed"] for r in reports]
        per = {a: {"cost": [], "acc": [], "calls": []} for a in ARMS}
        ratios_full: list[float] = []
        ratios_cache: list[float] = []
        distinct: list[int] = []
        for r in reports:
            distinct.append(r["distinct_queries"])
            by = {a["arm"]: a for a in r["arms"]}
            for a in ARMS:
                per[a]["cost"].append(by[a]["total_cost_usd"])
                per[a]["acc"].append(by[a]["accuracy"])
                per[a]["calls"].append(by[a]["teacher_calls"])
            ratios_full.append(
                by["llm_only"]["total_cost_usd"] / by["full_distillation"]["total_cost_usd"]
            )
            ratios_cache.append(
                by["llm_only"]["total_cost_usd"] / by["cache_cascade"]["total_cost_usd"]
            )
        summary[hop] = {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "distinct_mean": round(statistics.mean(distinct), 1),
            "ratio_full_over_llm": _mean_std(ratios_full),
            "ratio_cache_over_llm": _mean_std(ratios_cache),
            "ratios_full_per_seed": [round(x, 3) for x in ratios_full],
            "per_arm": {
                a: {
                    "cost": _mean_std(per[a]["cost"]),
                    "acc": _mean_std(per[a]["acc"]),
                    "calls": _mean_std(per[a]["calls"]),
                }
                for a in ARMS
            },
        }

    (R / "real_kg_controlled_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for hop, s in summary.items():
        rf_m, rf_s = s["ratio_full_over_llm"]
        rc_m, rc_s = s["ratio_cache_over_llm"]
        print(f"\n=== MetaQA {hop} | {s['n_seeds']} seeds {s['seeds']} ===")
        print(f"  distinct heads (mean): {s['distinct_mean']}")
        print(
            f"  full-distillation amortisation: {rf_m:.2f}x +/- {rf_s:.2f}  "
            f"(per-seed {s['ratios_full_per_seed']})"
        )
        print(f"  cache amortisation:             {rc_m:.2f}x +/- {rc_s:.2f}")
        for a in ARMS:
            c_m, c_s = s["per_arm"][a]["cost"]
            ac_m, ac_s = s["per_arm"][a]["acc"]
            cl_m, cl_s = s["per_arm"][a]["calls"]
            print(
                f"    {a:18s} cost ${c_m:.4f}+/-{c_s:.4f}  acc {ac_m:.3f}+/-{ac_s:.3f}  "
                f"calls {cl_m:.0f}+/-{cl_s:.0f}"
            )
    print(f"\nwrote {R / 'real_kg_controlled_summary.json'}")


if __name__ == "__main__":
    main()
