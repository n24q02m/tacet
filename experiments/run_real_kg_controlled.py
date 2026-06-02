"""Controlled real-LLM cost-at-matched-accuracy experiment (camera-ready).

The multi-seed runs of ``run_real_kg_amortization.py`` exposed two confounds in
its *cross-arm accuracy* comparison (the cost comparison was always valid):

1. **Tier-2 KGE ran uncalibrated.** ``TACET.warmup`` was called without a
   calibration split, so the link predictor's 0.60-gated confidence was
   unreliable on MetaQA's flat relations and fired up to 40% of queries on some
   seeds, returning wrong answers. Here Tier-2 is **disabled**
   (``l2_threshold`` above 1): the real-KG study isolates the caching /
   rule-distillation mechanism (Tier 1 + Tier 3), and a streamed workload offers
   no held-out split to calibrate Tier-2.
2. **Each arm sampled the stochastic teacher independently.** Three separate
   teachers meant the same (head, relation) could get three different Grok
   answers, so cross-arm accuracy differences were dominated by the LLM's
   run-to-run variance, not by routing. Here a **single deterministic answer is
   drawn per distinct (head, relation) and shared across all arms**, so the arms
   differ only in *routing*. Each arm's cost is its teacher invocations priced at
   the measured per-pair cost; real API spend is one call per distinct pair
   (so this controlled run is also far cheaper than the original).

Run::

    export TACET_TEACHER=grok TACET_XAI_API_KEY=<key> TACET_KGE_BACKEND=numpy
    uv run python experiments/run_real_kg_controlled.py --hop 1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from run_real_kg_amortization import (  # noqa: E402
    COMPOSITIONS,
    BudgetExceededError,
    BudgetGuard,
    _accuracy,
    _build_composed_workload,
    _build_workload,
    _kg_without,
    _new_metered,
    _zipf_stream,
)

from tacet.cascade.router import TACET  # noqa: E402
from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.core.ontology import Ontology, RelationType  # noqa: E402
from tacet.data.metaqa import load_metaqa  # noqa: E402
from tacet.llm.teacher import Teacher, TeacherResponse  # noqa: E402
from tacet.serve.config import CascadeConfig, KGEConfig  # noqa: E402
from tacet.serve.settings import load_settings  # noqa: E402

#: Confidence is a probability in [0, 1]; a threshold above 1 disables Tier-2.
TIER2_OFF = 1.01


def _line(r: dict) -> str:
    """Compact one-line arm summary for the console."""
    return (
        f"  {r['arm']}: acc={r['accuracy']} cost=${r['total_cost_usd']:.4f} "
        f"calls={r['teacher_calls']} rules={r.get('synthesised_rules', [])}"
    )


class SharedAnswerCache:
    """One real teacher call per distinct (head, relation), shared across arms.

    The first arm to ask a pair pays a real Grok call (metered, charged to the
    shared budget guard) and the answer + its measured cost are cached; every
    later request for that pair --- in the same or another arm --- returns the
    identical answer and the same per-pair cost. This removes the LLM-stochastic
    confound: all arms see the same teacher answer for a pair, so they differ
    only in routing.
    """

    def __init__(self, metered, guard: BudgetGuard, kg: WorldGraph) -> None:  # noqa: ANN001
        self.metered = metered
        self.guard = guard
        self.kg = kg
        self.answers: dict[tuple[str, str], list[str]] = {}
        self.cost: dict[tuple[str, str], float] = {}
        self.real_calls = 0

    def answer(self, head: str, relation: str) -> tuple[list[str], float]:
        key = (head, relation)
        if key not in self.answers:
            resp = self.metered.answer(self.kg, head, relation)
            self.answers[key] = resp.answers
            self.cost[key] = self.metered.last_cost_usd
            self.real_calls += 1
            self.guard.add(self.metered.last_cost_usd)  # real spend only
        return self.answers[key], self.cost[key]


class ReplayTeacher(Teacher):
    """A per-arm teacher that serves shared answers and tallies attributed cost.

    Makes no API calls of its own beyond populating the shared cache; an arm's
    ``total_cost`` is the sum of per-pair costs over the queries it routed to the
    teacher, and ``n_calls`` is that count.
    """

    def __init__(self, shared: SharedAnswerCache) -> None:
        self.shared = shared
        self.total_cost = 0.0
        self.n_calls = 0

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:  # noqa: ARG002
        ans, cost = self.shared.answer(head, relation)
        self.total_cost += cost
        self.n_calls += 1
        return TeacherResponse(answers=ans)


def _replay_llm_only(stream, shared: SharedAnswerCache) -> dict:  # noqa: ANN001
    correct = 0
    cost = 0.0
    t0 = time.time()
    for h, r, gold in stream:
        ans, c = shared.answer(h, r)
        correct += int(_accuracy(gold, ans))
        cost += c
    n = len(stream)
    return {
        "arm": "llm_only",
        "n": n,
        "total_cost_usd": round(cost, 6),
        "teacher_calls": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "tier_pct": {"3": 100.0},
        "wallclock_s": round(time.time() - t0, 1),
    }


def _replay_cascade(name, stream, bench, ontology, shared, cfg) -> dict:  # noqa: ANN001
    replay = ReplayTeacher(shared)
    ak = TACET(_kg_without(bench, stream), ontology, replay, config=cfg)
    ak.warmup()
    correct = 0
    tiers = {1: 0, 2: 0, 3: 0}
    t0 = time.time()
    for h, r, gold in stream:
        ans = ak.ask(h, r)
        correct += int(_accuracy(gold, ans.answers))
        tiers[ans.tier] = tiers.get(ans.tier, 0) + 1
    n = len(stream)
    return {
        "arm": name,
        "n": n,
        "total_cost_usd": round(replay.total_cost, 6),
        "teacher_calls": replay.n_calls,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "tier_counts": tiers,
        "tier_pct": {str(t): round(100.0 * tiers[t] / n, 1) if n else 0.0 for t in (1, 2, 3)},
        "synthesised_rules": list(ak.synthesised_rules),
        "wallclock_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metaqa-root", default="data/MetaQA")
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--zipf-a", type=float, default=1.5)
    ap.add_argument("--budget-usd", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--composition", default=None, choices=sorted(COMPOSITIONS))
    ap.add_argument("--out", default="experiments/results/real_kg_controlled.json")
    args = ap.parse_args()

    model = os.environ.get("TACET_PRICE_MODEL", "grok-4.3")
    rng = np.random.default_rng(args.seed)
    print(f"[cap] {args.limit} queries; hard budget ${args.budget_usd:.2f}; Tier-2 DISABLED")
    bench = load_metaqa(args.metaqa_root, hop=1, split=args.split)
    print(f"  kg stats: {bench.stats()}")

    settings = load_settings()
    if settings.teacher == "oracle" or not settings.xai_api_key:
        raise SystemExit(
            "controlled run needs a real teacher: TACET_TEACHER=grok + TACET_XAI_API_KEY"
        )
    print(f"  teacher=grok model={settings.xai_model} (priced as {model})")

    nl_template = None
    composed_relation = None
    if args.hop == 1:
        pool = _build_workload(bench, limit_pool=max(args.limit, 400), rng=rng)
        ontology = Ontology.induce(bench.kg)
    else:
        comp_name = args.composition or next(
            (k for k, v in COMPOSITIONS.items() if v.get("hop") == args.hop), None
        )
        if comp_name is None:
            raise SystemExit(f"no composition for hop={args.hop}")
        spec = COMPOSITIONS[comp_name]
        nl_template = spec["nl"]
        composed_relation = spec["kg_relation"]
        legs_desc = " . ".join(f"{'~' if inv else ''}{r}" for r, inv in spec["legs"])
        print(f"  COMPOSITION {comp_name!r} (hop={args.hop}): {composed_relation} := {legs_desc}")
        pool = _build_composed_workload(bench.kg, spec, limit_pool=max(args.limit, 400), rng=rng)
        if not pool:
            raise SystemExit(f"composition {comp_name!r} produced an empty pool")
        ontology = Ontology.induce(bench.kg)
        ontology.add_relation_type(
            RelationType(composed_relation, frozenset({"Entity"}), frozenset({"Entity"}))
        )

    stream = _zipf_stream(pool, args.limit, args.zipf_a, rng)
    distinct = len({(h, r) for h, r, _ in stream})
    print(f"  pool={len(pool)} stream={len(stream)} distinct={distinct}")

    kge_cfg = KGEConfig(
        dim=min(settings.kge_dim, 32), epochs=min(settings.kge_epochs, 15), batch_size=4096
    )
    guard = BudgetGuard(args.budget_usd)
    # ONE shared teacher + cache for all arms (LLM-only runs first and populates
    # every distinct pair, so the cascade arms make no further real calls).
    metered = _new_metered(settings, model, nl_template, None)
    shared = SharedAnswerCache(metered, guard, bench.kg)

    arms: list[dict] = []
    truncated = False
    try:
        print("\narm (a) LLM-only (shared deterministic answers) ...")
        r_a = _replay_llm_only(stream, shared)
        arms.append(r_a)
        print(_line(r_a))

        print("\narm (b) cache-cascade — write-back only, Tier-2 off ...")
        cfg_cache = CascadeConfig(
            kge=kge_cfg,
            rule_synthesis=False,
            kge_augment=False,
            write_back=True,
            l2_threshold=TIER2_OFF,
        )
        r_b = _replay_cascade("cache_cascade", stream, bench, ontology, shared, cfg_cache)
        arms.append(r_b)
        print(_line(r_b))

        print("\narm (c) full distillation — write-back + rule synthesis, Tier-2 off ...")
        cfg_full = CascadeConfig(
            kge=kge_cfg,
            rule_synthesis=True,
            kge_augment=True,
            write_back=True,
            l2_threshold=TIER2_OFF,
        )
        r_c = _replay_cascade("full_distillation", stream, bench, ontology, shared, cfg_full)
        arms.append(r_c)
        print(_line(r_c))
    except BudgetExceededError as e:
        truncated = True
        print(f"\n[HARD STOP] {e}")

    by = {a["arm"]: a for a in arms}
    verdict: dict[str, object] = {}
    if "cache_cascade" in by and "full_distillation" in by and "llm_only" in by:
        llm, cache, full = by["llm_only"], by["cache_cascade"], by["full_distillation"]
        verdict = {
            "llm_cost_usd": llm["total_cost_usd"],
            "cache_cost_usd": cache["total_cost_usd"],
            "full_cost_usd": full["total_cost_usd"],
            "amortisation_full_vs_llm": round(llm["total_cost_usd"] / full["total_cost_usd"], 3)
            if full["total_cost_usd"] > 0
            else None,
            "amortisation_cache_vs_llm": round(llm["total_cost_usd"] / cache["total_cost_usd"], 3)
            if cache["total_cost_usd"] > 0
            else None,
            "accuracy_llm": llm["accuracy"],
            "accuracy_cache": cache["accuracy"],
            "accuracy_full": full["accuracy"],
            "accuracy_matched_full_vs_llm": abs(full["accuracy"] - llm["accuracy"]) < 1e-9,
            "full_teacher_calls": full["teacher_calls"],
            "cache_teacher_calls": cache["teacher_calls"],
            "synthesised_rules": full.get("synthesised_rules", []),
        }

    report = {
        "dataset": f"MetaQA-{args.hop}hop-{args.split}",
        "hop": args.hop,
        "design": "controlled: Tier-2 disabled + shared teacher answers across arms",
        "tier2_disabled": True,
        "shared_teacher_answers": True,
        "kg_stats": bench.stats(),
        "real_llm": True,
        "teacher_model_called": settings.xai_model,
        "priced_as_model": model,
        "composed_relation": composed_relation,
        "workload_cap": args.limit,
        "zipf_a": args.zipf_a,
        "seed": args.seed,
        "stream_len": len(stream),
        "distinct_queries": distinct,
        "real_teacher_calls": shared.real_calls,
        "truncated_by_budget": truncated,
        "total_measured_spend_usd": round(guard.spent_usd, 6),
        "arms": arms,
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[spend] real measured: ${guard.spent_usd:.4f} ({shared.real_calls} distinct calls)")
    print(f"wrote {out}")
    if verdict:
        print(
            f"VERDICT (matched accuracy, Tier-2 off): full "
            f"{verdict['amortisation_full_vs_llm']}x, cache "
            f"{verdict['amortisation_cache_vs_llm']}x vs LLM-only; "
            f"acc llm={verdict['accuracy_llm']} cache={verdict['accuracy_cache']} "
            f"full={verdict['accuracy_full']}; rules={verdict['synthesised_rules']}"
        )


if __name__ == "__main__":
    main()
