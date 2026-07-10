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

A free ``TACET_TEACHER=oracle`` mode reuses the identical controlled design with a
ground-truth (optionally noisy) oracle instead of Grok: it costs $0, so the
imperfect-teacher noise sweep (``run_oracle_noise_sweep.py``, research entry E11)
and the real-teacher measurement now live under ONE design. In oracle mode the
decisive metric is the per-arm teacher **call count** (measured USD is 0 for
every arm by construction); ``TACET_ORACLE_ERROR_RATE`` dials the corruption.

Run::

    # real teacher
    export TACET_TEACHER=grok TACET_XAI_API_KEY=<key> TACET_KGE_BACKEND=numpy
    uv run python experiments/run_real_kg_controlled.py --hop 1 --seed 0

    # free (noisy) oracle
    export TACET_TEACHER=oracle TACET_ORACLE_ERROR_RATE=0.2
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
    _oracle_gold_from_pool,
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


def _teacher_answer_accuracy(
    shared: SharedAnswerCache, gold_map: dict[str, frozenset[str]]
) -> tuple[float, int, int]:
    """The teacher's OWN answer accuracy over the distinct pairs it answered.

    Computed exactly from the shared cache and the known gold: a cached answer is
    correct iff it is non-empty and every entity it returned is a true gold tail
    (``set(answer) <= gold``). That reproduces :class:`OracleTeacher`'s ``correct``
    bookkeeping (an uncorrupted answer is the full gold set; a corrupted answer is
    a single entity, correct only if it happens to be in gold) WITHOUT touching the
    teacher's RNG. It is the curve's x-axis for the E11 noise sweep and is NOT
    ``1 - error_rate``: ``error_rate`` is a per-key corruption *probability* (the
    realised fraction differs on a finite workload), and a corrupted answer can
    coincidentally still be a real gold tail, so it counts as correct here.

    Returns ``(accuracy, n_correct, n_pairs)``.
    """
    correct = 0
    total = 0
    for (h, r), ans in shared.answers.items():
        gold = gold_map.get(f"{h}\t{r}")
        if not gold:
            continue
        total += 1
        aset = set(ans)
        if aset and aset <= set(gold):
            correct += 1
    return (round(correct / total, 6) if total else 0.0, correct, total)


def run_controlled(
    *,
    metaqa_root: str = "data/MetaQA",
    hop: int = 1,
    split: str = "test",
    limit: int = 300,
    zipf_a: float = 1.5,
    budget_usd: float = 1.5,
    seed: int = 0,
    composition: str | None = None,
    oracle_error_rate: float = 0.0,
    settings=None,  # noqa: ANN001
    bench=None,  # noqa: ANN001
    verbose: bool = True,
) -> dict:
    """Run the controlled cost-at-matched-accuracy pipeline and return its report.

    The importable core of this module: ``main()`` is a thin argparse wrapper over
    it, and ``run_oracle_noise_sweep.sweep`` calls it once per (error_rate, seed)
    cell (loading the MetaQA bench ONCE and passing it in via ``bench``). ``bench``
    and ``settings`` are injectable so the sweep can share them and the tests can
    drive the pipeline on a tiny synthetic KG without MetaQA. With
    ``settings.teacher == "oracle"`` the single teacher feeding the shared cache is
    a free (optionally noisy) ground-truth oracle, so ``TACET_TEACHER=oracle``
    works; ``oracle_error_rate`` is the corruption dial.
    """
    model = os.environ.get("TACET_PRICE_MODEL", "grok-4.3")
    rng = np.random.default_rng(seed)
    if settings is None:
        settings = load_settings()
    if bench is None:
        bench = load_metaqa(metaqa_root, hop=1, split=split)
    oracle_mode = settings.teacher == "oracle"
    if verbose:
        print(f"[cap] {limit} queries; hard budget ${budget_usd:.2f}; Tier-2 DISABLED")
        print(f"  kg stats: {bench.stats()}")

    if oracle_mode:
        if verbose:
            print("  teacher=ORACLE (ground-truth, $0, instant) — mechanism/noise test, no cost")
    else:
        if not getattr(settings, "xai_api_key", None):
            raise SystemExit(
                "controlled run needs a teacher: TACET_TEACHER=grok + TACET_XAI_API_KEY, "
                "or TACET_TEACHER=oracle for the free (noisy) oracle mode"
            )
        if verbose:
            print(f"  teacher=grok model={settings.xai_model} (priced as {model})")

    nl_template = None
    composed_relation = None
    if hop == 1:
        pool = _build_workload(bench, limit_pool=max(limit, 400), rng=rng)
        ontology = Ontology.induce(bench.kg)
    else:
        comp_name = composition or next(
            (k for k, v in COMPOSITIONS.items() if v.get("hop") == hop), None
        )
        if comp_name is None:
            raise SystemExit(f"no composition for hop={hop}")
        spec = COMPOSITIONS[comp_name]
        nl_template = spec["nl"]
        composed_relation = spec["kg_relation"]
        legs_desc = " . ".join(f"{'~' if inv else ''}{r}" for r, inv in spec["legs"])
        if verbose:
            print(f"  COMPOSITION {comp_name!r} (hop={hop}): {composed_relation} := {legs_desc}")
        pool = _build_composed_workload(bench.kg, spec, limit_pool=max(limit, 400), rng=rng)
        if not pool:
            raise SystemExit(f"composition {comp_name!r} produced an empty pool")
        ontology = Ontology.induce(bench.kg)
        ontology.add_relation_type(
            RelationType(composed_relation, frozenset({"Entity"}), frozenset({"Entity"}))
        )

    stream = _zipf_stream(pool, limit, zipf_a, rng)
    distinct = len({(h, r) for h, r, _ in stream})
    if verbose:
        print(f"  pool={len(pool)} stream={len(stream)} distinct={distinct}")

    kge_cfg = KGEConfig(
        dim=min(settings.kge_dim, 32), epochs=min(settings.kge_epochs, 15), batch_size=4096
    )
    guard = BudgetGuard(budget_usd)
    # Ground-truth gold map over the whole pool (a superset of the stream). It
    # feeds the oracle teacher AND lets us measure the teacher's own answer
    # accuracy; the real-teacher path ignores it inside ``_new_metered``.
    gold_map = _oracle_gold_from_pool(pool)
    # ONE shared teacher + cache for all arms (LLM-only runs first and populates
    # every distinct pair, so the cascade arms make no further real calls). When
    # ``TACET_TEACHER=oracle`` this single teacher IS the (noisy) oracle.
    metered = _new_metered(
        settings, model, nl_template, gold_map, error_rate=oracle_error_rate, seed=seed
    )
    shared = SharedAnswerCache(metered, guard, bench.kg)

    arms: list[dict] = []
    truncated = False
    try:
        if verbose:
            print("\narm (a) LLM-only (shared deterministic answers) ...")
        r_a = _replay_llm_only(stream, shared)
        arms.append(r_a)
        if verbose:
            print(_line(r_a))

        if verbose:
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
        if verbose:
            print(_line(r_b))

        if verbose:
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
        if verbose:
            print(_line(r_c))
    except BudgetExceededError as e:
        truncated = True
        if verbose:
            print(f"\n[HARD STOP] {e}")

    teacher_acc, teacher_correct, teacher_total = _teacher_answer_accuracy(shared, gold_map)

    by = {a["arm"]: a for a in arms}
    verdict: dict[str, object] = {}
    if "cache_cascade" in by and "full_distillation" in by and "llm_only" in by:
        llm, cache, full = by["llm_only"], by["cache_cascade"], by["full_distillation"]
        cache_calls = cache["teacher_calls"]
        full_calls = full["teacher_calls"]
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
            "full_teacher_calls": full_calls,
            "cache_teacher_calls": cache_calls,
            # E11 rule-vs-cache metric: relative reduction in teacher calls of the
            # rule arm (full) vs the cache arm on structurally unseen heads, at
            # matched accuracy. This is the sweep's decisive, price-independent
            # signal (with a free oracle every arm reads USD=0).
            "calls_saved_vs_cache": cache_calls - full_calls,
            "calls_saved_pct": (
                round(100.0 * (cache_calls - full_calls) / cache_calls, 2)
                if cache_calls > 0
                else None
            ),
            "accuracy_matched": abs(cache["accuracy"] - full["accuracy"]) < 1e-9,
            "synthesised_rules": full.get("synthesised_rules", []),
        }

    teacher_kind = "oracle" if oracle_mode else settings.xai_model
    report = {
        "dataset": f"MetaQA-{hop}hop-{split}",
        "hop": hop,
        "design": "controlled: Tier-2 disabled + shared teacher answers across arms",
        "tier2_disabled": True,
        "shared_teacher_answers": True,
        "kg_stats": bench.stats(),
        "real_llm": not oracle_mode,
        "teacher_kind": teacher_kind,
        "oracle_error_rate": oracle_error_rate if oracle_mode else None,
        "noise_mode": "per_key" if oracle_mode else None,
        "teacher_model_called": settings.xai_model,
        "priced_as_model": model,
        "composed_relation": composed_relation,
        "workload_cap": limit,
        "zipf_a": zipf_a,
        "seed": seed,
        "stream_len": len(stream),
        "distinct_queries": distinct,
        "teacher_answer_accuracy": teacher_acc,
        "teacher_answers_correct": teacher_correct,
        "teacher_answers_total": teacher_total,
        "real_teacher_calls": shared.real_calls,
        "truncated_by_budget": truncated,
        "total_measured_spend_usd": round(guard.spent_usd, 6),
        "arms": arms,
        "verdict": verdict,
    }
    if verbose:
        print(
            f"\n[spend] measured: ${guard.spent_usd:.4f} ({shared.real_calls} distinct calls); "
            f"teacher answer accuracy={teacher_acc} ({teacher_correct}/{teacher_total})"
        )
        if verdict:
            print(
                f"VERDICT (matched accuracy, Tier-2 off): full "
                f"{verdict['amortisation_full_vs_llm']}x, cache "
                f"{verdict['amortisation_cache_vs_llm']}x vs LLM-only; "
                f"calls full={verdict['full_teacher_calls']} vs cache="
                f"{verdict['cache_teacher_calls']} ({verdict['calls_saved_pct']}% saved); "
                f"acc llm={verdict['accuracy_llm']} cache={verdict['accuracy_cache']} "
                f"full={verdict['accuracy_full']}; rules={verdict['synthesised_rules']}"
            )
    return report


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

    # Oracle-teacher noise dial (fraction of oracle answers corrupted); read the
    # same way the amortization runner does. Ignored by the real-teacher path.
    oracle_error_rate = float(os.environ.get("TACET_ORACLE_ERROR_RATE", "0.0"))
    report = run_controlled(
        metaqa_root=args.metaqa_root,
        hop=args.hop,
        split=args.split,
        limit=args.limit,
        zipf_a=args.zipf_a,
        budget_usd=args.budget_usd,
        seed=args.seed,
        composition=args.composition,
        oracle_error_rate=oracle_error_rate,
        verbose=True,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
