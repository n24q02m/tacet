"""MetaQA cost-decay stream (P4): external-validity for the headline claim.

Streams a Zipfian-repeat workload through the cascade and shows the blended
cost-per-query *bend* downward as online distillation migrates knowledge to
the cheap tiers.  Two runs:

* Run 1 (real teacher, e.g. Grok 4.3): cost trajectory + tier migration +
  cost-reduction vs an LLM-only baseline (computed analytically as n*c3, so we
  do NOT double the API spend with a separate LLM-only pass).
* Run 2 (oracle teacher, free): cache-only vs full ablation, evaluated on a
  held-out set of *unseen* queries -- only rule synthesis / KGE generalise to
  unseen heads, so the gap there is distillation beyond caching.

Costs are the paper's model-cost units (TIER_COST); real API spend is the
separate per-call Grok charge (we log the teacher-call count).

    TACET_TEACHER=grok TACET_XAI_API_KEY=... \\
      uv run python experiments/run_metaqa_costdecay.py --stream-len 800 --ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from metaqa2hop_matcher import relation_for_2hop  # noqa: E402
from run_metaqa import _relation_for_question  # noqa: E402

from tacet.cascade.router import TACET  # noqa: E402
from tacet.core.ontology import Ontology  # noqa: E402
from tacet.data.metaqa import load_metaqa  # noqa: E402
from tacet.llm.teacher import OracleTeacher  # noqa: E402
from tacet.llm.teachers import build_teacher_from_settings  # noqa: E402
from tacet.serve.config import TIER_COST, CascadeConfig, KGEConfig  # noqa: E402
from tacet.serve.settings import load_settings  # noqa: E402


def _resolve_pool(bench, matcher) -> list[tuple[str, str]]:  # noqa: ANN001
    pool = []
    for q in bench.questions:
        r = matcher(q.question)
        if r and q.head in bench.entities:
            pool.append((q.head, r))
    return pool


def _zipf_stream(pool, n, a, rng):  # noqa: ANN001
    """Sample n query instances Zipfian over distinct queries (hot queries
    repeat -> exact-repeat cache hits + same-relation rule reuse)."""
    out: list[tuple[str, str]] = []
    while len(out) < n:
        draw = rng.zipf(a, size=n)
        draw = draw[draw <= len(pool)] - 1
        out.extend(pool[i] for i in draw)
    return out[:n]


def _run_stream(kg, ontology, teacher, stream, config, consolidate_every):  # noqa: ANN001
    ak = TACET(kg.copy(), ontology, teacher, config=config)
    ak.warmup()
    costs, tiers = [], []
    for i, (h, r) in enumerate(stream):
        ans = ak.ask(h, r)
        costs.append(ans.cost)
        tiers.append(ans.tier)
        if consolidate_every and (i + 1) % consolidate_every == 0:
            ak.consolidate()
    return ak, costs, tiers


def _windowed(costs, w=50):  # noqa: ANN001
    return [round(float(np.mean(costs[i : i + w])), 5) for i in range(0, len(costs), w)]


def main() -> None:  # noqa: PLR0914, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--metaqa-root", default="data/MetaQA")
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--stream-len", type=int, default=800)
    ap.add_argument("--zipf-a", type=float, default=1.4)
    ap.add_argument("--consolidate-every", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=150)
    ap.add_argument(
        "--ablation", action="store_true", help="also run the oracle cache-vs-full ablation (free)"
    )
    ap.add_argument(
        "--force-oracle",
        action="store_true",
        help="force the oracle teacher for Run1 too (bypass the "
        "settings backfill that promotes oracle->grok). Correct "
        "for hop=2 where derived relations aren't real questions.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/metaqa_costdecay.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    c3 = TIER_COST[3]

    bench = load_metaqa(args.metaqa_root, hop=args.hop, split=args.split)
    ontology = Ontology.induce(bench.kg)
    # hop=1 -> base relations; hop=2 -> derived composition relations.  The
    # 2-hop derived relations are NOT base KB edges, so Tier 1 misses them
    # until the miner synthesises the composition -- which is exactly the
    # distillation cost-decay we want to measure.
    if args.hop == 2:
        matcher = relation_for_2hop
    else:
        matcher = lambda q: _relation_for_question(q, bench.relations)  # noqa: E731
    pool = _resolve_pool(bench, matcher)
    rng.shuffle(pool)
    holdout = pool[: args.holdout]
    stream_pool = pool[args.holdout :]
    stream = _zipf_stream(stream_pool, args.stream_len, args.zipf_a, rng)
    distinct = len(set(stream))
    print(
        f"pool={len(pool)} holdout={len(holdout)} stream={len(stream)} "
        f"distinct_in_stream={distinct} ({100 * distinct / len(stream):.0f}%)"
    )

    settings = load_settings()
    teacher = None if args.force_oracle else build_teacher_from_settings(settings)
    teacher_name = settings.teacher if teacher is not None else "oracle"
    gold = {(q.head, matcher(q.question) or "?"): q.answers for q in bench.questions}
    if teacher is None:
        teacher = OracleTeacher(lambda h, r: gold.get((h, r), []))
    print(f"teacher: {teacher_name}")

    kge_cfg = KGEConfig(dim=settings.kge_dim, epochs=min(settings.kge_epochs, 80))

    # --- Run 1: real teacher, full config, cost trajectory ---
    t0 = time.time()
    cfg_full = CascadeConfig(kge=kge_cfg)
    ak, costs, tiers = _run_stream(
        bench.kg, ontology, teacher, stream, cfg_full, args.consolidate_every
    )
    blended = sum(costs)
    llm_only = len(stream) * c3
    tier_early = {t: tiers[:200].count(t) for t in (1, 2, 3)}
    tier_late = {t: tiers[-200:].count(t) for t in (1, 2, 3)}
    print(
        f"Run1 ({teacher_name}): blended ${blended:.3f} vs LLM-only "
        f"${llm_only:.3f}  ({llm_only / blended:.2f}x)  in {time.time() - t0:.0f}s"
    )
    print(f"  tier early(first200)={tier_early}  late(last200)={tier_late}")
    print(f"  windowed cost/query: {_windowed(costs)}")

    report = {
        "dataset": bench.name,
        "teacher": teacher_name,
        "stream_len": len(stream),
        "distinct_in_stream": distinct,
        "zipf_a": args.zipf_a,
        "consolidate_every": args.consolidate_every,
        "run1_full": {
            "blended_cost": round(blended, 4),
            "llm_only_cost": round(llm_only, 4),
            "cost_reduction_x": round(llm_only / blended, 3) if blended else None,
            "windowed_cost_per_query": _windowed(costs),
            "tier_early_first200": tier_early,
            "tier_late_last200": tier_late,
            "synthesised_rules": list(ak.synthesised_rules),
        },
    }

    # --- Run 2: oracle ablation cache-only vs full on held-out unseen ---
    if args.ablation:
        oracle = OracleTeacher(lambda h, r: gold.get((h, r), []))
        abl = {}
        for name, cfg in (
            ("full", CascadeConfig(kge=kge_cfg)),
            ("cache_only", CascadeConfig(kge=kge_cfg, rule_synthesis=False, kge_augment=False)),
        ):
            ak_a, _, _ = _run_stream(
                bench.kg, ontology, oracle, stream, cfg, args.consolidate_every
            )
            # held-out: unseen queries; only rules/KGE can cheapen these
            ho_cost = 0.0
            ho_correct = 0
            ho_tiers = {1: 0, 2: 0, 3: 0}
            for h, r in holdout:
                a = ak_a.ask(h, r)
                ho_cost += a.cost
                ho_tiers[a.tier] += 1
                g = set(gold.get((h, r), []))
                if g and g.issubset(set(a.answers)):
                    ho_correct += 1
            abl[name] = {
                "holdout_cost": round(ho_cost, 4),
                "holdout_accuracy": (round(ho_correct / len(holdout), 4) if holdout else 0.0),
                "holdout_tiers": ho_tiers,
                "synthesised_rules": len(ak_a.synthesised_rules),
            }
            print(
                f"Run2 ablation [{name}]: held-out ${ho_cost:.3f}  "
                f"acc={ho_correct}/{len(holdout)}  tiers={ho_tiers}"
            )
        report["run2_ablation_oracle"] = abl

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
