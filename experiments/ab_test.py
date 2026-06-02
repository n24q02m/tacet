"""A/B test harness: TACET cascade vs baseline RAG on domain-style queries (G2.1).

Methodology:

* Take a corpus of N questions (default: a synthetic MetaQA + worldgeo
  mix, since a real production query log is deployment data not shipped
  in the repo).  Users can pass ``--corpus path.jsonl`` with the schema
  ``{head, relation, truth}`` to run on a real log.
* Split the corpus 50/50 (a deterministic alternating split by index)
  into ``Arm A — TACET cascade`` and ``Arm B — baseline RAG``.  The
  baseline RAG mock returns the oracle answer at a fixed cost of
  0.05 USD/query (representing the status quo — every query hits a full
  LLM-RAG round-trip).
* Measure cost / latency / accuracy per arm + emit the JSON
  ``ab_report.json``.

The output feeds the paper §13 deployment table.  In a real-world
deployment, swap ``baseline_rag()`` for an HTTP RPC call to the domain
RAG service; the harness itself does not change.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

from tacet.cascade.router import TACET
from tacet.core.ontology import Ontology
from tacet.data import worldgeo_dataset
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher


def _load_corpus(path: str | None, n: int, seed: int) -> list[dict]:
    """Load A/B corpus.  Falls back to a synthetic mix when no path is given.

    Each entry: ``{'head', 'relation', 'truth': list[str]}``.
    """
    if path:
        with Path(path).open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()][:n]
    # Synthetic fallback: half from worldgeo (geography QA), half from
    # synthetic-org benchmark (HR / org QA).  This is methodologically
    # honest — we are not claiming deployment-specific results, but exercising
    # the A/B harness on a similar mix of cheap-symbolic + harder-novel
    # questions.
    ds = worldgeo_dataset(seed=seed)
    bench = generate(BenchmarkConfig(seed=seed))
    corpus: list[dict] = []
    # geography slice
    for h, r, t in ds.all_triples()[: n // 2]:
        corpus.append({"head": h, "relation": r, "truth": [t]})
    # org-bench slice
    for (h, r), truth in list(bench.truth.items())[: n // 2]:
        corpus.append({"head": h, "relation": r, "truth": list(truth)})
    random.Random(seed).shuffle(corpus)
    return corpus[:n]


def _baseline_rag(head: str, relation: str, truth: list[str]) -> dict:
    """Mock baseline RAG: returns the truth at a fixed per-call cost.

    Replace this with an HTTP call to a live RAG service in
    production.  The cost (0.05) is the order-of-magnitude USD a
    Gemini-Flash RAG round-trip costs as of 2026-Q2; tune to the real
    rate at the deployment site.
    """
    t0 = time.time()
    # Simulate RAG retrieval + LLM call latency.  Range ~50-200 ms.
    time.sleep(0.05)
    return {
        "answers": list(truth),
        "cost": 0.05,
        "latency_ms": (time.time() - t0) * 1000,
        "tier": "rag",
    }


def _tacet_arm(tacet: TACET, head: str, relation: str) -> dict:
    ans = tacet.ask(head, relation)
    return {
        "answers": list(ans.answers),
        "cost": ans.cost,
        "latency_ms": ans.latency_ms,
        "tier": str(ans.tier),
    }


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    s = sorted(xs)
    n = len(s)
    return {
        "p50": float(st.median(s)),
        "p95": s[min(n - 1, int(0.95 * n))],
        "p99": s[min(n - 1, int(0.99 * n))],
        "mean": float(st.mean(s)),
    }


def _quality(answers: list[str], truth: list[str]) -> bool:
    return bool(truth) and set(truth).issubset(set(answers))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        default=None,
        help="JSONL file with {head, relation, truth} per line; "
        "default synthesises worldgeo + benchmark mix",
    )
    ap.add_argument(
        "--n", type=int, default=100, help="number of queries; default 100 (50 per arm)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/ab_report.json")
    args = ap.parse_args()

    print(f"loading corpus (n={args.n}, seed={args.seed})...")
    corpus = _load_corpus(args.corpus, args.n, args.seed)
    print(f"  loaded {len(corpus)} queries")
    if len(corpus) < 4:
        raise SystemExit("corpus too small for A/B test")

    # 50/50 alternating split (deterministic, reproducible).
    arm_a = [q for i, q in enumerate(corpus) if i % 2 == 0]
    arm_b = [q for i, q in enumerate(corpus) if i % 2 == 1]
    print(f"  arm A (TACET cascade) = {len(arm_a)} queries")
    print(f"  arm B (baseline RAG)   = {len(arm_b)} queries")

    # Build TACET cascade for arm A.  Use the geography world graph +
    # synthetic-org benchmark unioned as the substrate; oracle teacher
    # so the A/B is offline (no Gemini dependency).
    print("  warming up TACET (arm A)...")
    ds = worldgeo_dataset(seed=args.seed)
    bench = generate(BenchmarkConfig(seed=args.seed))
    truth_map: dict[tuple[str, str], list[str]] = {}
    for h, r, t in ds.all_triples():
        truth_map.setdefault((h, r), []).append(t)
    for (h, r), t_list in bench.truth.items():
        truth_map.setdefault((h, r), []).extend(list(t_list))
    teacher = OracleTeacher(lambda h, r: truth_map.get((h, r), []))
    # Merge graphs: re-use one of them as the base; TACET can ingest
    # the other after warmup.  Keep it simple — use bench.graph (richer).
    tacet = TACET(bench.graph.copy(), Ontology.induce(bench.graph), teacher)
    tacet.warmup()

    print("  running arm A (TACET cascade)...")
    t0 = time.time()
    arm_a_records = [
        _tacet_arm(tacet, q["head"], q["relation"]) | {"truth": q["truth"]} for q in arm_a
    ]
    arm_a_wall = time.time() - t0

    print("  running arm B (baseline RAG mock)...")
    t0 = time.time()
    arm_b_records = [
        _baseline_rag(q["head"], q["relation"], q["truth"]) | {"truth": q["truth"]} for q in arm_b
    ]
    arm_b_wall = time.time() - t0

    def _summarise(records: list[dict]) -> dict:
        costs = [r["cost"] for r in records]
        lats = [r["latency_ms"] / 1000.0 for r in records]
        accuracies = [_quality(r["answers"], r["truth"]) for r in records]
        tier_counts: dict[str, int] = {}
        for r in records:
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
        return {
            "n": len(records),
            "total_cost": sum(costs),
            "accuracy": sum(accuracies) / len(records) if records else 0.0,
            "cost_percentiles": _percentiles(costs),
            "latency_s_percentiles": _percentiles(lats),
            "tier_counts": tier_counts,
        }

    report = {
        "seed": args.seed,
        "corpus_size": len(corpus),
        "corpus_path": args.corpus,
        "arm_a_tacet": {
            "wallclock_s": round(arm_a_wall, 2),
            **_summarise(arm_a_records),
        },
        "arm_b_baseline_rag": {
            "wallclock_s": round(arm_b_wall, 2),
            **_summarise(arm_b_records),
        },
    }
    a_cost = report["arm_a_tacet"]["total_cost"]
    b_cost = report["arm_b_baseline_rag"]["total_cost"]
    if a_cost > 0:
        report["cost_reduction_x"] = round(b_cost / a_cost, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(
        f"  arm A: cost ${a_cost:.4f}  acc {report['arm_a_tacet']['accuracy']:.3f}  "
        f"wallclock {arm_a_wall:.1f}s"
    )
    print(
        f"  arm B: cost ${b_cost:.4f}  acc {report['arm_b_baseline_rag']['accuracy']:.3f}  "
        f"wallclock {arm_b_wall:.1f}s"
    )
    if "cost_reduction_x" in report:
        print(f"  cost reduction (B/A): {report['cost_reduction_x']:.2f}x")


if __name__ == "__main__":
    main()
