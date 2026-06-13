"""Controlled compliance amortisation experiment on PrivaCI-Bench GDPR.

Three arms over the SAME shuffled case stream, scored against gold verdicts
and violated articles (gold never enters the graph or the miner):

a. ``llm_only``   — the teacher answers every case.
b. ``cache``      — exact-match cache on the full normalised slot pattern;
                    a miss escalates to the teacher and stores the answer.
c. ``full``       — sound Tier-1 engine first (mined conjunctive rules over
                    the typed compliance graph, ontology-gated); engine
                    abstention falls through to the pattern cache, then the
                    teacher. Every ``--consolidate-every`` NEW teacher answers
                    the miner re-runs over all teacher labels so far and the
                    engine re-materialises, so a mined rule answers UNSEEN
                    cases that share the pattern.

The teacher-stochasticity confound is controlled exactly as in
``run_real_kg_controlled.py``: ONE real metered call per distinct case, shared
across arms; an arm's cost is the sum of the measured per-case costs over the
cases it escalated.

Run (keys in skret /tacet/prod)::

    MSYS_NO_PATHCONV=1 skret run -e prod --path=/tacet/prod -- \
        uv run python experiments/run_privaci_controlled.py --n 300 --teacher gemini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_real_kg_amortization import BudgetExceededError, BudgetGuard  # noqa: E402

from tacet.core.symbolic import RuleEngine  # noqa: E402
from tacet.data.privaci import load_privaci  # noqa: E402
from tacet.data.privaci_graph import ComplianceBenchmark, build_compliance_benchmark  # noqa: E402
from tacet.data.privaci_vocab import load_vocab, normalize_case  # noqa: E402
from tacet.distill.compliance_miner import LabeledCase, mine_compliance_rules  # noqa: E402
from tacet.llm.metering import MeteredTeacher, PriceTable  # noqa: E402
from tacet.llm.teachers.compliance import (  # noqa: E402
    COMPLIANCE_PROMPT_TEMPLATE,
    parse_compliance_answer,
)
from tacet.llm.teachers.llm import GeminiRestTeacher, GrokTeacher  # noqa: E402

Answer = tuple[str, tuple[str, ...]]  # (verdict, violated articles)

#: Cumulative-cost trajectory sampling interval (cases).
CURVE_EVERY = 50


def _build_teacher(name: str):  # noqa: ANN202
    if name == "grok":
        return "grok-4.3", GrokTeacher(
            os.environ["TACET_XAI_API_KEY"],
            "grok-4.3",
            prompt_template=COMPLIANCE_PROMPT_TEMPLATE,
        )
    return "gemini-3.5-flash", GeminiRestTeacher(
        os.environ["TACET_GEMINI_API_KEY"],
        model="gemini-3.5-flash",
        endpoint="vertex",
        qps=None,
        prompt_template=COMPLIANCE_PROMPT_TEMPLATE,
    )


class SharedComplianceCache:
    """One real teacher call per distinct case, shared across all arms.

    The first arm to escalate a case pays the real metered call (charged once
    to the shared budget guard); the parsed answer and its measured cost are
    cached, so every later arm sees the identical answer and books the same
    per-case cost. Arms therefore differ only in routing.
    """

    def __init__(self, metered: MeteredTeacher, guard: BudgetGuard, bench: ComplianceBenchmark):
        self.metered = metered
        self.guard = guard
        self.bench = bench
        self.answers: dict[str, Answer] = {}
        self.cost: dict[str, float] = {}
        self.real_calls = 0

    def answer(self, case_id: str) -> tuple[Answer, float]:
        if case_id not in self.answers:
            resp = self.metered.answer(None, self.bench.case_content[case_id], "verdict")
            self.answers[case_id] = parse_compliance_answer(resp.answers)
            self.cost[case_id] = self.metered.last_cost_usd
            self.real_calls += 1
            self.guard.add(self.metered.last_cost_usd)  # real spend only
        return self.answers[case_id], self.cost[case_id]


def _case_atoms(slots: dict[str, tuple[str, ...]]) -> frozenset[tuple[str, str]]:
    return frozenset((slot, v) for slot, values in slots.items() for v in values)


class ArmScore:
    """Streaming verdict accuracy + article micro-F1 + attributed cost."""

    def __init__(self, oracle: dict[str, Answer]) -> None:
        self.oracle = oracle
        self.n = 0
        self.verdict_ok = 0
        self.tp = self.fp = self.fn = 0
        self.usd = 0.0
        self.teacher_calls = 0
        self.curve: list[float] = []

    def record(self, case_id: str, pred: Answer, usd: float, teacher_call: bool) -> None:
        gold_v, gold_a = self.oracle[case_id]
        pred_a = set(pred[1])
        self.n += 1
        self.verdict_ok += int(pred[0] == gold_v)
        self.tp += len(pred_a & set(gold_a))
        self.fp += len(pred_a - set(gold_a))
        self.fn += len(set(gold_a) - pred_a)
        self.usd += usd
        self.teacher_calls += int(teacher_call)
        if self.n % CURVE_EVERY == 0:
            self.curve.append(round(self.usd, 6))

    def report(self, arm: str, wallclock_s: float, **extra) -> dict:  # noqa: ANN003
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {
            "arm": arm,
            "n": self.n,
            "verdict_acc": round(self.verdict_ok / self.n, 4) if self.n else 0.0,
            "article_micro_f1": round(f1, 4),
            "article_micro_p": round(p, 4),
            "article_micro_r": round(r, 4),
            "teacher_calls": self.teacher_calls,
            "total_cost_usd": round(self.usd, 6),
            "cost_curve_usd": self.curve,
            "wallclock_s": round(wallclock_s, 1),
            **extra,
        }


def _run_llm_only(bench: ComplianceBenchmark, shared: SharedComplianceCache) -> dict:
    score = ArmScore(bench.oracle)
    t0 = time.time()
    for case_id in bench.workload:
        pred, usd = shared.answer(case_id)
        score.record(case_id, pred, usd, teacher_call=True)
    return score.report("llm_only", time.time() - t0)


def _run_cache(
    bench: ComplianceBenchmark,
    shared: SharedComplianceCache,
    atoms: dict[str, frozenset[tuple[str, str]]],
) -> dict:
    score = ArmScore(bench.oracle)
    pattern_cache: dict[frozenset[tuple[str, str]], Answer] = {}
    t0 = time.time()
    for case_id in bench.workload:
        pattern = atoms[case_id]
        if pattern in pattern_cache:
            score.record(case_id, pattern_cache[pattern], 0.0, teacher_call=False)
            continue
        pred, usd = shared.answer(case_id)
        pattern_cache[pattern] = pred
        score.record(case_id, pred, usd, teacher_call=True)
    return score.report("cache", time.time() - t0, distinct_patterns=len(pattern_cache))


def _engine_answer(engine: RuleEngine, case_id: str) -> Answer | None:
    """Tier-1 verdict for a case, or None when the closure says nothing usable.

    A derived ``violates`` edge entails prohibit; a derived ``verdict:permit``
    edge entails permit. Both at once is a rule conflict — abstain and let the
    teacher arbitrate rather than guess.
    """
    violates = engine.query(case_id, "violates")
    verdict = engine.query(case_id, "verdict")
    permit = verdict.answered and "verdict:permit" in verdict.answers
    if violates.answered and not permit:
        arts = tuple(sorted(a.removeprefix("article:") for a in violates.answers))
        return ("prohibit", arts)
    if permit and not violates.answered:
        return ("permit", ())
    return None


def _run_full(
    bench: ComplianceBenchmark,
    shared: SharedComplianceCache,
    atoms: dict[str, frozenset[tuple[str, str]]],
    *,
    consolidate_every: int,
    min_support: int,
    min_confidence: float,
) -> dict:
    score = ArmScore(bench.oracle)
    engine = RuleEngine(bench.ontology)
    engine.materialise(bench.graph)
    pattern_cache: dict[frozenset[tuple[str, str]], Answer] = {}
    labeled: list[LabeledCase] = []
    new_since_mine = 0
    installed: list[dict] = []
    rejected = 0
    engine_hits = cache_hits = 0
    t0 = time.time()

    for case_id in bench.workload:
        pred = _engine_answer(engine, case_id)
        if pred is not None:
            engine_hits += 1
            score.record(case_id, pred, 0.0, teacher_call=False)
            continue
        pattern = atoms[case_id]
        if pattern in pattern_cache:
            cache_hits += 1
            score.record(case_id, pattern_cache[pattern], 0.0, teacher_call=False)
            continue

        pred, usd = shared.answer(case_id)
        pattern_cache[pattern] = pred
        score.record(case_id, pred, usd, teacher_call=True)
        labeled.append(
            LabeledCase(case_id=case_id, atoms=pattern, verdict=pred[0], articles=pred[1])
        )
        new_since_mine += 1

        if new_since_mine >= consolidate_every:
            new_since_mine = 0
            mined = mine_compliance_rules(
                labeled, min_support=min_support, min_confidence=min_confidence
            )
            added = False
            for m in mined:
                if engine.add_rule(m.rule):  # ontology gate + dedupe by name
                    installed.append(
                        {
                            "name": m.rule.name,
                            "target": m.target,
                            "confidence": round(m.confidence, 4),
                            "support": m.support,
                        }
                    )
                    added = True
                elif all(r.name != m.rule.name for r in engine.rules):
                    rejected += 1
            if added:
                engine.materialise(bench.graph)

    return score.report(
        "full",
        time.time() - t0,
        engine_hits=engine_hits,
        cache_hits=cache_hits,
        distinct_patterns=len(pattern_cache),
        rules_installed=installed,
        rules_rejected_by_gate=rejected,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--privaci", default="../PrivaCI-Bench")
    ap.add_argument("--split", default="GDPR")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", choices=("gemini", "grok"), default="gemini")
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--consolidate-every", type=int, default=25)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = load_privaci(args.privaci, split=args.split)
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.n]
    vocab = load_vocab()
    bench = build_compliance_benchmark(cases, vocab)
    atoms = {c.case_id: _case_atoms(normalize_case(c, vocab)) for c in cases}
    distinct = len(set(atoms.values()))
    print(
        f"[setup] {args.split} n={len(cases)} seed={args.seed} "
        f"distinct_patterns={distinct} (repeat-hit ceiling "
        f"{1 - distinct / len(cases):.1%}); budget ${args.budget_usd:.2f}"
    )

    model, raw_teacher = _build_teacher(args.teacher)
    metered = MeteredTeacher(raw_teacher, PriceTable.default(), model=model)
    guard = BudgetGuard(args.budget_usd)
    shared = SharedComplianceCache(metered, guard, bench)
    print(f"  teacher={args.teacher} model={model}")

    arms: list[dict] = []
    truncated = False
    try:
        print("\narm (a) llm_only ...")
        arms.append(_run_llm_only(bench, shared))
        print("\narm (b) cache (exact normalised-pattern match) ...")
        arms.append(_run_cache(bench, shared, atoms))
        print("\narm (c) full (engine + mined rules + pattern cache) ...")
        arms.append(
            _run_full(
                bench,
                shared,
                atoms,
                consolidate_every=args.consolidate_every,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
            )
        )
    except BudgetExceededError as e:
        truncated = True
        print(f"\n[HARD STOP] {e}")

    for r in arms:
        print(
            f"  {r['arm']}: verdict_acc={r['verdict_acc']} art_f1={r['article_micro_f1']} "
            f"calls={r['teacher_calls']} cost=${r['total_cost_usd']:.4f}"
        )

    by = {a["arm"]: a for a in arms}
    verdict: dict[str, object] = {}
    if {"llm_only", "cache", "full"} <= by.keys():
        llm, cache, full = by["llm_only"], by["cache"], by["full"]
        verdict = {
            "amortisation_cache_vs_llm": round(llm["total_cost_usd"] / cache["total_cost_usd"], 3)
            if cache["total_cost_usd"]
            else None,
            "amortisation_full_vs_llm": round(llm["total_cost_usd"] / full["total_cost_usd"], 3)
            if full["total_cost_usd"]
            else None,
            "calls_llm": llm["teacher_calls"],
            "calls_cache": cache["teacher_calls"],
            "calls_full": full["teacher_calls"],
            "verdict_acc": {
                "llm_only": llm["verdict_acc"],
                "cache": cache["verdict_acc"],
                "full": full["verdict_acc"],
            },
            "article_f1": {
                "llm_only": llm["article_micro_f1"],
                "cache": cache["article_micro_f1"],
                "full": full["article_micro_f1"],
            },
            "n_rules_installed": len(full["rules_installed"]),
        }
        print(
            f"\nVERDICT: cache {verdict['amortisation_cache_vs_llm']}x / full "
            f"{verdict['amortisation_full_vs_llm']}x cheaper than llm_only; "
            f"calls {verdict['calls_llm']}/{verdict['calls_cache']}/{verdict['calls_full']}; "
            f"rules={verdict['n_rules_installed']}"
        )

    out = Path(
        args.out or f"experiments/results/privaci_controlled_{args.teacher}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": f"PrivaCI-Bench-{args.split}",
                "design": "controlled: shared teacher answers across arms; gold only in oracle",
                "n": len(cases),
                "seed": args.seed,
                "teacher": args.teacher,
                "model": model,
                "distinct_patterns": distinct,
                "consolidate_every": args.consolidate_every,
                "min_support": args.min_support,
                "min_confidence": args.min_confidence,
                "real_teacher_calls": shared.real_calls,
                "total_measured_spend_usd": round(guard.spent_usd, 6),
                "truncated_by_budget": truncated,
                "arms": arms,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[spend] real measured: ${guard.spent_usd:.4f} ({shared.real_calls} distinct calls)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
