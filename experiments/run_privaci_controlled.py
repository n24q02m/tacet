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

import numpy as np

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
    NL_STRATEGY_PROMPT_TEMPLATE,
    parse_compliance_answer,
)
from tacet.llm.teachers.llm import GeminiRestTeacher, GrokTeacher  # noqa: E402

Answer = tuple[str, tuple[str, ...]]  # (verdict, violated articles)

#: Cumulative-cost trajectory sampling interval (cases).
CURVE_EVERY = 50


def _build_teacher(name: str, prompt_template: str = COMPLIANCE_PROMPT_TEMPLATE):  # noqa: ANN202
    """Build a teacher for the two approved models only (grok-4.3, gemini-3.5-flash).

    The same builder serves the frontier teacher and the nl_strategy in-context
    model: per the no-routing directive, nl_strategy re-prompts the SAME model
    as the frontier (just with retrieved guidelines), so the only difference
    measured is NL-memory vs executable symbolic rules -- no cheaper model is
    introduced and no model-capability confound enters the comparison.
    """
    if name == "grok":
        return "grok-4.3", GrokTeacher(
            os.environ["TACET_XAI_API_KEY"],
            "grok-4.3",
            prompt_template=prompt_template,
        )
    return "gemini-3.5-flash", GeminiRestTeacher(
        os.environ["TACET_GEMINI_API_KEY"],
        model="gemini-3.5-flash",
        endpoint="vertex",
        qps=None,
        prompt_template=prompt_template,
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
    """Streaming verdict accuracy + article micro-F1 + attributed cost.

    ``score_from`` enables suffix-only scoring: cases with ``idx < score_from``
    are still PROCESSED by the arm (teacher calls, rule mining happen — that is
    the training/warm-up exposure) but are NOT scored or cost-attributed, so an
    online arm and a compile-once arm can be compared on the identical test
    suffix without the prefix confounding the comparison.
    """

    def __init__(self, oracle: dict[str, Answer], score_from: int = 0) -> None:
        self.oracle = oracle
        self.score_from = score_from
        self.n = 0
        self.verdict_ok = 0
        self.tp = self.fp = self.fn = 0
        self.usd = 0.0
        self.teacher_calls = 0
        self.curve: list[float] = []

    def record(self, case_id: str, pred: Answer, usd: float, teacher_call: bool, idx: int) -> None:
        if idx < self.score_from:
            return
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


def _run_llm_only(
    bench: ComplianceBenchmark, shared: SharedComplianceCache, *, score_from: int = 0
) -> dict:
    score = ArmScore(bench.oracle, score_from)
    t0 = time.time()
    for idx, case_id in enumerate(bench.workload):
        pred, usd = shared.answer(case_id)
        score.record(case_id, pred, usd, teacher_call=True, idx=idx)
    return score.report("llm_only", time.time() - t0)


def _run_cache(
    bench: ComplianceBenchmark,
    shared: SharedComplianceCache,
    atoms: dict[str, frozenset[tuple[str, str]]],
    *,
    score_from: int = 0,
) -> dict:
    score = ArmScore(bench.oracle, score_from)
    pattern_cache: dict[frozenset[tuple[str, str]], Answer] = {}
    t0 = time.time()
    for idx, case_id in enumerate(bench.workload):
        pattern = atoms[case_id]
        if pattern in pattern_cache:
            score.record(case_id, pattern_cache[pattern], 0.0, teacher_call=False, idx=idx)
            continue
        pred, usd = shared.answer(case_id)
        pattern_cache[pattern] = pred
        score.record(case_id, pred, usd, teacher_call=True, idx=idx)
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
    score_from: int = 0,
) -> dict:
    score = ArmScore(bench.oracle, score_from)
    engine = RuleEngine(bench.ontology)
    engine.materialise(bench.graph)
    pattern_cache: dict[frozenset[tuple[str, str]], Answer] = {}
    labeled: list[LabeledCase] = []
    new_since_mine = 0
    installed: list[dict] = []
    rejected = 0
    engine_hits = cache_hits = 0
    t0 = time.time()

    for idx, case_id in enumerate(bench.workload):
        pred = _engine_answer(engine, case_id)
        if pred is not None:
            engine_hits += 1
            score.record(case_id, pred, 0.0, teacher_call=False, idx=idx)
            continue
        pattern = atoms[case_id]
        if pattern in pattern_cache:
            cache_hits += 1
            score.record(case_id, pattern_cache[pattern], 0.0, teacher_call=False, idx=idx)
            continue

        pred, usd = shared.answer(case_id)
        pattern_cache[pattern] = pred
        score.record(case_id, pred, usd, teacher_call=True, idx=idx)
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


def _run_compile_once(
    bench: ComplianceBenchmark,
    shared: SharedComplianceCache,
    atoms: dict[str, frozenset[tuple[str, str]]],
    *,
    prefix_k: int,
    min_support: int,
    min_confidence: float,
) -> dict:
    """Offline-distilled baseline: mine ONCE over a prefix, freeze, apply to suffix.

    Isolates the value of ONLINE consolidation: the prefix is escalated to the
    teacher and mined once with the SAME miner/gate/engine as the full arm, then
    the ruleset is FROZEN. On the test suffix a frozen-engine hit is free; a miss
    escalates to the teacher but NEVER re-mines. Scored on the suffix only
    (``score_from=prefix_k``), so the full arm must also be suffix-scored to
    compare fairly.
    """
    engine = RuleEngine(bench.ontology)
    engine.materialise(bench.graph)
    labeled: list[LabeledCase] = []
    installed: list[dict] = []
    rejected = 0
    score = ArmScore(bench.oracle, score_from=prefix_k)
    t0 = time.time()

    # ----- PHASE 1: prefix -> escalate all, mine once, freeze ----------------
    for idx in range(min(prefix_k, len(bench.workload))):
        case_id = bench.workload[idx]
        pred, usd = shared.answer(case_id)
        score.record(case_id, pred, usd, teacher_call=True, idx=idx)  # not scored (idx<k)
        labeled.append(
            LabeledCase(case_id=case_id, atoms=atoms[case_id], verdict=pred[0], articles=pred[1])
        )
    mined = mine_compliance_rules(labeled, min_support=min_support, min_confidence=min_confidence)
    for m in mined:
        if engine.add_rule(m.rule):
            installed.append(
                {
                    "name": m.rule.name,
                    "target": m.target,
                    "confidence": round(m.confidence, 4),
                    "support": m.support,
                }
            )
        elif all(r.name != m.rule.name for r in engine.rules):
            rejected += 1
    if installed:
        engine.materialise(bench.graph)
    frozen_rule_count = len(engine.rules)

    # ----- PHASE 2: test suffix -> frozen engine, no re-mining ---------------
    engine_hits = 0
    for idx in range(prefix_k, len(bench.workload)):
        case_id = bench.workload[idx]
        pred = _engine_answer(engine, case_id)
        if pred is not None:
            engine_hits += 1
            score.record(case_id, pred, 0.0, teacher_call=False, idx=idx)
        else:
            pred, usd = shared.answer(case_id)
            score.record(case_id, pred, usd, teacher_call=True, idx=idx)
        assert len(engine.rules) == frozen_rule_count, "compile_once ruleset must stay frozen"

    return score.report(
        "compile_once",
        time.time() - t0,
        prefix_k=prefix_k,
        n_test=len(bench.workload) - prefix_k,
        engine_hits=engine_hits,
        rules_installed=installed,
        rules_rejected_by_gate=rejected,
    )


def _run_nl_strategy(
    bench: ComplianceBenchmark,
    shared: SharedComplianceCache,
    *,
    weak_metered: MeteredTeacher,
    embed,  # noqa: ANN001 — Callable[[list[str]], np.ndarray]
    guard: BudgetGuard,
    tau: float = 0.5,
    top_k: int = 2,
    score_from: int = 0,
) -> dict:
    """Inter-Cascade-style online NL-distillation baseline (arXiv:2509.22984).

    Faithful port of the deferral-to-learning cascade: a growing repository of
    natural-language guidelines (distilled from earlier deferred cases) is
    retrieved by case similarity. A case is COVERED when its top retrieved
    guideline clears a similarity gate ``tau`` -- then an in-context model call
    conditioned on the retrieved guidelines answers it. Per the no-routing
    directive this model is the SAME as the frontier (no cheaper tier), so the
    comparison is purely NL-memory vs executable symbolic rules: the covered
    case STILL costs a full LLM call (a per-query floor the free symbolic engine
    eliminates) and its answer carries NO replayable proof tree. An UNCOVERED
    case (empty/low-similarity repo, i.e. the cold start) DEFERS to the same
    shared frontier teacher and appends a new guideline. The repo grows and
    coverage rises -- exactly Inter-Cascade's accumulation -- but because every
    answer is still an LLM call, NL strategies do not amortise teacher cost the
    way executable rules do.

    Deferral is gated on retrieval coverage rather than the model's
    self-reported confidence, which is uncalibrated (gemini-3.5-flash returns
    >=0.9 on essentially every case, so a confidence gate never fires); coverage
    is deterministic, reproducible, and actually exercises the repository.
    """
    score = ArmScore(bench.oracle, score_from)
    strategies: list[str] = []
    vectors: list[np.ndarray] = []
    weak_cost = 0.0
    defers = 0
    accepts = 0
    t0 = time.time()

    for idx, case_id in enumerate(bench.workload):
        scenario = bench.case_content[case_id]
        q = np.asarray(embed([scenario]), dtype=np.float32)[0]
        max_sim, top = -1.0, []
        if vectors:
            mat = np.vstack(vectors)
            qn = q / (np.linalg.norm(q) + 1e-12)
            mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
            sims = mn @ qn
            order = list(np.argsort(-sims))
            top = order[:top_k]
            max_sim = float(sims[order[0]])

        if max_sim >= tau:  # COVERED -> cheap weak call conditioned on guidelines
            block = "Guidelines from earlier cases:\n" + "\n".join(
                f"- {strategies[i]}" for i in top
            )
            resp = weak_metered.answer(None, f"{block}\n\nScenario:\n{scenario}", "verdict")
            guard.add(weak_metered.last_cost_usd)
            weak_cost += weak_metered.last_cost_usd
            verdict, arts = parse_compliance_answer(resp.answers)
            if verdict in ("permit", "prohibit"):
                accepts += 1
                score.record(case_id, (verdict, arts), weak_metered.last_cost_usd, False, idx)
                continue
            # weak model failed to produce a verdict -> fall through to defer
            extra_weak = weak_metered.last_cost_usd
        else:
            extra_weak = 0.0

        defers += 1
        pred, tcost = shared.answer(case_id)
        score.record(case_id, pred, extra_weak + tcost, True, idx)
        snippet = scenario[:160].replace("\n", " ")
        arts_txt = ", ".join(pred[1]) if pred[1] else "none"
        strategies.append(
            f"When a scenario resembles: {snippet} -> {pred[0]} (articles: {arts_txt})"
        )
        vectors.append(q)

    return score.report(
        "nl_strategy",
        time.time() - t0,
        n_strategies=len(strategies),
        defers=defers,
        accepts=accepts,
        weak_llm_cost_usd=round(weak_cost, 6),
        tau=tau,
        weak_model=weak_metered.model,
    )


def _load_embedder():  # noqa: ANN202
    """Lazy sentence-transformers all-MiniLM-L6-v2 encoder (Inter-Cascade config).

    Imported lazily so unit tests (which inject a deterministic fake) and the
    light CI never load torch / download the model.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return lambda texts: model.encode(list(texts), normalize_embeddings=False)


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
    ap.add_argument(
        "--arms",
        default="llm_only,cache,full",
        help="csv subset of: llm_only,cache,full,nl_strategy,compile_once",
    )
    ap.add_argument(
        "--prefix-k",
        default="0.5",
        help="compile_once train prefix: int cases or 0<frac<1 of n",
    )
    ap.add_argument("--nl-tau", type=float, default=0.5, help="nl_strategy retrieval-coverage gate")
    ap.add_argument("--nl-top-k", type=int, default=2)
    ap.add_argument(
        "--suffix-scoring",
        action="store_true",
        help="score every arm on the compile_once test suffix only (auto-on with compile_once)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    known = {"llm_only", "cache", "full", "nl_strategy", "compile_once"}
    unknown = [a for a in selected if a not in known]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; choose from {sorted(known)}")

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
    print(f"  teacher={args.teacher} model={model} arms={selected}")

    # prefix_k: int cases or a 0<frac<1 fraction of n
    raw_k = float(args.prefix_k)
    prefix_k = int(round(raw_k * len(cases))) if 0 < raw_k < 1 else int(raw_k)
    # suffix scoring isolates the online-vs-offline contrast: when compile_once
    # is present every arm is scored on the same test suffix [prefix_k:].
    suffix = args.suffix_scoring or "compile_once" in selected
    score_from = prefix_k if suffix else 0
    if suffix:
        print(f"  suffix-scoring ON: scoring cases [{prefix_k}:{len(cases)}] for every arm")

    weak_metered = None
    embed = None
    if "nl_strategy" in selected:
        # no routing: the nl_strategy in-context model is the SAME as the frontier
        weak_model, weak_raw = _build_teacher(args.teacher, NL_STRATEGY_PROMPT_TEMPLATE)
        weak_metered = MeteredTeacher(weak_raw, PriceTable.default(), model=weak_model)
        embed = _load_embedder()

    def _dispatch(name: str) -> dict:
        if name == "llm_only":
            return _run_llm_only(bench, shared, score_from=score_from)
        if name == "cache":
            return _run_cache(bench, shared, atoms, score_from=score_from)
        if name == "full":
            return _run_full(
                bench,
                shared,
                atoms,
                consolidate_every=args.consolidate_every,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
                score_from=score_from,
            )
        if name == "compile_once":
            return _run_compile_once(
                bench,
                shared,
                atoms,
                prefix_k=prefix_k,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
            )
        if name == "nl_strategy":
            return _run_nl_strategy(
                bench,
                shared,
                weak_metered=weak_metered,
                embed=embed,
                guard=guard,
                tau=args.nl_tau,
                top_k=args.nl_top_k,
                score_from=score_from,
            )
        raise SystemExit(f"unhandled arm {name}")

    arms: list[dict] = []
    truncated = False
    try:
        for name in selected:
            print(f"\narm {name} ...")
            arms.append(_dispatch(name))
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
    if "llm_only" in by:
        llm = by["llm_only"]
        amort = {}
        for name, a in by.items():
            if name == "llm_only":
                continue
            amort[f"amortisation_{name}_vs_llm"] = (
                round(llm["total_cost_usd"] / a["total_cost_usd"], 3)
                if a["total_cost_usd"]
                else None
            )
        verdict = {
            **amort,
            "calls": {name: a["teacher_calls"] for name, a in by.items()},
            "verdict_acc": {name: a["verdict_acc"] for name, a in by.items()},
            "article_f1": {name: a["article_micro_f1"] for name, a in by.items()},
            "weak_llm_cost_usd": by.get("nl_strategy", {}).get("weak_llm_cost_usd"),
            "prefix_k": prefix_k if "compile_once" in by else None,
            "suffix_scoring": suffix,
        }
        print("\nVERDICT: " + "; ".join(f"{k}={v}" for k, v in amort.items()))
        print(f"  calls={verdict['calls']}")

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
                "arms_run": selected,
                "distinct_patterns": distinct,
                "consolidate_every": args.consolidate_every,
                "min_support": args.min_support,
                "min_confidence": args.min_confidence,
                "suffix_scoring": suffix,
                "score_from": score_from,
                "prefix_k": prefix_k if "compile_once" in selected else None,
                "nl_tau": args.nl_tau if "nl_strategy" in selected else None,
                "weak_model": (weak_metered.model if weak_metered else None),
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
