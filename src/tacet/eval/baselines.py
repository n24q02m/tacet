"""Baseline systems and the unified run harness.

Every system processes the *same* streamed workload from a `Benchmark`; the
harness records the tier, cost, latency and correctness of each query so the
systems can be compared on a single cost-accuracy footing.

Systems
-------
* ``llm_only``       — every query goes to the LLM (the cost ceiling).
* ``symbolic_only``  — only the shipped rules; abstain otherwise (the accuracy
                       floor — measures pure rule coverage).
* ``cache_cascade``  — rules + an exact-match answer cache + LLM. The honest
                       "just memoise" baseline: it answers exact repeats for
                       free but never generalises to unseen heads.
* ``static_cascade`` — the full 3-tier cascade with distillation switched off.
* ``tacet``           — the full cascade with online distillation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tacet.cascade.router import TACET
from tacet.core.symbolic import RuleEngine
from tacet.eval.benchmark import Benchmark
from tacet.llm.teacher import Teacher
from tacet.serve.config import TIER_COST, TIER_LATENCY_MS, CascadeConfig


def answer_correct(answers: list[str], truth: list[str]) -> bool:
    """An answer is correct iff its set exactly matches the ground-truth set."""
    return set(answers) == set(truth)


@dataclass
class QueryRecord:
    idx: int
    head: str
    relation: str
    qclass: str
    tier: int  # 1/2/3; 0 = abstained
    correct: bool
    cost: float
    latency_ms: float


@dataclass
class RunResult:
    system: str
    records: list[QueryRecord] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.records) / self.n if self.n else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.records) / self.n if self.n else 0.0

    def tier_counts(self) -> dict[int, int]:
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for r in self.records:
            counts[r.tier] += 1
        return counts

    def cost_trajectory(self) -> list[float]:
        """Cumulative cost after each query."""
        traj, running = [], 0.0
        for r in self.records:
            running += r.cost
            traj.append(running)
        return traj

    def cost_by_class(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r.qclass] = out.get(r.qclass, 0.0) + r.cost
        return out

    def accuracy_by_class(self) -> dict[str, float]:
        hit: dict[str, int] = {}
        tot: dict[str, int] = {}
        for r in self.records:
            tot[r.qclass] = tot.get(r.qclass, 0) + 1
            hit[r.qclass] = hit.get(r.qclass, 0) + int(r.correct)
        return {c: hit[c] / tot[c] for c in tot}


# --------------------------------------------------------------------------
def _truth(bench: Benchmark, head: str, relation: str) -> list[str]:
    return bench.truth.get((head, relation), [])


def run_llm_only(
    bench: Benchmark, teacher: Teacher, cost: dict[int, float] | None = None
) -> RunResult:
    cost = cost or TIER_COST
    res = RunResult("llm_only")
    for idx, ((h, r), cls) in enumerate(zip(bench.workload, bench.classes, strict=True)):
        ans = teacher.answer(bench.graph, h, r).answers
        res.records.append(
            QueryRecord(
                idx,
                h,
                r,
                cls,
                3,
                answer_correct(ans, _truth(bench, h, r)),
                cost[3],
                TIER_LATENCY_MS[3],
            )
        )
    return res


def run_symbolic_only(bench: Benchmark, cost: dict[int, float] | None = None) -> RunResult:
    cost = cost or TIER_COST
    engine = RuleEngine(bench.ontology, list(bench.given_rules))
    engine.materialise(bench.graph)
    res = RunResult("symbolic_only")
    for idx, ((h, r), cls) in enumerate(zip(bench.workload, bench.classes, strict=True)):
        sym = engine.query(h, r)
        tier = 1 if sym.answered else 0
        ans = sym.answers if sym.answered else []
        res.records.append(
            QueryRecord(
                idx,
                h,
                r,
                cls,
                tier,
                answer_correct(ans, _truth(bench, h, r)),
                cost[1] if sym.answered else cost[1],
                TIER_LATENCY_MS[1],
            )
        )
    return res


def run_cache_cascade(
    bench: Benchmark, teacher: Teacher, cost: dict[int, float] | None = None
) -> RunResult:
    """Rules + exact-match cache + LLM. Memoises repeats; never generalises."""
    cost = cost or TIER_COST
    engine = RuleEngine(bench.ontology, list(bench.given_rules))
    engine.materialise(bench.graph)
    cache: dict[tuple[str, str], list[str]] = {}
    res = RunResult("cache_cascade")
    for idx, ((h, r), cls) in enumerate(zip(bench.workload, bench.classes, strict=True)):
        sym = engine.query(h, r)
        if sym.answered:
            ans, tier, c, lat = sym.answers, 1, cost[1], TIER_LATENCY_MS[1]
        elif (h, r) in cache:
            ans, tier, c, lat = cache[(h, r)], 1, cost[1], TIER_LATENCY_MS[1]
        else:
            ans = teacher.answer(bench.graph, h, r).answers
            cache[(h, r)] = ans
            tier, c, lat = 3, cost[3], TIER_LATENCY_MS[3]
        res.records.append(
            QueryRecord(idx, h, r, cls, tier, answer_correct(ans, _truth(bench, h, r)), c, lat)
        )
    return res


def run_cascade(
    bench: Benchmark,
    teacher: Teacher,
    config: CascadeConfig,
    consolidate_every: int = 50,
    system_name: str = "tacet",
) -> RunResult:
    """Run the full cascade; `config.distillation` toggles TACET vs static."""
    graph = bench.graph.copy()
    ak = TACET(graph, bench.ontology, teacher, rules=list(bench.given_rules), config=config)
    ak.warmup(calibration=bench.calibration)
    res = RunResult(system_name)
    for idx, ((h, r), cls) in enumerate(zip(bench.workload, bench.classes, strict=True)):
        ans = ak.ask(h, r)
        res.records.append(
            QueryRecord(
                idx,
                h,
                r,
                cls,
                ans.tier,
                answer_correct(ans.answers, _truth(bench, h, r)),
                ans.cost,
                ans.latency_ms,
            )
        )
        if config.distillation and consolidate_every > 0 and (idx + 1) % consolidate_every == 0:
            ak.consolidate()
    res.meta["synthesised_rules"] = list(ak.synthesised_rules)
    return res
