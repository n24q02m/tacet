"""Unseen-entity, new-relation, and noisy-teacher evaluation benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.core.symbolic import Rule, RuleEngine
from tacet.eval.benchmark import Benchmark, BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher

UnseenVariant = Literal["unseen_entity", "new_relation", "noisy_teacher"]
Query = tuple[str, str]


@dataclass(frozen=True)
class UnseenBenchmarkConfig:
    """Configuration for generating an unseen-generalisation benchmark."""

    seed: int = 0
    holdout_entity_fraction: float = 0.25
    noisy_teacher_error_rate: float = 0.15


@dataclass
class UnseenBenchmark:
    """Split KG benchmark isolating unseen-entity, new-relation, or noisy-teacher regimes."""

    variant: str
    train_graph: WorldGraph
    eval_graph: WorldGraph
    ontology: Ontology
    given_rules: list[Rule]
    train_entities: frozenset[str]
    eval_entities: frozenset[str]
    workload: list[Query]
    truth: dict[Query, list[str]]
    oracle: dict[Query, list[str]]
    tier1_abstain_rate: float
    headroom: float
    noisy_teacher_accuracy: float | None = None


def _copy_filtered_graph(
    graph: WorldGraph,
    *,
    excluded_entities: frozenset[str] = frozenset(),
    excluded_relations: frozenset[str] = frozenset(),
    name_suffix: str = "train",
) -> WorldGraph:
    out = WorldGraph(name=f"{graph.name}-{name_suffix}")
    for n in graph.nodes:
        if n.id not in excluded_entities:
            out.add_node(n.id, n.type, **n.props)
    for e in graph.edges:
        if (
            e.source not in excluded_entities
            and e.target not in excluded_entities
            and e.relation not in excluded_relations
        ):
            out.add_edge(e.source, e.relation, e.target, **e.props)
    return out


def _evaluate_tier1_headroom(
    train_graph: WorldGraph,
    ontology: Ontology,
    given_rules: list[Rule],
    workload: list[Query],
    truth: dict[Query, list[str]],
    oracle: dict[Query, list[str]],
) -> tuple[float, float]:
    if not workload:
        return 0.0, 0.0
    engine = RuleEngine(ontology, list(given_rules))
    engine.materialise(train_graph)
    abstained = 0
    t1_correct = 0
    oracle_correct = 0
    for h, r in workload:
        sym = engine.query(h, r)
        if not sym.answered:
            abstained += 1
        elif set(sym.answers) == set(truth[(h, r)]):
            t1_correct += 1
        if set(oracle[(h, r)]) == set(truth[(h, r)]):
            oracle_correct += 1
    abstain_rate = abstained / len(workload)
    headroom = max(0.0, (oracle_correct - t1_correct) / len(workload))
    return abstain_rate, headroom


def _build_unseen_entity(base: Benchmark, cfg: UnseenBenchmarkConfig) -> UnseenBenchmark:
    rng = random.Random(cfg.seed)
    people = sorted(base.entity_pool)
    n_holdout = max(1, min(len(people) - 1, round(len(people) * cfg.holdout_entity_fraction)))
    eval_entities = frozenset(rng.sample(people, n_holdout))
    train_entities = frozenset(p for p in people if p not in eval_entities)

    train_graph = _copy_filtered_graph(
        base.graph, excluded_entities=eval_entities, name_suffix="train"
    )
    eval_graph = base.graph.copy()
    given_rules = list(base.given_rules)

    # Re-compute derived closure on train_graph so train truth never leaks eval entities.
    train_engine = RuleEngine(base.ontology, given_rules)
    train_engine.materialise(train_graph)
    truth: dict[Query, list[str]] = {}
    for h, r, t in train_engine.closure:
        if h in train_entities and t not in eval_entities:
            truth.setdefault((h, r), []).append(t)
    for key in truth:
        truth[key] = sorted(set(truth[key]))

    target_relations = (
        "works_at",
        "lives_in",
        "member_of",
        "ancestor_of",
        "superior_of",
        "colleague_of",
        "primary_language",
    )
    workload: list[Query] = []
    for p in sorted(eval_entities):
        for rel in target_relations:
            ans = list(base.truth.get((p, rel), ()))
            if ans:
                workload.append((p, rel))
                truth[(p, rel)] = sorted(ans)

    oracle = {q: list(truth[q]) for q in workload}
    abstain_rate, headroom = _evaluate_tier1_headroom(
        train_graph, base.ontology, given_rules, workload, truth, oracle
    )
    return UnseenBenchmark(
        variant="unseen_entity",
        train_graph=train_graph,
        eval_graph=eval_graph,
        ontology=base.ontology,
        given_rules=given_rules,
        train_entities=train_entities,
        eval_entities=eval_entities,
        workload=workload,
        truth=truth,
        oracle=oracle,
        tier1_abstain_rate=abstain_rate,
        headroom=headroom,
        noisy_teacher_accuracy=None,
    )


def _build_new_relation(base: Benchmark) -> UnseenBenchmark:
    withheld = frozenset({"superior_of"})
    train_graph = _copy_filtered_graph(
        base.graph, excluded_relations=withheld, name_suffix="new-rel"
    )
    eval_graph = base.graph.copy()
    given_rules = [r for r in base.given_rules if r.head[1] not in withheld]
    train_entities = frozenset(base.entity_pool)
    eval_entities: frozenset[str] = frozenset()

    truth: dict[Query, list[str]] = {}
    workload: list[Query] = []
    for p in sorted(base.entity_pool):
        ans = list(base.truth.get((p, "superior_of"), ()))
        if ans:
            workload.append((p, "superior_of"))
            truth[(p, "superior_of")] = sorted(ans)

    oracle = {q: list(truth[q]) for q in workload}
    abstain_rate, headroom = _evaluate_tier1_headroom(
        train_graph, base.ontology, given_rules, workload, truth, oracle
    )
    return UnseenBenchmark(
        variant="new_relation",
        train_graph=train_graph,
        eval_graph=eval_graph,
        ontology=base.ontology,
        given_rules=given_rules,
        train_entities=train_entities,
        eval_entities=eval_entities,
        workload=workload,
        truth=truth,
        oracle=oracle,
        tier1_abstain_rate=abstain_rate,
        headroom=headroom,
        noisy_teacher_accuracy=None,
    )


def _build_noisy_teacher(base: Benchmark, cfg: UnseenBenchmarkConfig) -> UnseenBenchmark:
    withheld = frozenset({"superior_of", "colleague_of"})
    train_graph = _copy_filtered_graph(base.graph, excluded_relations=withheld, name_suffix="noisy")
    eval_graph = base.graph.copy()
    given_rules = [r for r in base.given_rules if r.head[1] not in withheld]
    train_entities = frozenset(base.entity_pool)
    eval_entities: frozenset[str] = frozenset()

    truth: dict[Query, list[str]] = {}
    workload: list[Query] = []
    for p in sorted(base.entity_pool):
        for rel in ("superior_of", "colleague_of"):
            ans = list(base.truth.get((p, rel), ()))
            if ans:
                workload.append((p, rel))
                truth[(p, rel)] = sorted(ans)

    teacher = OracleTeacher(
        oracle=base.oracle,
        error_rate=cfg.noisy_teacher_error_rate,
        entity_pool=list(base.entity_pool),
        seed=cfg.seed,
        noise_mode="per_key",
    )
    oracle: dict[Query, list[str]] = {}
    teacher_hits = 0
    for h, r in workload:
        resp = teacher.answer(eval_graph, h, r)
        oracle[(h, r)] = sorted(resp.answers)
        if set(resp.answers) == set(truth[(h, r)]):
            teacher_hits += 1
    noisy_acc = teacher_hits / len(workload) if workload else 0.0

    abstain_rate, headroom = _evaluate_tier1_headroom(
        train_graph, base.ontology, given_rules, workload, truth, oracle
    )
    return UnseenBenchmark(
        variant="noisy_teacher",
        train_graph=train_graph,
        eval_graph=eval_graph,
        ontology=base.ontology,
        given_rules=given_rules,
        train_entities=train_entities,
        eval_entities=eval_entities,
        workload=workload,
        truth=truth,
        oracle=oracle,
        tier1_abstain_rate=abstain_rate,
        headroom=headroom,
        noisy_teacher_accuracy=noisy_acc,
    )


def generate_unseen_benchmark(
    variant: UnseenVariant,
    config: UnseenBenchmarkConfig | None = None,
) -> UnseenBenchmark:
    """Generate a deterministic unseen-entity, new-relation, or noisy-teacher benchmark."""
    if variant not in {"unseen_entity", "new_relation", "noisy_teacher"}:
        raise ValueError(
            f"variant must be 'unseen_entity', 'new_relation', or 'noisy_teacher', got {variant!r}"
        )
    cfg = config or UnseenBenchmarkConfig()
    if not (0.0 < cfg.holdout_entity_fraction < 1.0):
        raise ValueError(
            f"holdout_entity_fraction must be in (0, 1), got {cfg.holdout_entity_fraction!r}"
        )
    if not (0.0 <= cfg.noisy_teacher_error_rate <= 1.0):
        raise ValueError(
            f"noisy_teacher_error_rate must be in [0, 1], got {cfg.noisy_teacher_error_rate!r}"
        )

    base = generate(BenchmarkConfig(seed=cfg.seed))
    if variant == "unseen_entity":
        return _build_unseen_entity(base, cfg)
    if variant == "new_relation":
        return _build_new_relation(base)
    return _build_noisy_teacher(base, cfg)
