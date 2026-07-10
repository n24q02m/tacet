"""Fairness-invariant tests for the real-KG controlled runner.

The MetaQA amortisation result (2.8x / 5.1x) relies on every arm seeing the
*same* teacher answer per (head, relation) pair, so the arms differ only in
routing and not in LLM stochasticity. These tests pin that invariant for
``SharedAnswerCache`` / ``ReplayTeacher`` the way ``test_privaci_arms.py`` does
for the compliance runner, so a regression cannot silently turn a baseline into
a strawman (or double-charge the shared budget).

No network, no real LLM: a FakeMetered stands in for the metered Grok teacher.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_real_kg_amortization import BudgetGuard, _new_metered  # noqa: E402
from run_real_kg_controlled import (  # noqa: E402
    ReplayTeacher,
    SharedAnswerCache,
    _replay_llm_only,
)

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.llm.teacher import TeacherResponse  # noqa: E402


class FakeMetered:
    """Stands in for the metered Grok teacher: scripted answer + cost per pair."""

    def __init__(self, table: dict[tuple[str, str], tuple[list[str], float]]) -> None:
        self.table = table
        self.calls = 0
        self.last_cost_usd = 0.0

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        self.calls += 1
        ans, cost = self.table[(head, relation)]
        self.last_cost_usd = cost
        return TeacherResponse(answers=list(ans))


def test_shared_cache_one_real_call_per_pair() -> None:
    kg = WorldGraph()
    metered = FakeMetered(
        {("alice", "works_at"): (["acme"], 0.05), ("bob", "lives_in"): (["paris"], 0.03)}
    )
    guard = BudgetGuard(budget_usd=100.0)
    shared = SharedAnswerCache(metered, guard, kg)

    a1, c1 = shared.answer("alice", "works_at")
    assert (a1, c1) == (["acme"], 0.05)
    # a repeat of the same pair must NOT trigger a second real call, and must
    # return the identical answer and cost
    a2, c2 = shared.answer("alice", "works_at")
    assert (a2, c2) == (["acme"], 0.05)
    assert metered.calls == 1
    assert shared.real_calls == 1
    assert guard.spent_usd == 0.05  # charged exactly once

    # a distinct pair is a second real call
    shared.answer("bob", "lives_in")
    assert metered.calls == 2
    assert shared.real_calls == 2
    assert abs(guard.spent_usd - 0.08) < 1e-9


def test_replay_teachers_share_answers_across_arms() -> None:
    kg = WorldGraph()
    metered = FakeMetered({("alice", "works_at"): (["acme"], 0.05)})
    guard = BudgetGuard(budget_usd=100.0)
    shared = SharedAnswerCache(metered, guard, kg)

    arm_a = ReplayTeacher(shared)
    arm_b = ReplayTeacher(shared)
    ra = arm_a.answer(kg, "alice", "works_at")
    rb = arm_b.answer(kg, "alice", "works_at")

    # both arms see the identical teacher answer for the pair
    assert ra.answers == rb.answers == ["acme"]
    # only the first arm to ask paid the real call
    assert metered.calls == 1
    # each arm attributes the per-pair cost to itself
    assert arm_a.total_cost == arm_b.total_cost == 0.05
    assert arm_a.n_calls == arm_b.n_calls == 1


def test_replay_llm_only_sends_every_query_but_pays_per_distinct_pair() -> None:
    kg = WorldGraph()
    metered = FakeMetered({("a", "r"): (["x"], 0.05), ("b", "r"): (["y"], 0.05)})
    guard = BudgetGuard(budget_usd=100.0)
    shared = SharedAnswerCache(metered, guard, kg)

    # the third query repeats the first pair
    stream = [
        ("a", "r", frozenset({"x"})),
        ("b", "r", frozenset({"y"})),
        ("a", "r", frozenset({"x"})),
    ]
    rep = _replay_llm_only(stream, shared)

    assert rep["n"] == 3
    assert rep["teacher_calls"] == 3  # llm_only escalates every query
    assert metered.calls == 2  # but only two distinct pairs hit the real teacher
    assert rep["accuracy"] == 1.0


# ----------------------------------------------- engine-hit parity (scored-vs-gold)
class _FakeBench:
    """Minimal bench exposing only ``.kg`` (all ``_kg_without`` touches)."""

    def __init__(self, kg: WorldGraph) -> None:
        self.kg = kg


def test_engine_hit_is_free_but_still_scored_against_gold() -> None:
    """A cascade query served by the engine (Tier-1, cost 0) is scored against
    gold exactly like a paid teacher answer -- it gets no accuracy free pass.

    The teacher caches a WRONG answer on the first query; the repeat is served
    for free by the engine. If the free hit were excluded from scoring, accuracy
    would be 0.5 (only the paid query counted) or 1.0; the parity invariant
    requires 0.0 (both the paid and the free hit are wrong against gold).
    """
    from run_real_kg_controlled import TIER2_OFF, _replay_cascade

    from tacet.core.ontology import NodeType, Ontology, RelationType
    from tacet.serve.config import CascadeConfig, KGEConfig

    kg = WorldGraph()
    kg.add_node("alice", "Person")
    kg.add_node("acme", "Company")
    kg.add_node("wrong_co", "Company")
    kg.add_node("bob", "Person")
    kg.add_edge("bob", "works_at", "acme")  # an un-held-out edge so the KGE has data
    bench = _FakeBench(kg)
    onto = (
        Ontology()
        .add_node_type(NodeType("Person"))
        .add_node_type(NodeType("Company"))
        .add_relation_type(RelationType("works_at", frozenset({"Person"}), frozenset({"Company"})))
    )
    metered = FakeMetered({("alice", "works_at"): (["wrong_co"], 0.05)})
    guard = BudgetGuard(budget_usd=100.0)
    shared = SharedAnswerCache(metered, guard, kg)
    gold = frozenset({"acme"})
    stream = [("alice", "works_at", gold), ("alice", "works_at", gold)]
    cfg = CascadeConfig(
        kge=KGEConfig(epochs=1),
        write_back=True,
        rule_synthesis=False,
        kge_augment=False,
        l2_threshold=TIER2_OFF,  # Tier-2 off: first query must reach the teacher
    )
    rep = _replay_cascade("cache_cascade", stream, bench, onto, shared, cfg)

    # the first query paid the teacher (Tier-3); the repeat was served free (Tier-1)
    assert rep["tier_counts"][3] == 1
    assert rep["tier_counts"][1] == 1
    assert metered.calls == 1
    assert rep["total_cost_usd"] == 0.05  # the free engine hit added no cost
    # parity: the free hit is scored against gold, not waved through -> both wrong
    assert rep["accuracy"] == 0.0


# ----------------------------------------------- oracle noise dial (E11 sweep)
def test_new_metered_oracle_injects_workload_noise() -> None:
    """The oracle-teacher path must forward the noise dial + a workload-derived
    entity pool, so a nonzero error rate actually corrupts answers (the E11
    imperfect-teacher sweep). With the dial unwired (``error_rate`` defaulting to
    0.0) the oracle is perfect and this corruption cannot happen.
    """
    settings = SimpleNamespace(teacher="oracle")
    gold = {"alice\tstarred_in": frozenset({"m1", "m2", "m3"})}
    graph = WorldGraph()

    # error_rate=0.0 -> a perfect oracle returns the exact gold set.
    perfect = _new_metered(settings, "grok-4.3", None, gold, error_rate=0.0, seed=0)
    assert perfect.answer(graph, "alice", "starred_in").answers == ["m1", "m2", "m3"]

    # error_rate=1.0 -> every answer is a single corrupted entity drawn from the
    # workload's own gold tails (a plausible wrong entity), so it is NOT the full
    # gold set and it comes from the derived pool.
    noisy = _new_metered(settings, "grok-4.3", None, gold, error_rate=1.0, seed=0)
    corrupted = noisy.answer(graph, "alice", "starred_in").answers
    assert corrupted != ["m1", "m2", "m3"]
    assert len(corrupted) == 1
    assert corrupted[0] in {"m1", "m2", "m3"}


if __name__ == "__main__":
    import unittest

    unittest.main()
