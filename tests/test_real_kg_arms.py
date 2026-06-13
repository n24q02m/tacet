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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_real_kg_amortization import BudgetGuard  # noqa: E402
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


if __name__ == "__main__":
    import unittest

    unittest.main()
