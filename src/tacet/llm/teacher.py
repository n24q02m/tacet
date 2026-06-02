"""Tier 3 — the LLM teacher and the natural-language narrator.

The teacher is the expensive, always-available fallback. In a real deployment
it is a frontier LLM (Gemini / Grok via a multi-provider fallback chain);
its answers seed the distillation loop that teaches the cheaper tiers.

To keep every experiment in this repository **reproducible and offline**, the
teacher is modelled as an oracle over ground-truth world knowledge, optionally
*noisy* (`error_rate`) so the effect of an imperfect teacher can be measured.
`CallableTeacher` is the integration point for a real LLM — wrap any
`(head, relation) -> list[str]` callable and the rest of the system is
unchanged.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph


@dataclass
class TeacherResponse:
    answers: list[str]
    cost: float = 1.0
    correct: bool = True  # bookkeeping for noisy-teacher experiments


class Teacher:
    """Abstract teacher. A teacher answers any (head, relation) tail query."""

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        raise NotImplementedError


class OracleTeacher(Teacher):
    """A ground-truth oracle, optionally noisy — the reproducible LLM stand-in.

    Parameters
    ----------
    oracle:
        `oracle(head, relation) -> list[tails]`, the true answer set.
    error_rate:
        Probability the teacher returns a corrupted answer (a wrong entity),
        modelling LLM hallucination. 0.0 = perfect teacher.
    entity_pool:
        Entities used to fabricate a wrong answer when an error is injected.
    """

    def __init__(
        self,
        oracle: Callable[[str, str], list[str]],
        error_rate: float = 0.0,
        entity_pool: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        self._oracle = oracle
        self.error_rate = error_rate
        self._pool = entity_pool or []
        self._rng = random.Random(seed)

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        truth = list(self._oracle(head, relation))
        if self.error_rate > 0 and self._pool and self._rng.random() < self.error_rate:
            wrong = self._rng.choice(self._pool)
            return TeacherResponse(answers=[wrong], cost=1.0, correct=(wrong in truth))
        return TeacherResponse(answers=truth, cost=1.0, correct=True)


class CallableTeacher(Teacher):
    """Wraps an arbitrary callable — the hook for a real LLM backend.

    Example (production)::

        from google import genai
        client = genai.Client()

        def gemini(head, relation):
            prompt = f"List the tails of ({head}, {relation}) ..."
            ...  # call client, parse a JSON list
            return parsed

        teacher = CallableTeacher(gemini)
    """

    def __init__(self, fn: Callable[[str, str], list[str]], cost: float = 1.0) -> None:
        self._fn = fn
        self._cost = cost

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        return TeacherResponse(answers=list(self._fn(head, relation)), cost=self._cost)


@dataclass
class Narrator:
    """Deterministic graph-reasoning -> natural-language renderer.

    In production this is a *small* LLM call (the cheap "narrator" use of an
    LLM); offline it is a template so results stay reproducible.
    """

    tier_label: dict[int, str] = field(
        default_factory=lambda: {
            1: "verified rules",
            2: "structural inference",
            3: "expert reasoning",
        }
    )

    def render(
        self,
        head: str,
        relation: str,
        answers: list[str],
        tier: int,
        proof: list[str] | None = None,
    ) -> str:
        verb = relation.replace("_", " ")
        if not answers:
            return f"No answer found for ({head}, {verb})."
        text = f"{head} — {verb} → {', '.join(answers)}  [via {self.tier_label[tier]}]"
        if proof and tier == 1:
            text += "\n  proof:\n" + "\n".join("    " + ln for ln in proof)
        return text
