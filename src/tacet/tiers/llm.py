"""Tier 3 implementation wrapping an LLM `Teacher`."""

from __future__ import annotations

from tacet.core.graph import WorldGraph
from tacet.llm.teacher import Teacher
from tacet.tiers.base import TierResult


class LLMTier:
    """Tier 3: always-available LLM teacher fallback."""

    tier_id: int = 3
    name: str = "llm"

    def __init__(self, teacher: Teacher, graph: WorldGraph) -> None:
        self.teacher = teacher
        self.graph = graph

    def can_handle(self, head: str, relation: str) -> bool:
        return True

    def answer(self, head: str, relation: str) -> TierResult | None:
        resp = self.teacher.answer(self.graph, head, relation)
        return TierResult(
            tier=self.tier_id,
            answers=list(resp.answers),
            confidence=1.0,
            proof=[],
            grounded=False,
            note="answered by LLM teacher",
        )
