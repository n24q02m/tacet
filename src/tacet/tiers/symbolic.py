"""Tier 1 implementation wrapping `RuleEngine`."""

from __future__ import annotations

from tacet.core.symbolic import RuleEngine
from tacet.tiers.base import TierResult


class SymbolicTier:
    """Tier 1: sound, explainable Datalog closure lookup."""

    tier_id: int = 1
    name: str = "symbolic"

    def __init__(self, engine: RuleEngine) -> None:
        self.engine = engine

    def can_handle(self, head: str, relation: str) -> bool:
        return True

    def answer(self, head: str, relation: str) -> TierResult | None:
        sym = self.engine.query(head, relation)
        if not sym.answered:
            return None
        return TierResult(
            tier=self.tier_id,
            answers=list(sym.answers),
            confidence=1.0,
            proof=list(sym.proof),
            grounded=True,
            note="entailed by rules",
        )
