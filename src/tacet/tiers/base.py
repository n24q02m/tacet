"""Core Tier protocol and TierResult value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TierResult:
    """Output returned by a cascade tier when it answers a query."""

    tier: int
    answers: list[str]
    confidence: float
    proof: list[str] = field(default_factory=list)
    grounded: bool = False
    note: str = ""


@runtime_checkable
class Tier(Protocol):
    """Protocol implemented by every pluggable cascade tier."""

    tier_id: int
    name: str

    def can_handle(self, head: str, relation: str) -> bool:
        """Return True if this tier can attempt `(head, relation)`."""
        ...

    def answer(self, head: str, relation: str) -> TierResult | None:
        """Attempt to answer `(head, relation)`, returning None to abstain."""
        ...
