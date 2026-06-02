"""Atomic event types for Layer 4 forward dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    """Atomic state-change action in the world graph."""

    timestamp: float
    type: str
    actor: str
    target: str | None = None
    payload: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class EventBatch:
    """All events that occur within one tick (e.g. one day in ICEWS14)."""

    timestamp: float
    events: list[Event]
