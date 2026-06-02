"""Layer 4 forward dynamics — Trajectory storage and event application."""

from __future__ import annotations

from dataclasses import dataclass

from tacet.core.graph import WorldGraph
from tacet.experimental.dynamics.events import Event, EventBatch


@dataclass
class Trajectory:
    """List of (G_t, EventBatch_t, G_{t+1}) tuples in chronological order."""

    snapshots: list[WorldGraph]
    event_batches: list[EventBatch]

    def __len__(self) -> int:
        return len(self.event_batches)

    def at(self, t: int) -> WorldGraph:
        return self.snapshots[t]

    def event_at(self, t: int) -> EventBatch:
        return self.event_batches[t]

    def transition(self, t: int) -> tuple[WorldGraph, EventBatch, WorldGraph]:
        return self.snapshots[t], self.event_batches[t], self.snapshots[t + 1]


def apply_event(graph: WorldGraph, event: Event) -> WorldGraph:
    """Deterministic state transition: add the event's edge to a copy of graph."""
    g = graph.copy()
    target = event.target if event.target is not None else event.actor
    g.add_edge(event.actor, event.type, target, valid_from=event.timestamp, valid_to=None)
    return g
