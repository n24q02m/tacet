"""Temporal extension — bi-temporal edges, time-sliced reasoning, Allen relations.

Edges carry two optional time-points in their `props`:

* ``valid_from`` — when the fact began to hold (``None`` ⇒ -∞);
* ``valid_to``   — when the fact ceased to hold (``None`` ⇒ +∞, half-open).

The graph's ``slice_at(t)`` / ``slice_between(t1, t2)`` views return a filtered
``WorldGraph`` containing only edges whose validity interval covers / overlaps
the queried time, which the regular symbolic engine then reasons over without
modification. This is the standard validity-time database model — simple,
composable, and orthogonal to the rules.

``TemporalEngine`` is the convenience wrapper used by the demos and tests:
it owns an ontology + rule set, slices the graph for each query, and
materialises the closure on the slice. Allen's thirteen interval relations
(Allen, 1983) are provided as predicates over edge props for downstream
rule authors who want to reason about events.

Note: this is a *point-in-time* reasoning model. Time-aware unification
(timestamps as rule variables) is a richer formalism left for Tier B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tacet.core.graph import WorldGraph, _edge_overlaps, _edge_valid_at
from tacet.core.ontology import Ontology
from tacet.core.symbolic import Rule, RuleEngine, SymbolicResult

Time = float
AllenRelation = Literal[
    "before",
    "meets",
    "overlaps",
    "starts",
    "during",
    "finishes",
    "equal",
    "finished_by",
    "contains",
    "started_by",
    "overlapped_by",
    "met_by",
    "after",
]


@dataclass
class TemporalQuery:
    head: str
    relation: str
    at: Time | None = None
    between: tuple[Time, Time] | None = None


class TemporalEngine:
    """Rule engine that answers queries at a point or over an interval."""

    def __init__(self, ontology: Ontology, rules: list[Rule] | None = None) -> None:
        self.ontology = ontology
        self.rules = list(rules or [])

    def query_at(self, graph: WorldGraph, head: str, relation: str, time: Time) -> SymbolicResult:
        """Answer (head, relation, ?) using only facts valid at `time`."""
        view = graph.slice_at(time)
        engine = RuleEngine(self.ontology, self.rules)
        engine.materialise(view)
        return engine.query(head, relation)

    def query_history(
        self, graph: WorldGraph, head: str, relation: str, times: list[Time]
    ) -> dict[Time, list[str]]:
        """Trajectory of the answer set across a sequence of timepoints."""
        out: dict[Time, list[str]] = {}
        for t in times:
            r = self.query_at(graph, head, relation, t)
            out[t] = list(r.answers) if r.answered else []
        return out

    def query_during(
        self, graph: WorldGraph, head: str, relation: str, start: Time, end: Time
    ) -> SymbolicResult:
        """Union of (head, relation, ?) answers across any time in `[start, end)`.

        Used for ``which X did head <relation> at any point in this window?``
        questions — e.g.\\ which companies someone has ever worked at.
        """
        view = graph.slice_between(start, end)
        engine = RuleEngine(self.ontology, self.rules)
        engine.materialise(view)
        return engine.query(head, relation)


# --- helpers for edge construction --------------------------------------------
def temporal_edge(
    source: str,
    relation: str,
    target: str,
    valid_from: Time | None = None,
    valid_to: Time | None = None,
    **extra: object,
) -> dict:
    """Build a property dict for a bi-temporal edge — convenience for ``add_edge``."""
    return {"valid_from": valid_from, "valid_to": valid_to, **extra}


# --- Allen's interval algebra (Allen, 1983) ----------------------------------
def allen_relation(a: dict, b: dict) -> AllenRelation:
    """Classify the temporal relation between two edges' validity intervals.

    Open endpoints (``None``) are treated as ±∞. Edges with no temporal
    annotation are considered ``equal`` to each other and ``contains``
    every other interval.
    """
    a_start = -float("inf") if a.get("valid_from") is None else a["valid_from"]
    a_end = float("inf") if a.get("valid_to") is None else a["valid_to"]
    b_start = -float("inf") if b.get("valid_from") is None else b["valid_from"]
    b_end = float("inf") if b.get("valid_to") is None else b["valid_to"]
    if a_end < b_start:
        return "before"
    if a_end == b_start:
        return "meets"
    if a_start == b_start and a_end == b_end:
        return "equal"
    if a_start == b_start and a_end < b_end:
        return "starts"
    if a_start == b_start and a_end > b_end:
        return "started_by"
    if a_end == b_end and a_start > b_start:
        return "finishes"
    if a_end == b_end and a_start < b_start:
        return "finished_by"
    if a_start > b_start and a_end < b_end:
        return "during"
    if a_start < b_start and a_end > b_end:
        return "contains"
    if a_start < b_start < a_end < b_end:
        return "overlaps"
    if b_start < a_start < b_end < a_end:
        return "overlapped_by"
    if a_start == b_end:
        return "met_by"
    return "after"


__all__ = [
    "AllenRelation",
    "TemporalEngine",
    "TemporalQuery",
    "Time",
    "allen_relation",
    "temporal_edge",
    "_edge_overlaps",  # re-export for callers who want low-level filters
    "_edge_valid_at",
]
