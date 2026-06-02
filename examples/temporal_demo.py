#!/usr/bin/env python3
"""Temporal TACET — employment history with bi-temporal edges.

    python examples/temporal_demo.py

Builds a small KG with three jobs over a decade, then queries the
same head/relation at different timepoints to recover the trajectory.
"""

from __future__ import annotations

from tacet import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.temporal import TemporalEngine

g = WorldGraph(name="careers")
for who in ("alice",):
    g.add_node(who, "Person")
for co in ("acme", "zoot", "ever"):
    g.add_node(co, "Company")
g.add_edge("alice", "works_at", "acme", valid_from=2016.0, valid_to=2019.0)
g.add_edge("alice", "works_at", "zoot", valid_from=2019.0, valid_to=2024.0)
g.add_edge("alice", "works_at", "ever", valid_from=2024.0)  # open-ended

onto = Ontology()
onto.add_node_type(NodeType("Person"))
onto.add_node_type(NodeType("Company"))
onto.add_relation_type(
    RelationType("works_at", frozenset({"Person"}), frozenset({"Company"}), functional=True)
)

engine = TemporalEngine(onto, [])

print("Alice's employer at successive years:")
for year in (2015, 2017, 2020, 2024, 2030):
    answer = engine.query_at(g, "alice", "works_at", year).answers
    print(f"  {year}: {answer or '(none)'}")

print("\nUnique employers across 2018–2026:")
print(" ", set(engine.query_during(g, "alice", "works_at", 2018, 2026).answers))

print("\nFull trajectory:")
for t, who in engine.query_history(g, "alice", "works_at", list(range(2015, 2031))).items():
    print(f"  {t}: {who or '(none)'}")
