"""Regression test for the ontology-preservation invariant under write-back.

Theorem 1 (ontology-preservation) holds *under the premise* that the graph
stays ontology-consistent. The distillation loop writes teacher facts back into
the graph; if a written-back fact introduces a brand-new endpoint, that endpoint
must be typed from the relation's declared schema, otherwise it defaults to the
catch-all ``Entity`` type and a perfectly valid rule then derives a
type-violating fact into the closure (breaking the premise the system is
supposed to maintain by construction).

This test pins that the write-back path types new endpoints from the relation
schema, so a query that triggers a teacher write-back of a fresh entity leaves
the graph ontology-consistent.
"""

from __future__ import annotations

import unittest

from tacet.cascade.router import TACET
from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule
from tacet.llm.teacher import CallableTeacher
from tacet.serve.config import CascadeConfig, KGEConfig


class TestWriteBackOntologyPreservation(unittest.TestCase):
    def _setup(self) -> tuple[TACET, WorldGraph, Ontology]:
        person = frozenset({"Person"})
        onto = (
            Ontology()
            .add_node_type(NodeType("Person"))
            .add_relation_type(RelationType("parent_of", person, person))
            .add_relation_type(RelationType("ancestor_of", person, person))
        )
        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("bob", "Person")
        g.add_edge("alice", "parent_of", "bob")
        # a valid (subset-checked) rule: ancestor_of(x, y) <= parent_of(x, y)
        rule = Rule(name="anc", body=(("?x", "parent_of", "?y"),), head=("?x", "ancestor_of", "?y"))
        # teacher answers (bob, parent_of) with a BRAND-NEW entity
        teacher = CallableTeacher(
            lambda h, r: ["mystery_x"] if (h, r) == ("bob", "parent_of") else []
        )
        ak = TACET(
            g,
            onto,
            teacher,
            rules=[rule],
            config=CascadeConfig(
                kge=KGEConfig(epochs=1),
                distillation=True,
                write_back=True,
                rule_synthesis=False,
            ),
        )
        return ak, g, onto

    def test_new_writeback_endpoint_is_typed_from_relation_schema(self) -> None:
        ak, g, _ = self._setup()
        # Tier-1 abstains on (bob, parent_of); the teacher answers and the fact
        # (bob, parent_of, mystery_x) is written back.
        ak.ask("bob", "parent_of")
        node = g.node("mystery_x")
        self.assertIsNotNone(node, "write-back must create the new endpoint")
        self.assertEqual(
            node.type,
            "Person",
            "new endpoint must inherit the relation's declared range type, not 'Entity'",
        )

    def test_writeback_keeps_graph_ontology_consistent(self) -> None:
        ak, g, onto = self._setup()
        ak.ask("bob", "parent_of")
        self.assertEqual(
            onto.validate(g),
            [],
            "teacher write-back must not introduce a type-violating fact",
        )

    def test_closure_stays_ontology_consistent_after_rule_fires(self) -> None:
        ak, g, onto = self._setup()
        ak.ask("bob", "parent_of")
        # materialise so the given rule fires on the written-back fact
        ak.engine.materialise(g)
        self.assertEqual(onto.validate(g), [])


if __name__ == "__main__":
    unittest.main()
