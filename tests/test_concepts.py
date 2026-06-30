"""Tests for concept induction (distill/concepts.py)."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.distill.concepts import (
    induce_node_types,
    induce_relations,
    revise_ontology,
)


class TestConcepts(unittest.TestCase):
    def test_induce_node_types_empty(self) -> None:
        g = WorldGraph()
        types = induce_node_types(g)
        self.assertEqual(types, [])

    def test_induce_node_types_no_relations(self) -> None:
        g = WorldGraph()
        g.add_node("a")
        g.add_node("b")
        types = induce_node_types(g)
        self.assertEqual(len(types), 1)
        self.assertEqual(set(types[0].members), {"a", "b"})
        self.assertEqual(types[0].signature, {})

    def test_induce_node_types_clusters(self) -> None:
        # Create a graph with two distinct structural groups
        g = WorldGraph()
        # Group 1: Sources (only outgoing edges)
        for i in range(5):
            g.add_edge(f"src_{i}", "rel", f"mid_{i}")
        # Group 2: Sinks (only incoming edges)
        for i in range(5):
            g.add_edge(f"mid2_{i}", "rel", f"sink_{i}")

        # Note: mid_i and mid2_i nodes are also created automatically.
        # src_* have only (rel, out)
        # sink_* have only (rel, in)
        # mid_* have only (rel, in)
        # mid2_* have only (rel, out)

        types = induce_node_types(g, k=2)
        self.assertEqual(len(types), 2)

        # Verify describe() doesn't crash
        for t in types:
            self.assertIsInstance(t.describe(), str)

    def test_induce_relations_found(self) -> None:
        g = WorldGraph()
        # Create 10 instances of A -> r1 -> B -> r2 -> C
        for i in range(10):
            g.add_edge(f"x_{i}", "r1", f"z_{i}")
            g.add_edge(f"z_{i}", "r2", f"y_{i}")

        proposals = induce_relations(g, min_support=5)
        self.assertGreaterEqual(len(proposals), 1)

        names = [p.name for p in proposals]
        self.assertIn("r1+r2", names)

        p = next(p for p in proposals if p.name == "r1+r2")
        self.assertEqual(p.support, 10)
        self.assertEqual(p.body, (("?x", "r1", "?z"), ("?z", "r2", "?y")))

    def test_induce_relations_none(self) -> None:
        g = WorldGraph()
        g.add_edge("a", "r1", "b")
        g.add_edge("c", "r2", "d")
        proposals = induce_relations(g, min_support=1)
        # No length-2 paths (x->z->y)
        self.assertEqual(proposals, [])

    def test_revise_ontology(self) -> None:
        g = WorldGraph()
        # Add some structure for types and relations
        for i in range(10):
            g.add_edge(f"x_{i}", "r1", f"z_{i}")
            g.add_edge(f"z_{i}", "r2", f"y_{i}")

        onto = Ontology()
        added = revise_ontology(g, onto, min_support=5)

        self.assertGreater(len(added["added_types"]), 0)
        self.assertIn("r1+r2", added["added_relations"])

        self.assertIn("r1+r2", onto.relation_types)
        for tname in added["added_types"]:
            self.assertIn(tname, onto.node_types)


if __name__ == "__main__":
    unittest.main()
