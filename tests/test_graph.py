"""Tests for the world graph and the typed ontology."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType


class TestWorldGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.g = WorldGraph.from_triples(
            [("a", "rel", "b"), ("b", "rel", "c")],
            types={"a": "T", "b": "T", "c": "T"},
        )

    def test_stats(self) -> None:
        s = self.g.stats()
        self.assertEqual(s["nodes"], 3)
        self.assertEqual(s["edges"], 2)

    def test_upsert_idempotent(self) -> None:
        self.g.add_node("a", "T", role="x")
        self.g.add_edge("a", "rel", "b")
        self.assertEqual(self.g.stats()["edges"], 2)
        self.assertEqual(self.g.node("a").props["role"], "x")

    def test_traversal_both_directions(self) -> None:
        self.assertEqual(self.g.out("a", "rel"), {"b"})
        self.assertEqual(self.g.into("b", "rel"), {"a"})

    def test_copy_is_independent(self) -> None:
        h = self.g.copy()
        h.add_edge("c", "rel", "a")
        self.assertEqual(self.g.stats()["edges"], 2)
        self.assertEqual(h.stats()["edges"], 3)

    def test_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.json"
            self.g.to_json(path)
            loaded = WorldGraph.from_json(path)
            self.assertEqual(set(loaded.triples()), set(self.g.triples()))

    def test_from_json_real_geography(self) -> None:
        data = Path(__file__).resolve().parents[1] / "src/tacet/data/worldgeo.json"
        g = WorldGraph.from_json(data)
        self.assertGreater(g.stats()["nodes"], 40)
        self.assertIn("France", g.entities())


class TestOntology(unittest.TestCase):
    def _onto(self) -> Ontology:
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        onto.add_node_type(NodeType("City"))
        onto.add_relation_type(
            RelationType("lives_in", frozenset({"Person"}), frozenset({"City"}), functional=True)
        )
        return onto

    def test_validate_clean_graph(self) -> None:
        g = WorldGraph()
        g.add_node("p", "Person")
        g.add_node("c", "City")
        g.add_edge("p", "lives_in", "c")
        self.assertEqual(self._onto().validate(g), [])

    def test_validate_detects_domain_range(self) -> None:
        g = WorldGraph()
        g.add_node("p", "Person")
        g.add_node("c", "City")
        g.add_edge("c", "lives_in", "p")  # City can't live_in a Person
        kinds = {v.kind for v in self._onto().validate(g)}
        self.assertIn("domain_range", kinds)

    def test_functional_gate_blocks_second_edge(self) -> None:
        g = WorldGraph()
        g.add_node("p", "Person")
        g.add_node("c1", "City")
        g.add_node("c2", "City")
        g.add_edge("p", "lives_in", "c1")
        onto = self._onto()
        self.assertFalse(onto.allows(g, "p", "lives_in", "c2"))

    def test_induce_detects_symmetric(self) -> None:
        g = WorldGraph()
        for a, b in [("p", "q"), ("q", "p"), ("q", "r"), ("r", "q")]:
            g.add_node(a, "Person")
            g.add_node(b, "Person")
            g.add_edge(a, "friend", b)
        onto = Ontology.induce(g)
        self.assertTrue(onto.relation("friend").symmetric)


if __name__ == "__main__":
    unittest.main()
