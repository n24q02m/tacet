"""Tests cho TACETService.save_state / load_state + rule JSON helpers (G2.3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tacet.core.symbolic import Rule, load_rules_json, save_rules_json


class TestRuleJsonRoundtrip(unittest.TestCase):
    def test_save_then_load_returns_same_rules(self) -> None:
        rules = [
            Rule(
                name="parent_to_ancestor",
                body=(("?x", "parent_of", "?y"),),
                head=("?x", "ancestor_of", "?y"),
            ),
            Rule(
                name="trans_ancestor",
                body=(
                    ("?x", "ancestor_of", "?y"),
                    ("?y", "ancestor_of", "?z"),
                ),
                head=("?x", "ancestor_of", "?z"),
                distinct=(("?x", "?z"),),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            save_rules_json(rules, path)
            self.assertTrue(path.exists())
            restored = load_rules_json(path)
            self.assertEqual(len(restored), len(rules))
            self.assertEqual({r.name for r in restored}, {r.name for r in rules})
            for orig, rest in zip(rules, restored, strict=False):
                self.assertEqual(orig.head, rest.head)
                self.assertEqual(orig.body, rest.body)
                self.assertEqual(orig.distinct, rest.distinct)


class TestTACETServiceState(unittest.TestCase):
    """save_state -> load_state round-trip preserves rules + graph + episodes."""

    def _build_service(self):  # noqa: ANN202
        from tacet.cascade.router import TACET
        from tacet.core.graph import WorldGraph
        from tacet.core.ontology import NodeType, Ontology, RelationType
        from tacet.llm.teacher import OracleTeacher
        from tacet.serve.server import TACETService
        from tacet.serve.settings import load_settings

        ont = Ontology()
        ont.add_node_type(NodeType("Person"))
        ont.add_relation_type(RelationType("knows", frozenset({"Person"}), frozenset({"Person"})))
        g = WorldGraph(name="state-test")
        for s, r, t in [
            ("alice", "knows", "bob"),
            ("bob", "knows", "carol"),
            ("alice", "knows", "carol"),
        ]:
            g.add_edge(s, r, t)
        # Use the oracle teacher so the cascade builds without LLM keys.
        teacher = OracleTeacher(lambda _h, _r: [])
        engine = TACET(g, ont, teacher)
        settings = load_settings()
        return TACETService(engine, settings)

    def test_save_state_writes_files(self) -> None:
        service = self._build_service()
        with tempfile.TemporaryDirectory() as tmp:
            report = service.save_state(tmp)
            self.assertEqual(report["edges"], 3)
            self.assertGreaterEqual(report["rules"], 0)
            self.assertTrue((Path(tmp) / "rules.json").exists())
            self.assertTrue((Path(tmp) / "graph.tsv").exists())
            self.assertTrue((Path(tmp) / "episodes.jsonl").exists())

    def test_load_state_restores_graph(self) -> None:
        # Save from one service with 3 edges, then load into a *fresh*
        # service that started with an empty graph.
        from tacet.cascade.router import TACET
        from tacet.core.graph import WorldGraph
        from tacet.core.ontology import NodeType, Ontology, RelationType
        from tacet.llm.teacher import OracleTeacher
        from tacet.serve.server import TACETService
        from tacet.serve.settings import load_settings

        original = self._build_service()
        with tempfile.TemporaryDirectory() as tmp:
            original.save_state(tmp)

            # Fresh service with an empty graph (same ontology, no edges).
            ont = Ontology()
            ont.add_node_type(NodeType("Person"))
            ont.add_relation_type(
                RelationType("knows", frozenset({"Person"}), frozenset({"Person"}))
            )
            empty_graph = WorldGraph(name="empty")
            teacher = OracleTeacher(lambda _h, _r: [])
            empty_service = TACETService(TACET(empty_graph, ont, teacher), load_settings())
            self.assertEqual(len(empty_service.engine.graph.edges), 0)
            report = empty_service.load_state(tmp)
            self.assertEqual(report["edges"], 3)
            self.assertEqual(len(empty_service.engine.graph.edges), 3)

    def test_load_state_missing_dir_raises(self) -> None:
        service = self._build_service()
        with self.assertRaises(FileNotFoundError):
            service.load_state("/path/that/does/not/exist_xyz")


if __name__ == "__main__":
    unittest.main()
