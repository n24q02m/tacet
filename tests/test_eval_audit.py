"""Tests for the proof-tree validity and auditability evaluation (Task 3)."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule, RuleEngine
from tacet.eval.eval_audit import proof_coverage, proof_validity


def _engine(rules: list[Rule]) -> tuple[RuleEngine, WorldGraph]:
    onto = Ontology()
    onto.add_node_type(NodeType("T"))
    onto.add_relation_type(RelationType("parent_of", frozenset({"T"}), frozenset({"T"})))
    onto.add_relation_type(RelationType("ancestor_of", frozenset({"T"}), frozenset({"T"})))
    onto.add_relation_type(RelationType("favourite_food", frozenset({"T"}), frozenset({"T"})))
    return RuleEngine(onto, rules), WorldGraph()


class TestProofValidity(unittest.TestCase):
    def test_grandparent_proof_is_fully_grounded(self) -> None:
        # a -parent-> b -parent-> c ; ancestor closure of `a` is {b, c}.
        rules = [
            Rule("anc_base", (("?x", "parent_of", "?y"),), ("?x", "ancestor_of", "?y")),
            Rule(
                "anc_step",
                (("?x", "ancestor_of", "?z"), ("?z", "parent_of", "?y")),
                ("?x", "ancestor_of", "?y"),
            ),
        ]
        engine, g = _engine(rules)
        for a, b in [("a", "b"), ("b", "c")]:
            g.add_node(a, "T")
            g.add_node(b, "T")
            g.add_edge(a, "parent_of", b)
        engine.materialise(g)
        result = engine.query("a", "ancestor_of")
        self.assertEqual(result.answers, ["b", "c"])
        # every derivation step reduces to a base fact or a known rule.
        self.assertEqual(proof_validity(engine, result), 1.0)
        self.assertEqual(proof_coverage(engine, result), 1.0)

    def test_unanswered_query_has_zero_validity_and_coverage(self) -> None:
        # no rule entails favourite_food, and no base fact stored -> abstain.
        engine, g = _engine([])
        g.add_node("a", "T")
        engine.materialise(g)
        result = engine.query("a", "favourite_food")
        self.assertFalse(result.answered)
        self.assertEqual(proof_coverage(engine, result), 0.0)
        self.assertEqual(proof_validity(engine, result), 0.0)

    def test_writeback_fact_is_grounded_base_leaf(self) -> None:
        # a fact inserted via add_fact is a base leaf -> grounded proof.
        engine, g = _engine([])
        g.add_node("a", "T")
        engine.materialise(g)
        engine.add_fact(("a", "favourite_food", "pho"))
        result = engine.query("a", "favourite_food")
        self.assertTrue(result.answered)
        self.assertEqual(proof_coverage(engine, result), 1.0)
        self.assertEqual(proof_validity(engine, result), 1.0)


if __name__ == "__main__":
    unittest.main()
