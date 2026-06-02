"""Tests for the simplified AMIE+ rule miner (G1.4)."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.distill.amie import AMIERule, compare_with_induced, mine_amie_plus_rules


def _toy_graph() -> WorldGraph:
    """Manager-of-manager-of → indirect_manager_of (rule transitivity).

    Pattern:
        alice manages bob
        bob   manages carol
        ⇒ alice indirect_manages carol  (truth in graph)
    Repeated 10× with different names to get a large enough support.
    """
    g = WorldGraph(name="amie-toy")
    for i in range(20):
        a, b, c = f"a{i}", f"b{i}", f"c{i}"
        g.add_edge(a, "manages", b)
        g.add_edge(b, "manages", c)
        g.add_edge(a, "indirect_manages", c)
    return g


class TestAMIEMiner(unittest.TestCase):
    def test_discovers_transitivity_rule(self) -> None:
        rules = mine_amie_plus_rules(
            _toy_graph(), min_support=5, min_pca=0.8, min_head_coverage=0.1
        )
        # Must discover manages ∘ manages → indirect_manages.
        sigs = {(r.body1, r.body2, r.head) for r in rules}
        self.assertIn(("manages", "manages", "indirect_manages"), sigs)

    def test_pca_confidence_in_unit_interval(self) -> None:
        rules = mine_amie_plus_rules(_toy_graph(), min_support=5, min_pca=0.0)
        for r in rules:
            self.assertGreaterEqual(r.pca_confidence, 0.0)
            self.assertLessEqual(r.pca_confidence, 1.0)
            self.assertGreaterEqual(r.std_confidence, 0.0)
            self.assertLessEqual(r.std_confidence, 1.0)

    def test_min_support_filter(self) -> None:
        # With min_support = 100, no rule passes (the toy has only 20 instances).
        rules = mine_amie_plus_rules(_toy_graph(), min_support=100, min_pca=0.0)
        self.assertEqual(rules, [])

    def test_rule_dataclass_fields(self) -> None:
        rules = mine_amie_plus_rules(_toy_graph(), min_support=5, min_pca=0.8)
        r = rules[0]
        self.assertIsInstance(r, AMIERule)
        for field in (
            "body1",
            "body2",
            "head",
            "support",
            "head_coverage",
            "std_confidence",
            "pca_confidence",
        ):
            self.assertTrue(hasattr(r, field))


class TestCompareWithInduced(unittest.TestCase):
    def test_no_induced_yields_zero_recall(self) -> None:
        out = compare_with_induced(_toy_graph(), induced_rules=[], min_support=5, min_pca=0.8)
        self.assertEqual(out["n_overlap"], 0)
        self.assertEqual(out["amie_recall_of_induced"], 0.0)


if __name__ == "__main__":
    unittest.main()
