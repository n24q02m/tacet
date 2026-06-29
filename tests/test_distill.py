import unittest

from tacet.core.graph import WorldGraph
from tacet.distill.distill import (
    Distiller,
    _adj,
    _directed,
    _pairs_index,
    mine_rules,
    mine_rules_with_stats,
)


class TestDistillHelpers(unittest.TestCase):
    def test_pairs_index(self):
        facts = {("a", "r1", "b"), ("c", "r1", "d"), ("e", "r2", "f")}
        idx = _pairs_index(facts)
        self.assertEqual(idx["r1"], {("a", "b"), ("c", "d")})
        self.assertEqual(idx["r2"], {("e", "f")})

    def test_directed(self):
        pairs = {("a", "b"), ("c", "d")}
        self.assertEqual(_directed(pairs, False), pairs)
        self.assertEqual(_directed(pairs, True), {("b", "a"), ("d", "c")})

    def test_adj(self):
        pairs = {("a", "b"), ("a", "c"), ("b", "d")}
        adj = _adj(pairs)
        self.assertEqual(adj["a"], {"b", "c"})
        self.assertEqual(adj["b"], {"d"})
        self.assertNotIn("c", adj)

class TestRuleMining(unittest.TestCase):
    def setUp(self):
        self.graph = WorldGraph(name="test-graph")
        # R1(x, y) => target(x, y)
        for i in range(10):
            self.graph.add_edge(f"x{i}", "R1", f"y{i}")
            self.graph.add_edge(f"x{i}", "target", f"y{i}")

        # R2(x, z) & R3(z, y) => target2(x, y)
        for i in range(10):
            self.graph.add_edge(f"a{i}", "R2", f"b{i}")
            self.graph.add_edge(f"b{i}", "R3", f"c{i}")
            self.graph.add_edge(f"a{i}", "target2", f"c{i}")

    def test_mine_length_1_rule(self):
        rules = mine_rules(self.graph, set(), "target", min_support=5)
        names = {r.rule.name for r in rules}
        self.assertIn("syn:target<=R1", names)

        rule = next(r for r in rules if r.rule.name == "syn:target<=R1")
        self.assertEqual(rule.confidence, 1.0)
        self.assertEqual(rule.support, 10)

    def test_mine_length_2_rule(self):
        rules = mine_rules(self.graph, set(), "target2", min_support=5)
        names = {r.rule.name for r in rules}
        self.assertIn("syn:target2<=R2.R3", names)

        rule = next(r for r in rules if r.rule.name == "syn:target2<=R2.R3")
        self.assertEqual(rule.confidence, 1.0)
        self.assertEqual(rule.support, 10)

    def test_mine_rules_with_stats(self):
        rules, n_proposed = mine_rules_with_stats(self.graph, set(), "target", min_support=5)
        self.assertGreater(n_proposed, 0)
        self.assertEqual(len(rules), 1)

    def test_min_confidence_filter(self):
        # Add some noise to R1
        self.graph.add_edge("noise_h", "R1", "noise_t")
        # target does NOT have noise_h -> noise_t

        # 10 matches, 11 total in R1 body. Conf = 10/11 = 0.909
        rules = mine_rules(self.graph, set(), "target", min_confidence=0.95, min_support=5)
        names = {r.rule.name for r in rules}
        self.assertNotIn("syn:target<=R1", names)

        rules = mine_rules(self.graph, set(), "target", min_confidence=0.90, min_support=5)
        names = {r.rule.name for r in rules}
        self.assertIn("syn:target<=R1", names)

class TestDistiller(unittest.TestCase):
    def test_distiller_flow(self):
        d = Distiller(synth_trigger=3, min_support=2)
        graph = WorldGraph()

        # Add some base facts to graph
        graph.add_edge("a1", "R1", "b1")
        graph.add_edge("a2", "R1", "b2")
        graph.add_edge("a3", "R1", "b3")

        # Record teacher facts for 'target'
        d.record("a1", "target", ["b1"])
        d.record("a2", "target", ["b2"])

        self.assertFalse(d.ready_to_synthesise("target"))

        d.record("a3", "target", ["b3"])
        self.assertTrue(d.ready_to_synthesise("target"))

        rules = d.synthesise(graph, "target")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rule.name, "syn:target<=R1")

        # KGE augmentation
        kge_facts = d.kge_augmentation()
        self.assertEqual(len(kge_facts), 3)
        self.assertIn(("a1", "target", "b1"), kge_facts)

if __name__ == "__main__":
    unittest.main()
