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


class TestTargetInBody(unittest.TestCase):
    """The length-2 branch may chain the mining target into its own body.

    ``mine_rules`` whitelists ``allowed_body | {target}``, and only the
    length-1 branch guards against ``r1 == target``. The self-referential
    candidate that results is mined from the teacher's own write-back edges,
    so ``allow_target_in_body=False`` suppresses it without touching rules
    whose body is made of base relations.
    """

    def _graph(self) -> WorldGraph:
        # tgt(x, y) is genuinely R1(x, z) & R2(z, y), and the write-back edges
        # for tgt are present, so tgt <= tgt.tgt is also a well-formed candidate.
        g = WorldGraph()
        for i in range(8):
            g.add_edge(f"x{i}", "R1", f"z{i}")
            g.add_edge(f"z{i}", "R2", f"y{i}")
            g.add_edge(f"x{i}", "tgt", f"y{i}")
        # a chain of tgt edges, so tgt.tgt has groundings of its own
        for i in range(7):
            g.add_edge(f"y{i}", "tgt", f"y{i + 1}")
        return g

    def test_target_reaches_its_own_body_by_default(self) -> None:
        _, n_default = mine_rules_with_stats(
            self._graph(), set(), "tgt", min_confidence=0.0, min_support=1
        )
        _, n_forbidden = mine_rules_with_stats(
            self._graph(),
            set(),
            "tgt",
            min_confidence=0.0,
            min_support=1,
            allow_target_in_body=False,
        )
        # Forbidding it can only remove candidates, never add them.
        self.assertLess(n_forbidden, n_default)

    def test_forbidding_the_target_drops_only_self_referential_rules(self) -> None:
        graph = self._graph()
        kw = dict(min_confidence=0.0, min_support=1, allowed_body={"R1", "R2"})
        default = mine_rules(graph, set(), "tgt", **kw)
        forbidden = mine_rules(graph, set(), "tgt", allow_target_in_body=False, **kw)

        def bodies(rules):
            return {r.rule.name for r in rules}

        # every dropped rule mentions the target in its body
        for name in bodies(default) - bodies(forbidden):
            self.assertIn("tgt", name.split("<=", 1)[1])
        # nothing survives that still mentions the target in its body
        for rule in forbidden:
            self.assertNotIn("tgt", {rel for _s, rel, _o in rule.rule.body})

    def test_the_true_base_composition_survives(self) -> None:
        graph = self._graph()
        kw = dict(min_confidence=0.9, min_support=3, allowed_body={"R1", "R2"})
        default = {r.rule.name for r in mine_rules(graph, set(), "tgt", **kw)}
        forbidden = {
            r.rule.name for r in mine_rules(graph, set(), "tgt", allow_target_in_body=False, **kw)
        }
        self.assertIn("syn:tgt<=R1.R2", default)
        self.assertIn("syn:tgt<=R1.R2", forbidden)

    def test_default_is_unchanged(self) -> None:
        # Reproducibility guard: the published runs must replay bit-for-bit, so
        # the default must behave exactly as the flag-free call did.
        graph = self._graph()
        kw = dict(min_confidence=0.5, min_support=2)
        explicit = mine_rules(graph, set(), "tgt", allow_target_in_body=True, **kw)
        implicit = mine_rules(graph, set(), "tgt", **kw)
        self.assertEqual(
            [(r.rule.name, r.confidence, r.support) for r in explicit],
            [(r.rule.name, r.confidence, r.support) for r in implicit],
        )


if __name__ == "__main__":
    unittest.main()
