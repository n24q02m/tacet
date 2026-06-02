"""Tests for distillation, the cascade router, the benchmark and baselines."""

from __future__ import annotations

import unittest

from tacet.cascade.router import TACET
from tacet.core.graph import WorldGraph
from tacet.distill.distill import Distiller, mine_rules
from tacet.eval import baselines
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.llm.teacher import OracleTeacher
from tacet.serve.config import CascadeConfig, KGEConfig

_FAST = KGEConfig(epochs=60, seed=0)


class TestRuleMining(unittest.TestCase):
    def test_recovers_compositional_rule(self) -> None:
        # grandparent(x,y) <= parent(x,z) & parent(z,y)
        g = WorldGraph()
        chain = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")]
        for x, y in chain:
            g.add_node(x, "P")
            g.add_node(y, "P")
            g.add_edge(x, "parent", y)
        grand = {("a", "c"), ("b", "d"), ("c", "e"), ("d", "f")}
        teacher_facts = {(x, "grandparent", y) for x, y in grand}
        heads = {x for x, _ in grand}
        rules = mine_rules(
            g, teacher_facts, "grandparent", min_confidence=0.9, min_support=3, complete_heads=heads
        )
        self.assertTrue(rules)
        bodies = {r.rule.name for r in rules}
        self.assertIn("syn:grandparent<=parent.parent", bodies)

    def test_mined_rules_carry_distinct_guard(self) -> None:
        g = WorldGraph()
        for x, y in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
            g.add_node(x, "P")
            g.add_node(y, "P")
            g.add_edge(x, "parent", y)
        tf = {("a", "g", "c"), ("b", "g", "d"), ("c", "g", "e")}
        rules = mine_rules(
            g, tf, "g", min_confidence=0.9, min_support=3, complete_heads={"a", "b", "c"}
        )
        for m in rules:
            self.assertEqual(m.rule.distinct, (("?x", "?y"),))

    def test_distiller_trigger(self) -> None:
        d = Distiller(synth_trigger=3)
        self.assertFalse(d.ready_to_synthesise("rel"))
        for i in range(3):
            d.record(f"h{i}", "rel", ["t"])
        self.assertTrue(d.ready_to_synthesise("rel"))


class TestBenchmark(unittest.TestCase):
    def test_generates_valid_instance(self) -> None:
        bench = generate(BenchmarkConfig(seed=0))
        self.assertGreater(bench.graph.stats()["nodes"], 50)
        self.assertEqual(len(bench.workload), len(bench.classes))
        self.assertEqual(bench.ontology.validate(bench.graph), [])

    def test_workload_covers_all_classes(self) -> None:
        bench = generate(BenchmarkConfig(seed=0))
        self.assertEqual(
            set(bench.classes), {"STATED", "DED_GIVEN", "DED_DISCOVER", "INDUCTIVE", "NOVEL"}
        )


class TestCascadeRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = generate(BenchmarkConfig(seed=0))
        cls.teacher = OracleTeacher(cls.bench.oracle, entity_pool=cls.bench.entity_pool)

    def _tacet(self, **cfg) -> TACET:
        ak = TACET(
            self.bench.graph.copy(),
            self.bench.ontology,
            self.teacher,
            rules=list(self.bench.given_rules),
            config=CascadeConfig(kge=_FAST, **cfg),
        )
        ak.warmup(calibration=self.bench.calibration)
        return ak

    def test_tier1_answers_stated_fact(self) -> None:
        ak = self._tacet()
        person = self.bench.graph.nodes_of_type("Person")[0]
        self.assertEqual(ak.ask(person, "works_at").tier, 1)

    def test_distillation_demotes_discoverable_to_tier1(self) -> None:
        ak = self._tacet()
        # query superior_of for many heads -> teacher answers, then synthesis.
        supers = [
            p
            for p in self.bench.graph.nodes_of_type("Person")
            if self.bench.truth.get((p, "superior_of"))
        ]
        tiers = [ak.ask(p, "superior_of").tier for p in supers[:20]]
        self.assertEqual(tiers[0], 3)  # first is the teacher
        self.assertIn(1, tiers[10:])  # later ones become symbolic
        self.assertTrue(ak.synthesised_rules)

    def test_no_distillation_keeps_paying_teacher(self) -> None:
        ak = self._tacet(distillation=False)
        supers = [
            p
            for p in self.bench.graph.nodes_of_type("Person")
            if self.bench.truth.get((p, "superior_of"))
        ]
        tiers = [ak.ask(p, "superior_of").tier for p in supers[:15]]
        self.assertTrue(all(t == 3 for t in tiers))  # never learns


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = generate(BenchmarkConfig(seed=0))
        cls.teacher = OracleTeacher(cls.bench.oracle, entity_pool=cls.bench.entity_pool)
        cfg = CascadeConfig(kge=_FAST)
        cls.llm = baselines.run_llm_only(cls.bench, cls.teacher)
        cls.sym = baselines.run_symbolic_only(cls.bench)
        cls.cache = baselines.run_cache_cascade(cls.bench, cls.teacher)
        cls.tacet = baselines.run_cascade(cls.bench, cls.teacher, cfg)

    def test_llm_only_is_most_expensive_and_accurate(self) -> None:
        self.assertEqual(self.llm.accuracy, 1.0)
        self.assertGreater(self.llm.total_cost, self.cache.total_cost)

    def test_symbolic_only_is_cheap_but_incomplete(self) -> None:
        self.assertLess(self.sym.total_cost, self.tacet.total_cost)
        self.assertLess(self.sym.accuracy, 0.8)

    def test_tacet_beats_llm_on_cost(self) -> None:
        self.assertLess(self.tacet.total_cost, self.llm.total_cost / 2)

    def test_tacet_beats_cache_on_cost(self) -> None:
        # rule synthesis generalises to unseen heads; a cache cannot.
        self.assertLess(self.tacet.total_cost, self.cache.total_cost)

    def test_tacet_accuracy_is_high(self) -> None:
        self.assertGreater(self.tacet.accuracy, 0.9)

    def test_tacet_keeps_graph_well_typed(self) -> None:
        # the original benchmark graph must not be mutated by a run.
        self.assertEqual(self.bench.ontology.validate(self.bench.graph), [])


if __name__ == "__main__":
    unittest.main()
