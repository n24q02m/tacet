import unittest
from unittest.mock import MagicMock

from tacet.cascade.config import CascadeConfig
from tacet.cascade.router import TACET, Answer
from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology, RelationType
from tacet.core.symbolic import Rule
from tacet.distill.distill import MinedRule
from tacet.kge.kge import KGEConfig, Prediction
from tacet.llm.teacher import OracleTeacher


class TestAnswer(unittest.TestCase):
    def test_answer_creation(self):
        ans = Answer(
            head="Alice",
            relation="works_at",
            tier=1,
            answers=["ACME"],
            text="Alice works at ACME",
            confidence=1.0,
            cost=0.0001,
            latency_ms=3.0,
            proof=["rule1"],
            note="test note",
        )
        self.assertEqual(ans.head, "Alice")
        self.assertEqual(ans.tier, 1)
        self.assertEqual(ans.answers, ["ACME"])


class TestTACET(unittest.TestCase):
    def setUp(self):
        self.graph = WorldGraph()
        self.graph.add_node("Alice", "Person")
        self.graph.add_node("Bob", "Person")
        self.graph.add_node("ACME", "Company")
        self.graph.add_edge("Alice", "works_at", "ACME")

        self.ontology = Ontology.induce(self.graph)

        def oracle(h, r):
            if h == "Bob" and r == "works_at":
                return ["Globex"]
            if h == "Alice" and r == "works_at":
                return ["ACME"]
            return []

        self.teacher = OracleTeacher(oracle)
        self.config = CascadeConfig(kge=KGEConfig(epochs=1))

    def test_init_defaults(self):
        ak = TACET(self.graph, self.ontology, self.teacher)
        self.assertEqual(ak.graph, self.graph)

    def test_warmup(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak.warmup()
        self.assertTrue(ak._kge_ready)

    def test_ask_tier1(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak.warmup()
        ans = ak.ask("Alice", "works_at")
        self.assertEqual(ans.tier, 1)

    def test_ask_tier3(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak.warmup()
        ans = ak.ask("Bob", "works_at")
        self.assertEqual(ans.tier, 3)

    def test_report(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak.warmup()
        ak.ask("Alice", "works_at")
        ak.ask("Bob", "works_at")
        rep = ak.report()
        self.assertEqual(rep["queries"], 2)

    def test_type_endpoints(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak._type_endpoints("works_at", "Charlie", "Globex")
        self.assertEqual(ak.graph.node("Charlie").type, "Person")
        self.assertEqual(ak.graph.node("Globex").type, "Company")

    def test_type_endpoints_self_loop(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        self.ontology.add_relation_type(
            RelationType("r_self", domain=frozenset({"A", "B"}), range=frozenset({"B", "C"}))
        )
        ak._type_endpoints("r_self", "Z", "Z")
        self.assertEqual(ak.graph.node("Z").type, "B")

    def test_candidates(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        cands = ak._candidates("works_at")
        self.assertEqual(cands, ["ACME"])

    def test_rule_synthesis_called(self):
        cfg = CascadeConfig(synth_trigger=1, rule_synthesis=True)
        ak = TACET(self.graph, self.ontology, self.teacher, config=cfg)
        ak.warmup()
        ak.distiller.synthesise = MagicMock(return_value=[])

        ak.ask("Bob", "works_at")
        ak.distiller.synthesise.assert_called_once()

    def test_consolidate(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak.warmup()
        ak.consolidate()
        self.assertTrue(ak._kge_ready)

    def test_warmup_with_calibration_and_tier2_hit(self):
        ak = TACET(self.graph, self.ontology, self.teacher, config=self.config)
        ak.warmup(calibration=[("Alice", "works_at", ["ACME"], "ACME")])
        self.assertTrue(ak._kge_ready)

        # Second call exercises _candidate_cache hit and unknown-relation fallback
        c1 = ak._candidates("works_at")
        c2 = ak._candidates("works_at")
        self.assertEqual(c1, c2)
        all_ents = ak._candidates("unknown_rel")
        self.assertEqual(set(all_ents), {"Alice", "Bob", "ACME"})

        # Tier 2 KGE hit in default cascade
        ak.kge.predict_tail = MagicMock(return_value=Prediction(tail="ACME", confidence=0.95))
        ans = ak.ask("Bob", "works_at")
        self.assertEqual(ans.tier, 2)
        self.assertEqual(ans.answers, ["ACME"])
        self.assertFalse(ans.grounded)

    def test_distil_installs_synthesised_rule_and_wildcard_endpoints(self):
        cfg = CascadeConfig(synth_trigger=1, rule_synthesis=True, kge=KGEConfig(epochs=1))
        ak = TACET(self.graph, self.ontology, self.teacher, config=cfg)
        ak.warmup()

        new_rule = Rule("syn_works", (("?x", "works_at", "?y"),), ("?x", "works_at", "?y"))
        mined = MinedRule(rule=new_rule, confidence=1.0, support=3)
        ak.distiller.synthesise = MagicMock(return_value=[mined])
        ak.ask("Bob", "works_at")
        self.assertIn("syn_works", ak.synthesised_rules)

        # Unknown relation and open "*" range in _type_endpoints / _ensure_type
        ak._type_endpoints("no_such_rel", "X", "Y")
        self.ontology.add_relation_type(
            RelationType("open_rel", domain=frozenset({"*"}), range=frozenset({"*"}))
        )
        ak._type_endpoints("open_rel", "X_new", "Y_new")
        self.assertIsNone(ak.graph.node("X_new"))


if __name__ == "__main__":
    unittest.main()
