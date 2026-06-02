"""Tests for the ProofWriter (CWA) deductive-reasoning loader."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.core.symbolic import RuleEngine
from tacet.data.proofwriter import (
    ProofWriterBenchmark,
    ProofWriterTheory,
    _parse_atom,
    _parse_rule,
    load_proofwriter,
)


class TestParser(unittest.TestCase):
    def test_parse_atom_positive(self) -> None:
        self.assertEqual(
            _parse_atom('("Bob" "is" "smart" "+")'),
            ("Bob", "is", "smart", "+"),
        )

    def test_parse_atom_negation_marker(self) -> None:
        # rule-body negation-as-failure uses the '~' polarity marker
        self.assertEqual(
            _parse_atom('("someone" "is" "blue" "~")'),
            ("someone", "is", "blue", "~"),
        )

    def test_parse_atom_negated_query(self) -> None:
        # negated queries use the '-' polarity marker
        self.assertEqual(
            _parse_atom('("Erin" "is" "smart" "-")'),
            ("Erin", "is", "smart", "-"),
        )

    def test_parse_rule_variable_becomes_qx(self) -> None:
        # "All nice things are smart." -> body (?x is nice) head (?x is smart)
        rep = '((("something" "is" "nice" "+")) -> ("something" "is" "smart" "+"))'
        body, head, has_neg = _parse_rule(rep)
        self.assertFalse(has_neg)
        self.assertEqual(body, (("?x", "is", "nice"),))
        self.assertEqual(head, ("?x", "is", "smart"))

    def test_parse_rule_someone_is_also_a_variable(self) -> None:
        rep = '((("someone" "is" "white" "+")) -> ("someone" "is" "blue" "+"))'
        body, head, has_neg = _parse_rule(rep)
        self.assertFalse(has_neg)
        self.assertEqual(body, (("?x", "is", "white"),))
        self.assertEqual(head, ("?x", "is", "blue"))

    def test_parse_rule_multi_body(self) -> None:
        rep = (
            '((("something" "is" "smart" "+") ("something" "is" "nice" "+")) '
            '-> ("something" "is" "green" "+"))'
        )
        body, head, has_neg = _parse_rule(rep)
        self.assertFalse(has_neg)
        self.assertEqual(body, (("?x", "is", "smart"), ("?x", "is", "nice")))
        self.assertEqual(head, ("?x", "is", "green"))

    def test_parse_rule_constant_grounded(self) -> None:
        # a fully grounded rule keeps Bob as a constant (no '?')
        rep = '((("Bob" "is" "young" "+")) -> ("Bob" "is" "rough" "+"))'
        body, head, has_neg = _parse_rule(rep)
        self.assertFalse(has_neg)
        self.assertEqual(body, (("Bob", "is", "young"),))
        self.assertEqual(head, ("Bob", "is", "rough"))

    def test_parse_rule_detects_negation(self) -> None:
        rep = (
            '((("someone" "is" "white" "+") ("someone" "is" "blue" "~")) '
            '-> ("someone" "is" "young" "+"))'
        )
        _body, _head, has_neg = _parse_rule(rep)
        self.assertTrue(has_neg)


class TestLoadProofWriter(unittest.TestCase):
    def setUp(self) -> None:
        # The ProofWriter corpus is gitignored and is not shipped in this
        # public clone (it lives only in the private source's data/). Skip the
        # dataset-backed tests when the files are absent, matching the
        # missing-resource skip pattern used by the GPU/data tests.
        try:
            load_proofwriter(depth=2, split="dev", limit=1)
        except FileNotFoundError:
            self.skipTest("ProofWriter dataset not present (gitignored data/)")

    def test_load_depth2_dev(self) -> None:
        bench = load_proofwriter(depth=2, split="dev", limit=20)
        self.assertIsInstance(bench, ProofWriterBenchmark)
        self.assertEqual(bench.depth, 2)
        self.assertEqual(len(bench.theories), 20)
        # every theory carries a graph, rule list, questions and an expressible flag
        for th in bench.theories:
            self.assertIsInstance(th, ProofWriterTheory)
            self.assertIsInstance(th.graph, WorldGraph)
            self.assertIsInstance(th.expressible, bool)
            self.assertGreater(len(th.questions), 0)

    def test_at_least_one_expressible_theory_has_facts_and_rules(self) -> None:
        bench = load_proofwriter(depth=2, split="dev", limit=20)
        expressible = [t for t in bench.theories if t.expressible]
        self.assertGreater(len(expressible), 0)
        th = expressible[0]
        self.assertGreater(len(th.graph.edges), 0)
        self.assertGreater(len(th.rules), 0)
        # one query per question
        self.assertEqual(len(th.questions), len(th.question_atoms()))

    def test_excluded_count_is_tracked(self) -> None:
        bench = load_proofwriter(depth=2, split="dev", limit=50)
        n_excl = sum(1 for t in bench.theories if not t.expressible)
        self.assertEqual(bench.n_excluded, n_excl)

    def test_known_line_solves_via_engine(self) -> None:
        # The first depth-2 dev theory (AttNoneg-CWA-D2-1286): Harry is nice,
        # all nice things are smart, smart+nice -> green; so "Harry is green"
        # is derivable under positive Datalog.
        bench = load_proofwriter(depth=2, split="dev", limit=1)
        th = bench.theories[0]
        self.assertTrue(th.expressible)
        onto = Ontology()
        engine = RuleEngine(onto, list(th.rules))
        engine.materialise(th.graph)
        res = engine.query("Harry", "is")
        self.assertTrue(res.answered)
        self.assertIn("green", res.answers)
        self.assertIn("smart", res.answers)


if __name__ == "__main__":
    unittest.main()
