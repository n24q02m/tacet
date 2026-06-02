"""Tests for the Tier-1 symbolic rule engine."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Derivation, Rule, RuleEngine


def _engine(
    rules: list[Rule], symmetric: bool = False, transitive: bool = False
) -> tuple[RuleEngine, WorldGraph]:
    onto = Ontology()
    onto.add_node_type(NodeType("T"))
    onto.add_relation_type(
        RelationType(
            "r", frozenset({"T"}), frozenset({"T"}), symmetric=symmetric, transitive=transitive
        )
    )
    onto.add_relation_type(RelationType("p", frozenset({"T"}), frozenset({"T"})))
    onto.add_relation_type(RelationType("anc", frozenset({"T"}), frozenset({"T"})))
    return RuleEngine(onto, rules), WorldGraph()


class TestRuleEngine(unittest.TestCase):
    def test_transitive_axiom_closure(self) -> None:
        engine, g = _engine([], transitive=True)
        for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_node(a, "T")
            g.add_node(b, "T")
            g.add_edge(a, "r", b)
        engine.materialise(g)
        self.assertEqual(engine.query("a", "r").answers, ["b", "c", "d"])

    def test_symmetric_axiom_closure(self) -> None:
        engine, g = _engine([], symmetric=True)
        g.add_node("a", "T")
        g.add_node("b", "T")
        g.add_edge("a", "r", "b")
        engine.materialise(g)
        self.assertEqual(engine.query("b", "r").answers, ["a"])

    def test_rule_derivation_and_provenance(self) -> None:
        rule = Rule("anc_base", (("?x", "p", "?y"),), ("?x", "anc", "?y"))
        engine, g = _engine([rule])
        g.add_node("a", "T")
        g.add_node("b", "T")
        g.add_edge("a", "p", "b")
        engine.materialise(g)
        res = engine.query("a", "anc")
        self.assertEqual(res.answers, ["b"])
        self.assertTrue(any("DERIVED" in line for line in res.proof))

    def test_recursive_rule_computes_closure(self) -> None:
        rules = [
            Rule("anc_base", (("?x", "p", "?y"),), ("?x", "anc", "?y")),
            Rule("anc_step", (("?x", "anc", "?z"), ("?z", "p", "?y")), ("?x", "anc", "?y")),
        ]
        engine, g = _engine(rules)
        for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_node(a, "T")
            g.add_node(b, "T")
            g.add_edge(a, "p", b)
        engine.materialise(g)
        self.assertEqual(engine.query("a", "anc").answers, ["b", "c", "d"])

    def test_distinct_guard_blocks_self_loops(self) -> None:
        # without the guard, p(x,y) & p(y,x) would derive anc(x,x).
        rule = Rule(
            "loopy",
            (("?x", "p", "?z"), ("?z", "p", "?y")),
            ("?x", "anc", "?y"),
            distinct=(("?x", "?y"),),
        )
        engine, g = _engine([rule])
        g.add_node("a", "T")
        g.add_node("b", "T")
        g.add_edge("a", "p", "b")
        g.add_edge("b", "p", "a")
        engine.materialise(g)
        self.assertNotIn(("a", "anc", "a"), engine.closure)

    def test_abstains_when_underivable(self) -> None:
        engine, g = _engine([])
        g.add_node("a", "T")
        engine.materialise(g)
        self.assertFalse(engine.query("a", "anc").answered)

    def test_range_restriction_enforced(self) -> None:
        with self.assertRaises(ValueError):
            Rule("bad", (("?x", "p", "?y"),), ("?x", "anc", "?z"))

    def test_add_fact_is_queryable(self) -> None:
        engine, g = _engine([])
        g.add_node("a", "T")
        engine.materialise(g)
        engine.add_fact(("a", "anc", "b"))
        self.assertEqual(engine.query("a", "anc").answers, ["b"])

    def test_add_rule_rejects_ontology_inconsistent_rule(self) -> None:
        # A typed ontology: People relate via parent_of/ancestor_of (Person->Person)
        # and live_in (Person->City). Variable types are inferred from domain/range.
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        onto.add_node_type(NodeType("City"))
        onto.add_relation_type(
            RelationType("parent_of", frozenset({"Person"}), frozenset({"Person"}))
        )
        onto.add_relation_type(
            RelationType("ancestor_of", frozenset({"Person"}), frozenset({"Person"}))
        )
        onto.add_relation_type(RelationType("live_in", frozenset({"Person"}), frozenset({"City"})))
        engine = RuleEngine(onto)

        # (1) Rule over a relation unknown to the ontology is REJECTED.
        unknown = Rule("u", (("?x", "sibling_of", "?y"),), ("?x", "ancestor_of", "?y"))
        self.assertFalse(engine.add_rule(unknown))
        self.assertNotIn(unknown, engine.rules)

        # (2) Rule that violates declared domain/range types is REJECTED.
        # parent_of(x,z) & live_in(z,y) => ancestor_of(x,y): z must be Person (as
        # tail of parent_of) AND Person (as head of live_in) -> ok for z; but y is
        # a City (tail of live_in) yet ancestor_of requires y to be a Person.
        type_clash = Rule(
            "clash",
            (("?x", "parent_of", "?z"), ("?z", "live_in", "?y")),
            ("?x", "ancestor_of", "?y"),
            distinct=(("?x", "?y"),),
        )
        self.assertFalse(engine.add_rule(type_clash))
        self.assertNotIn(type_clash, engine.rules)

        # (3) A type-consistent rule is ACCEPTED (returns True).
        good = Rule(
            "anc_base",
            (("?x", "parent_of", "?y"),),
            ("?x", "ancestor_of", "?y"),
            distinct=(("?x", "?y"),),
        )
        self.assertTrue(engine.add_rule(good))
        self.assertIn(good, engine.rules)

        # (4) Regression: a representative MINED rule shape (length-2 Horn rule
        # over existing typed relations, as the Distiller produces) is ACCEPTED.
        mined = Rule(
            "syn:ancestor_of<=parent_of.parent_of",
            (("?x", "parent_of", "?z"), ("?z", "parent_of", "?y")),
            ("?x", "ancestor_of", "?y"),
            distinct=(("?x", "?y"),),
        )
        self.assertTrue(engine.add_rule(mined))
        self.assertIn(mined, engine.rules)

    def test_add_rule_rejects_multitype_domain_widening(self) -> None:
        # Ontology-preservation needs body-induced types SUBSET-OF head types, not
        # merely a non-empty intersection. ``manages`` admits {Person, Robot} but
        # ``boss`` requires {Person}: the chain rule below has a non-empty
        # intersection ({Person, Robot} & {Person} = {Person}) yet can bind the
        # head subject to a Robot. add_rule MUST reject it, and -- crucially -- the
        # materialised closure must contain no ontology violation.
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        onto.add_node_type(NodeType("Robot"))
        onto.add_relation_type(
            RelationType("manages", frozenset({"Person", "Robot"}), frozenset({"Person", "Robot"}))
        )
        onto.add_relation_type(RelationType("boss", frozenset({"Person"}), frozenset({"Person"})))
        engine = RuleEngine(onto)

        widening = Rule(
            "boss_via_chain",
            (("?x", "manages", "?y"), ("?y", "manages", "?z")),
            ("?x", "boss", "?z"),
            distinct=(("?x", "?z"),),
        )
        self.assertFalse(engine.add_rule(widening))
        self.assertNotIn(widening, engine.rules)

        g = WorldGraph()
        g.add_node("robot1", "Robot")
        g.add_node("alice", "Person")
        g.add_node("bob", "Person")
        g.add_edge("robot1", "manages", "alice")
        g.add_edge("alice", "manages", "bob")
        closure = engine.materialise(g)

        closed = WorldGraph()
        for node in g.nodes:
            closed.add_node(node.id, node.type)
        for h, r, t in closure:
            closed.add_edge(h, r, t)
        self.assertEqual(onto.validate(closed), [])

    def test_materialise_raises_when_fixpoint_not_reached(self) -> None:
        # A deep recursive chain needs many passes; a tight iteration cap cannot
        # reach the fixpoint, so the closure would be incomplete. The engine must
        # fail loudly rather than silently return a truncated closure.
        onto = Ontology()
        onto.add_node_type(NodeType("T"))
        onto.add_relation_type(
            RelationType("r", frozenset({"T"}), frozenset({"T"}), transitive=True)
        )
        engine = RuleEngine(onto, max_iterations=2)
        g = WorldGraph()
        for i in range(8):
            g.add_node(f"n{i}", "T")
            if i:
                g.add_edge(f"n{i - 1}", "r", f"n{i}")
        with self.assertRaises(RuntimeError):
            engine.materialise(g)

        # With a sufficient cap the same graph converges and the closure is complete.
        engine_ok = RuleEngine(onto, max_iterations=100)
        closure = engine_ok.materialise(g)
        self.assertIn(("n0", "r", "n7"), closure)

    def test_explain_terminates_on_cyclic_rules(self) -> None:
        # A symmetric rule set can leave the proof provenance with a cycle:
        # col(a,b) and col(b,a) each cite the other as their derivation. The
        # proof renderer (`_explain`) must not recurse into an already-visited
        # triple, exactly as `proof_is_grounded` already guards against.
        onto = Ontology()
        onto.add_node_type(NodeType("T"))
        onto.add_relation_type(
            RelationType("col", frozenset({"T"}), frozenset({"T"}), symmetric=True)
        )
        engine = RuleEngine(onto, [Rule("rec_sym", (("?x", "col", "?y"),), ("?y", "col", "?x"))])
        g = WorldGraph()
        g.add_node("a", "T")
        g.add_node("b", "T")
        g.add_edge("a", "col", "b")
        engine.materialise(g)

        # Force the pathological symmetric provenance the guard defends against:
        # the two directions cite each other, so a naive renderer recurses forever.
        t1, t2 = ("a", "col", "b"), ("b", "col", "a")
        engine._derivations[t1] = Derivation(t1, "rec_sym", (t2,))
        engine._derivations[t2] = Derivation(t2, "rec_sym", (t1,))

        # Run the renderer in a worker thread so a non-terminating recursion is
        # caught as a timeout rather than hanging the whole suite.
        import threading

        result: list[list[str]] = []

        def _run() -> None:
            result.append(engine._explain(t1))

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive(), "_explain did not terminate on a cyclic rule set")
        self.assertTrue(result, "_explain returned no proof")
        proof = result[0]
        # A bounded proof: each cited triple appears at most a small number of
        # times (the cycle is cut, not unfolded indefinitely).
        self.assertLessEqual(len(proof), 16)

    def test_add_rule_permissive_on_untyped_ontology(self) -> None:
        # An untyped/empty ontology constrains nothing: every relation defaults to
        # domain/range {"*"}, so any range-restricted rule must be accepted.
        engine = RuleEngine(Ontology())
        rule = Rule("anything", (("?x", "p", "?y"),), ("?x", "anc", "?y"))
        self.assertTrue(engine.add_rule(rule))


if __name__ == "__main__":
    unittest.main()
