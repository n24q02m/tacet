"""Tests for the Tier-1 symbolic rule engine."""

from __future__ import annotations

import unittest
import unittest.mock
from dataclasses import FrozenInstanceError

from tacet.core import symbolic
from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import (
    Derivation,
    EntailmentCheck,
    Rule,
    RuleEngine,
    load_rules_json,
    save_rules_json,
)


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

    def test_join_streams_bindings_instead_of_materialising_them(self) -> None:
        # Regression for the mining OOM: _join used to build the whole binding
        # list for every atom, so a high-fanout relation put the entire
        # intermediate result in memory before the caller saw one row.
        n = 300
        facts = [(f"x{i}", "r", f"z{i}") for i in range(n)]
        facts += [(f"z{i}", "p", f"y{i}") for i in range(n)]
        idx_all: dict[str, list[tuple[str, str, str]]] = {}
        idx_subj: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        idx_obj: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for fact in facts:
            h, r, t = fact
            idx_all.setdefault(r, []).append(fact)
            idx_subj.setdefault((r, h), []).append(fact)
            idx_obj.setdefault((r, t), []).append(fact)

        body = (("?x", "r", "?z"), ("?z", "p", "?y"))
        calls = {"n": 0}
        real_unify = symbolic._unify

        def counting_unify(pattern, triple, binding):  # noqa: ANN001, ANN202
            calls["n"] += 1
            return real_unify(pattern, triple, binding)

        with unittest.mock.patch.object(symbolic, "_unify", counting_unify):
            first = next(iter(RuleEngine._join(body, idx_all, idx_subj, idx_obj)))

        self.assertEqual(first["?x"], "x0")
        # One unification per body atom to reach the first row. Materialising the
        # join first would cost ~2n (600) before yielding anything.
        self.assertLessEqual(calls["n"], 2 * len(body))

    def test_join_preserves_binding_order(self) -> None:
        # The order matters: materialise() records the FIRST derivation it sees
        # for each fact, so a reordered join would change the proof trees.
        facts = [
            ("x0", "r", "z0"),
            ("x1", "r", "z1"),
            ("z0", "p", "y0"),
            ("z1", "p", "y1"),
        ]
        idx_all: dict[str, list[tuple[str, str, str]]] = {}
        idx_subj: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        idx_obj: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for fact in facts:
            h, r, t = fact
            idx_all.setdefault(r, []).append(fact)
            idx_subj.setdefault((r, h), []).append(fact)
            idx_obj.setdefault((r, t), []).append(fact)

        body = (("?x", "r", "?z"), ("?z", "p", "?y"))
        rows = [
            (b["?x"], b["?z"], b["?y"]) for b in RuleEngine._join(body, idx_all, idx_subj, idx_obj)
        ]
        self.assertEqual(rows, [("x0", "z0", "y0"), ("x1", "z1", "y1")])

    def test_join_order_matches_a_level_by_level_reference(self) -> None:
        # The claim is that yielding depth-first emits bindings in the order the
        # level-by-level version produced, so materialise() keeps recording the
        # same first derivation and proof trees do not move. The case above is
        # too thin to show that: with fan-out of one, every sane order agrees.
        # Here each atom fans out, and a three-atom body makes depth-first and
        # breadth-first genuinely disagree unless the property holds.
        facts: list[tuple[str, str, str]] = []
        for i in range(3):
            for j in range(3):
                facts.append((f"x{i}", "r", f"z{i}{j}"))
                for k in range(2):
                    facts.append((f"z{i}{j}", "p", f"w{i}{j}{k}"))
                    facts.append((f"w{i}{j}{k}", "q", f"y{i}{j}{k}"))
        idx_all: dict[str, list[tuple[str, str, str]]] = {}
        idx_subj: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        idx_obj: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for fact in facts:
            h, r, t = fact
            idx_all.setdefault(r, []).append(fact)
            idx_subj.setdefault((r, h), []).append(fact)
            idx_obj.setdefault((r, t), []).append(fact)

        body = (("?x", "r", "?z"), ("?z", "p", "?w"), ("?w", "q", "?y"))

        def level_by_level() -> list[dict[str, str]]:
            """The pre-generator implementation, kept here as the reference."""
            bindings: list[dict[str, str]] = [{}]
            for s, r, o in body:
                nxt: list[dict[str, str]] = []
                for binding in bindings:
                    s_val = binding.get(s) if s.startswith("?") else s
                    o_val = binding.get(o) if o.startswith("?") else o
                    if s_val is not None:
                        candidates = idx_subj.get((r, s_val), [])
                    elif o_val is not None:
                        candidates = idx_obj.get((r, o_val), [])
                    else:
                        candidates = idx_all.get(r, [])
                    for fact in candidates:
                        merged = symbolic._unify((s, r, o), fact, binding)
                        if merged is not None:
                            nxt.append(merged)
                bindings = nxt
                if not bindings:
                    break
            return bindings

        expected = level_by_level()
        actual = list(RuleEngine._join(body, idx_all, idx_subj, idx_obj))
        self.assertEqual(len(actual), 18, "the fixture should produce a real fan-out")
        self.assertEqual(actual, expected)

    def test_add_rule_permissive_on_untyped_ontology(self) -> None:
        # An untyped/empty ontology constrains nothing: every relation defaults to
        # domain/range {"*"}, so any range-restricted rule must be accepted.
        engine = RuleEngine(Ontology())
        rule = Rule("anything", (("?x", "p", "?y"),), ("?x", "anc", "?y"))
        self.assertTrue(engine.add_rule(rule))

    def test_derivation_and_entailment_check_are_frozen(self) -> None:
        d = Derivation(("a", "p", "b"), "rule1", (("a", "p", "b"),))
        with self.assertRaises(FrozenInstanceError):
            d.rule = "other"  # type: ignore[misc]
        chk = EntailmentCheck(entailed=True, rule="rule1", grounded=True, proof=("line1",))
        with self.assertRaises(FrozenInstanceError):
            chk.entailed = False  # type: ignore[misc]

    def test_derivations_sorted_and_filtered_by_grounded(self) -> None:
        rules = [
            Rule("anc_base", (("?x", "p", "?y"),), ("?x", "anc", "?y")),
            Rule("anc_step", (("?x", "anc", "?z"), ("?z", "p", "?y")), ("?x", "anc", "?y")),
        ]
        engine, g = _engine(rules)
        for a, b in [("a", "b"), ("b", "c")]:
            g.add_node(a, "T")
            g.add_node(b, "T")
            g.add_edge(a, "p", b)
        engine.materialise(g)

        grounded = engine.derivations()
        self.assertTrue(grounded)
        keys = [(d.triple, d.rule) for d in grounded]
        self.assertEqual(keys, sorted(keys))

        # Inject a dangling derivation referencing an unknown rule.
        bad_triple = ("a", "anc", "z_bad")
        engine._closure.add(bad_triple)
        engine._derivations[bad_triple] = Derivation(bad_triple, "unknown_rule", ())

        only_grounded = engine.derivations(grounded_only=True)
        self.assertNotIn(bad_triple, [d.triple for d in only_grounded])

        all_derivs = engine.derivations(grounded_only=False)
        self.assertIn(bad_triple, [d.triple for d in all_derivs])
        self.assertEqual(
            [(d.triple, d.rule) for d in all_derivs],
            sorted([(d.triple, d.rule) for d in all_derivs]),
        )

    def test_entails_base_derived_axiom_ungrounded_and_missing(self) -> None:
        rule = Rule("anc_base", (("?x", "p", "?y"),), ("?x", "anc", "?y"))
        engine, g = _engine([rule], symmetric=True)
        g.add_node("a", "T")
        g.add_node("b", "T")
        g.add_edge("a", "p", "b")
        g.add_edge("a", "r", "b")
        engine.materialise(g)

        # Base fact
        base_chk = engine.entails("a", "p", "b")
        self.assertTrue(base_chk.entailed)
        self.assertIsNone(base_chk.rule)
        self.assertTrue(base_chk.grounded)
        self.assertEqual(base_chk.proof, tuple(engine.explain(("a", "p", "b"))))
        self.assertIn("FACT", base_chk.proof[0])

        # Rule-derived fact
        rule_chk = engine.entails("a", "anc", "b")
        self.assertTrue(rule_chk.entailed)
        self.assertEqual(rule_chk.rule, "anc_base")
        self.assertTrue(rule_chk.grounded)
        self.assertTrue(any("DERIVED" in line for line in rule_chk.proof))

        # Axiom-derived fact
        ax_chk = engine.entails("b", "r", "a")
        self.assertTrue(ax_chk.entailed)
        self.assertEqual(ax_chk.rule, "ax:sym:r")
        self.assertTrue(ax_chk.grounded)

        # Fact added directly via add_fact (stated)
        engine.add_fact(("b", "anc", "a"))
        added_chk = engine.entails("b", "anc", "a")
        self.assertTrue(added_chk.entailed)
        self.assertIsNone(added_chk.rule)
        self.assertTrue(added_chk.grounded)

        # Entailed in closure but ungrounded derivation
        dangling = ("a", "anc", "ghost")
        engine._closure.add(dangling)
        engine._derivations[dangling] = Derivation(dangling, "dangling_rule", ())
        ungrounded_chk = engine.entails("a", "anc", "ghost")
        self.assertTrue(ungrounded_chk.entailed)
        self.assertEqual(ungrounded_chk.rule, "dangling_rule")
        self.assertFalse(ungrounded_chk.grounded)
        self.assertEqual(ungrounded_chk.proof, tuple(engine.explain(dangling)))

        # Not entailed
        missing_chk = engine.entails("b", "p", "a")
        self.assertEqual(
            missing_chk,
            EntailmentCheck(entailed=False, rule=None, grounded=False, proof=()),
        )

    def test_save_and_load_rules_json_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path

        rules = [
            Rule("r1", (("?x", "p", "?y"),), ("?x", "anc", "?y")),
            Rule(
                "r2",
                (("?x", "p", "?z"), ("?z", "p", "?y")),
                ("?x", "anc", "?y"),
                distinct=(("?x", "?y"),),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            save_rules_json(rules, path)
            loaded = load_rules_json(path)
        self.assertEqual(loaded, rules)

    def test_unify_constants_repeated_vars_and_ontology_edge_cases(self) -> None:
        # Exercise constant & repeated-variable unification branches
        self.assertIsNone(symbolic._unify(("a", "r", "?y"), ("b", "r", "c"), {}))
        self.assertIsNone(symbolic._unify(("?x", "r", "?y"), ("b", "r", "c"), {"?x": "a"}))
        self.assertIsNone(symbolic._unify(("?x", "s", "?y"), ("a", "r", "c"), {}))
        self.assertIsNone(symbolic._unify(("?x", "?x", "?y"), ("a", "r", "c"), {}))
        self.assertIsNone(symbolic._unify(("?x", "?r", "?y"), ("a", "r", "c"), {"?r": "s"}))
        self.assertIsNone(symbolic._unify(("?x", "r", "d"), ("a", "r", "c"), {}))
        self.assertIsNone(symbolic._unify(("?x", "r", "?x"), ("a", "r", "c"), {}))
        self.assertIsNone(symbolic._unify(("?x", "?r", "?r"), ("a", "r", "c"), {}))
        self.assertIsNone(symbolic._unify(("?x", "r", "?y"), ("a", "r", "c"), {"?y": "d"}))
        bound = symbolic._unify(("?x", "?r", "?y"), ("a", "r", "c"), {})
        self.assertEqual(bound, {"?x": "a", "?r": "r", "?y": "c"})

        # Duplicate rule name, wildcard relation, unknown head relation, inverse_of axiom,
        # duplicate add_fact, and cyclic proof_is_grounded
        onto = Ontology()
        onto.add_node_type(NodeType("T"))
        onto.add_relation_type(
            RelationType("p", frozenset({"*"}), frozenset({"*"}), inverse_of="p_inv")
        )
        onto.add_relation_type(RelationType("p_inv", frozenset({"*"}), frozenset({"*"})))
        r = Rule("dup", (("?x", "p", "?y"),), ("?x", "p_inv", "?y"))
        engine = RuleEngine(onto, [r])
        self.assertFalse(engine.add_rule(r))
        bad_head = Rule("bad_head", (("?x", "p", "?y"),), ("?x", "unk", "?y"))
        self.assertFalse(engine.add_rule(bad_head))

        g = WorldGraph()
        g.add_node("a", "T")
        g.add_node("b", "T")
        g.add_edge("a", "p", "b")
        engine.materialise(g)
        self.assertIn(("b", "p_inv", "a"), engine.closure)
        self.assertFalse(engine.add_fact(("a", "p", "b")))

        # Cyclic derivation guard in proof_is_grounded
        t1, t2 = ("a", "p", "b"), ("b", "p_inv", "a")
        engine._derivations[t1] = Derivation(t1, "ax:inv:p:p_inv", (t2,))
        engine._derivations[t2] = Derivation(t2, "ax:inv:p:p_inv", (t1,))
        self.assertTrue(engine.proof_is_grounded(t1))


if __name__ == "__main__":
    unittest.main()
