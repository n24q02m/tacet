"""Regression test for the ontology-preservation invariant under write-back.

Theorem 1 (ontology-preservation) holds *under the premise* that the graph
stays ontology-consistent. The distillation loop writes teacher facts back into
the graph; if a written-back fact introduces a brand-new endpoint, that endpoint
must be typed from the relation's declared schema, otherwise it defaults to the
catch-all ``Entity`` type and a perfectly valid rule then derives a
type-violating fact into the closure (breaking the premise the system is
supposed to maintain by construction).

This test pins that the write-back path types new endpoints from the relation
schema, so a query that triggers a teacher write-back of a fresh entity leaves
the graph ontology-consistent.
"""

from __future__ import annotations

import unittest

from tacet.cascade.router import TACET
from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule
from tacet.llm.teacher import CallableTeacher
from tacet.serve.config import CascadeConfig, KGEConfig


class TestWriteBackOntologyPreservation(unittest.TestCase):
    def _setup(self) -> tuple[TACET, WorldGraph, Ontology]:
        person = frozenset({"Person"})
        onto = (
            Ontology()
            .add_node_type(NodeType("Person"))
            .add_relation_type(RelationType("parent_of", person, person))
            .add_relation_type(RelationType("ancestor_of", person, person))
        )
        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("bob", "Person")
        g.add_edge("alice", "parent_of", "bob")
        # a valid (subset-checked) rule: ancestor_of(x, y) <= parent_of(x, y)
        rule = Rule(name="anc", body=(("?x", "parent_of", "?y"),), head=("?x", "ancestor_of", "?y"))
        # teacher answers (bob, parent_of) with a BRAND-NEW entity
        teacher = CallableTeacher(
            lambda h, r: ["mystery_x"] if (h, r) == ("bob", "parent_of") else []
        )
        ak = TACET(
            g,
            onto,
            teacher,
            rules=[rule],
            config=CascadeConfig(
                kge=KGEConfig(epochs=1),
                distillation=True,
                write_back=True,
                rule_synthesis=False,
            ),
        )
        return ak, g, onto

    def test_new_writeback_endpoint_is_typed_from_relation_schema(self) -> None:
        ak, g, _ = self._setup()
        # Tier-1 abstains on (bob, parent_of); the teacher answers and the fact
        # (bob, parent_of, mystery_x) is written back.
        ak.ask("bob", "parent_of")
        node = g.node("mystery_x")
        self.assertIsNotNone(node, "write-back must create the new endpoint")
        self.assertEqual(
            node.type,
            "Person",
            "new endpoint must inherit the relation's declared range type, not 'Entity'",
        )

    def test_writeback_keeps_graph_ontology_consistent(self) -> None:
        ak, g, onto = self._setup()
        ak.ask("bob", "parent_of")
        self.assertEqual(
            onto.validate(g),
            [],
            "teacher write-back must not introduce a type-violating fact",
        )

    def test_closure_stays_ontology_consistent_after_rule_fires(self) -> None:
        ak, g, onto = self._setup()
        ak.ask("bob", "parent_of")
        # materialise so the given rule fires on the written-back fact
        ak.engine.materialise(g)
        self.assertEqual(onto.validate(g), [])


class TestWriteBackInducedMultiType(unittest.TestCase):
    """The default path is a *schema-free* ``Ontology.induce``, which learns
    domain/range as the SET of observed endpoint types. A relation that connects
    more than one type (e.g. ``located_in`` over City->Country and
    Building->City) therefore has a multi-type domain/range. The write-back path
    must still type a brand-new endpoint from such a set, or it defaults to the
    catch-all ``Entity`` and the closure becomes ontology-inconsistent — exactly
    the failure mode the single-type test cannot catch.
    """

    def _setup_induced(self) -> tuple[TACET, WorldGraph, Ontology]:
        g = WorldGraph()
        g.add_node("paris", "City")
        g.add_node("france", "Country")
        g.add_node("europe", "Continent")
        g.add_node("eiffel", "Building")
        g.add_node("arch", "Building")  # a Building with no located_in yet
        g.add_edge("paris", "located_in", "france")  # City -> Country
        g.add_edge("paris", "located_in", "europe")  # City -> Continent (=> non-functional)
        g.add_edge("eiffel", "located_in", "paris")  # Building -> City
        onto = Ontology.induce(g)
        # teacher answers (arch, located_in) with a BRAND-NEW endpoint
        teacher = CallableTeacher(
            lambda h, r: ["mystery_place"] if (h, r) == ("arch", "located_in") else []
        )
        ak = TACET(
            g,
            onto,
            teacher,
            config=CascadeConfig(
                kge=KGEConfig(epochs=1),
                distillation=True,
                write_back=True,
                rule_synthesis=False,
            ),
        )
        return ak, g, onto

    def test_induced_multitype_endpoint_is_typed_from_the_set(self) -> None:
        ak, g, onto = self._setup_induced()
        rt = onto.relation("located_in")
        # precondition: induce really produced a multi-type range (else this test
        # would silently degenerate to the single-type case).
        self.assertGreater(len({t for t in rt.range if t != "*"}), 1, "expected multi-type range")
        ak.ask("arch", "located_in")
        node = g.node("mystery_place")
        self.assertIsNotNone(node, "write-back must create the new endpoint")
        self.assertNotEqual(node.type, "Entity", "must not default to the catch-all type")
        self.assertIn(
            node.type,
            set(rt.range),
            "new endpoint must take a type the relation's declared range admits",
        )

    def test_induced_multitype_keeps_graph_consistent(self) -> None:
        ak, g, onto = self._setup_induced()
        ak.ask("arch", "located_in")
        self.assertEqual(
            onto.validate(g),
            [],
            "multi-type write-back must not introduce a type-violating fact",
        )


class TestWriteBackSelfLoop(unittest.TestCase):
    """A teacher write-back can be a self-loop r(x, x). The single endpoint then
    has to satisfy BOTH the relation's domain and its range, so it must be typed
    from their intersection (left untyped only when they are disjoint, which is a
    genuine ill-typing the validator should surface).
    """

    def test_self_loop_types_from_domain_range_intersection(self) -> None:
        onto = (
            Ontology()
            .add_node_type(NodeType("A"))
            .add_node_type(NodeType("B"))
            .add_node_type(NodeType("C"))
            .add_relation_type(RelationType("rel", frozenset({"A", "B"}), frozenset({"B", "C"})))
        )
        g = WorldGraph()
        teacher = CallableTeacher(lambda h, r: ["x"] if (h, r) == ("x", "rel") else [])
        ak = TACET(
            g,
            onto,
            teacher,
            config=CascadeConfig(
                kge=KGEConfig(epochs=1),
                distillation=True,
                write_back=True,
                rule_synthesis=False,
            ),
        )
        ak.ask("x", "rel")
        node = g.node("x")
        self.assertIsNotNone(node)
        # {A,B} ∩ {B,C} = {B}: the only type consistent with both ends of r(x,x)
        self.assertEqual(node.type, "B")
        self.assertEqual(onto.validate(g), [])


if __name__ == "__main__":
    unittest.main()
