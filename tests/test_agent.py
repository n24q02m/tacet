"""Tests for the STRIPS planner and agent concepts."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.experimental.agent import (
    Action,
    Goal,
    GroundedAction,
    Plan,
    Planner,
    _apply,
    _goal_satisfied,
    _ground_action,
)


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.onto = Ontology()
        self.onto.add_node_type(NodeType("Person"))
        self.onto.add_node_type(NodeType("Place"))
        self.onto.add_relation_type(RelationType("at", frozenset({"Person"}), frozenset({"Place"})))
        self.onto.add_relation_type(
            RelationType("neighbor", frozenset({"Place"}), frozenset({"Place"}), symmetric=True)
        )
        self.onto.add_relation_type(
            RelationType("path", frozenset({"Place"}), frozenset({"Place"}), transitive=True)
        )

    def test_action_and_grounding(self) -> None:
        # Move action: if ?p is at ?from, and ?from and ?to are neighbors,
        # ?p can move to ?to.
        move = Action(
            name="move",
            preconditions=(("?p", "at", "?from"), ("?from", "neighbor", "?to")),
            add=(("?p", "at", "?to"),),
            remove=(("?p", "at", "?from"),),
        )

        facts = {
            ("alice", "at", "kitchen"),
            ("kitchen", "neighbor", "hall"),
            ("hall", "neighbor", "kitchen"),  # neighbor is symmetric but provided explicitly
        }

        grounded = list(_ground_action(move, facts))
        self.assertEqual(len(grounded), 1)
        ga = grounded[0]
        self.assertEqual(ga.action.name, "move")
        bindings = dict(ga.bindings)
        self.assertEqual(bindings["?p"], "alice")
        self.assertEqual(bindings["?from"], "kitchen")
        self.assertEqual(bindings["?to"], "hall")
        self.assertEqual(ga.signature(), "move(?from=kitchen,?p=alice,?to=hall)")

    def test_apply_action(self) -> None:
        move = Action(
            name="move",
            preconditions=(("?p", "at", "?from"), ("?from", "neighbor", "?to")),
            add=(("?p", "at", "?to"),),
            remove=(("?p", "at", "?from"),),
        )
        ga = GroundedAction(move, (("?from", "kitchen"), ("?p", "alice"), ("?to", "hall")))

        facts = frozenset([("alice", "at", "kitchen"), ("kitchen", "neighbor", "hall")])

        next_facts = _apply(facts, ga)
        self.assertIn(("alice", "at", "hall"), next_facts)
        self.assertNotIn(("alice", "at", "kitchen"), next_facts)
        self.assertIn(("kitchen", "neighbor", "hall"), next_facts)

    def test_goal_satisfied(self) -> None:
        goal = Goal((("alice", "at", "hall"),))

        self.assertFalse(_goal_satisfied({("alice", "at", "kitchen")}, goal))
        self.assertTrue(_goal_satisfied({("alice", "at", "hall")}, goal))

        # Test with variables in goal
        goal_vars = Goal((("?who", "at", "hall"),))
        self.assertTrue(_goal_satisfied({("alice", "at", "hall")}, goal_vars))

    def test_planner_simple(self) -> None:
        move = Action(
            name="move",
            preconditions=(("?p", "at", "?from"), ("?from", "neighbor", "?to")),
            add=(("?p", "at", "?to"),),
            remove=(("?p", "at", "?from"),),
        )
        planner = Planner([move], self.onto)

        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("kitchen", "Place")
        g.add_node("hall", "Place")
        g.add_edge("alice", "at", "kitchen")
        g.add_edge("kitchen", "neighbor", "hall")

        goal = Goal((("alice", "at", "hall"),))
        plan = planner.plan(g, goal)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].action.name, "move")
        self.assertEqual(dict(plan.actions[0].bindings)["?to"], "hall")

    def test_planner_multi_step(self) -> None:
        move = Action(
            name="move",
            preconditions=(("?p", "at", "?from"), ("?from", "neighbor", "?to")),
            add=(("?p", "at", "?to"),),
            remove=(("?p", "at", "?from"),),
        )
        planner = Planner([move], self.onto)

        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("kitchen", "Place")
        g.add_node("hall", "Place")
        g.add_node("bedroom", "Place")
        g.add_edge("alice", "at", "kitchen")
        g.add_edge("kitchen", "neighbor", "hall")
        g.add_edge("hall", "neighbor", "bedroom")

        goal = Goal((("alice", "at", "bedroom"),))
        plan = planner.plan(g, goal)

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 2)
        self.assertEqual(plan.actions[0].action.name, "move")
        self.assertEqual(plan.actions[1].action.name, "move")

    def test_planner_no_plan(self) -> None:
        move = Action(
            name="move",
            preconditions=(("?p", "at", "?from"), ("?from", "neighbor", "?to")),
            add=(("?p", "at", "?to"),),
            remove=(("?p", "at", "?from"),),
        )
        planner = Planner([move], self.onto)

        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("kitchen", "Place")
        g.add_node("bedroom", "Place")
        g.add_edge("alice", "at", "kitchen")
        # No neighbor edge to bedroom

        goal = Goal((("alice", "at", "bedroom"),))
        plan = planner.plan(g, goal)
        self.assertIsNone(plan)

    def test_planner_with_derived_triples(self) -> None:
        # Define a goal that requires a derived triple (transitive path)
        # Action: fly to any place.
        fly = Action(
            name="fly",
            preconditions=(("?p", "at", "?from"),),
            add=(("?p", "at", "?to"),),
            remove=(("?p", "at", "?from"),),
        )

        planner = Planner([fly], self.onto)

        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("london", "Place")
        g.add_node("paris", "Place")
        g.add_node("lyon", "Place")
        g.add_edge("alice", "at", "london")
        g.add_edge("paris", "path", "lyon")

        # In the original behavior (non-consistent bindings), this goal
        # is satisfied if Alice is at ANY place (?somewhere matches london)
        # AND some place (?somewhere matches paris) has a path to Lyon.
        # Since London and Paris are both in the graph entities, the planner
        # can satisfy this goal WITHOUT moving Alice to Paris, IF ?somewhere
        # matches differently for each triple.
        goal = Goal((("alice", "at", "?somewhere"), ("?somewhere", "path", "lyon")))

        plan = planner.plan(g, goal)
        self.assertIsNotNone(plan)
        # Expected: 0 actions because alice is already at london (?somewhere=london)
        # and paris has a path to lyon (?somewhere=paris).
        self.assertEqual(len(plan.actions), 0)

    def test_planner_already_satisfied(self) -> None:
        planner = Planner([], self.onto)
        g = WorldGraph()
        g.add_node("alice", "Person")
        g.add_node("kitchen", "Place")
        g.add_edge("alice", "at", "kitchen")

        goal = Goal((("alice", "at", "kitchen"),))
        plan = planner.plan(g, goal)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 0)
        self.assertEqual(plan.cost, 0.0)

    def test_plan_str(self) -> None:
        move = Action(name="move")
        ga = GroundedAction(move, (("?p", "alice"),))
        plan = Plan([ga], 1.0)
        self.assertEqual(str(plan), "move(?p=alice)")

        empty_plan = Plan([], 0.0)
        self.assertEqual(str(empty_plan), "(empty plan)")


if __name__ == "__main__":
    unittest.main()
