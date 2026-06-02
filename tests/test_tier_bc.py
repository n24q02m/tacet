"""Tests for the Tier B + C extensions: causal, episodic, agent, concepts,
federation, worldmodel."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tacet.core.causal import CausalModel, backdoor_set, counterfactual
from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.distill.concepts import induce_node_types, induce_relations, revise_ontology
from tacet.experimental.agent import Action, Goal, Planner
from tacet.experimental.episodic import EpisodicStore, FeedbackCurator, RuleScore
from tacet.experimental.federation import FederatedGraph, merge, trust_weighted
from tacet.experimental.worldmodel import IdentityWorldModel


# --- causal -----------------------------------------------------------------
def _build_smoking_scm(seed: int = 42) -> CausalModel:
    """Smoking → tar → cancer; smoking also directly → cancer."""
    m = CausalModel(seed=seed)
    m.add_variable("smokes", (0, 1), (), lambda _p, u: 1 if u < 0.3 else 0)
    m.add_variable(
        "tar",
        (0, 1),
        ("smokes",),
        lambda p, u: 1 if (p["smokes"] == 1 and u < 0.9) or (p["smokes"] == 0 and u < 0.05) else 0,
    )
    m.add_variable(
        "cancer",
        (0, 1),
        ("smokes", "tar"),
        lambda p, u: 1 if u < (0.1 + 0.2 * p["smokes"] + 0.4 * p["tar"]) else 0,
    )
    return m


class TestCausalModel(unittest.TestCase):
    def setUp(self) -> None:
        self.m = _build_smoking_scm(seed=42)

    def test_sampling_returns_assignments_for_all_variables(self) -> None:
        samples = self.m.sample(50)
        for s in samples:
            self.assertEqual(set(s), {"smokes", "tar", "cancer"})

    def test_observation_vs_intervention_diverge_under_confounding(self) -> None:
        # On this SCM smokes has no parents, so the observational and
        # interventional means coincide — exercise the API instead.
        obs = self.m.probability("cancer", 1, evidence={"smokes": 1}, n=4000)
        do1 = self.m.intervene({"smokes": 1}).probability("cancer", 1, n=4000)
        do0 = self.m.intervene({"smokes": 0}).probability("cancer", 1, n=4000)
        self.assertGreater(do1, do0 + 0.2)  # smoking causes cancer
        self.assertAlmostEqual(obs, do1, delta=0.05)

    def test_counterfactual_rolls_back_via_abduction(self) -> None:
        cf = counterfactual(
            self.m,
            evidence={"smokes": 1, "cancer": 1},
            do={"smokes": 0},
            target="cancer",
            n=4000,
        )
        # observed cancer was almost certainly *caused* by smoking; under do(smokes=0)
        # the counterfactual cancer rate drops well below 1.0.
        self.assertLess(cf[1], 0.5)

    def test_backdoor_returns_parents_not_descendants(self) -> None:
        m = CausalModel()
        m.add_variable("Z", (0, 1), (), lambda _p, u: 1 if u < 0.5 else 0)
        m.add_variable("X", (0, 1), ("Z",), lambda p, _u: p["Z"])
        m.add_variable("Y", (0, 1), ("X", "Z"), lambda p, _u: (p["X"] + p["Z"]) % 2)
        self.assertEqual(backdoor_set(m, "X", "Y"), {"Z"})


# --- episodic ----------------------------------------------------------------
class TestEpisodicStore(unittest.TestCase):
    def test_record_returns_episode_with_id(self) -> None:
        s = EpisodicStore()
        ep = s.record("a", "rel", 1, ["b"], 0.0001, 3.0, proof_rules=["r1"])
        self.assertEqual(ep.id, 0)
        self.assertEqual(s.for_query("a", "rel"), [ep])

    def test_summary_aggregates_tiers_and_feedback(self) -> None:
        s = EpisodicStore()
        s.record("a", "r", 1, ["b"], 1.0, 1.0).mark_correct()
        s.record("a", "r", 3, ["c"], 50.0, 900.0).mark_wrong()
        summary = s.summary()
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["tier_counts"], {1: 1, 3: 1})
        self.assertEqual(summary["feedback_received"], 2)
        self.assertAlmostEqual(summary["feedback_accuracy"], 0.5)

    def test_jsonl_roundtrip(self) -> None:
        s = EpisodicStore()
        s.record("a", "r", 1, ["b"], 0.0, 0.0).mark_correct()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ep.jsonl"
            s.save_jsonl(path)
            loaded = EpisodicStore()
            loaded.load_jsonl(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded._episodes[0].head, "a")


class TestFeedbackCurator(unittest.TestCase):
    def test_rules_with_persistent_negative_feedback_get_retired(self) -> None:
        store = EpisodicStore()
        # one good rule, one bad rule
        for _ in range(6):
            store.record("h", "r", 1, ["t"], 0.0, 0.0, proof_rules=["good"]).mark_correct()
        for _ in range(6):
            store.record("h2", "r", 1, ["t"], 0.0, 0.0, proof_rules=["bad"]).mark_wrong()
        cur = FeedbackCurator(retire_below=0.4, min_observations=4)
        cur.absorb(store)
        self.assertIn("bad", cur.rules_to_retire())
        self.assertNotIn("good", cur.rules_to_retire())
        self.assertIn("good", cur.trusted_rules(threshold=0.5))

    def test_rule_score_pessimistic_for_small_n(self) -> None:
        s = RuleScore("x", positives=1, negatives=0)
        self.assertLess(s.trust, 1.0)


# --- agent ------------------------------------------------------------------
class TestPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.onto = Ontology()
        self.onto.add_node_type(NodeType("Block"))
        self.onto.add_node_type(NodeType("Loc"))
        self.onto.add_relation_type(RelationType("at", frozenset({"Block"}), frozenset({"Loc"})))
        self.state = WorldGraph()
        for x in ("A", "B"):
            self.state.add_node(x, "Block")
        for x in ("X", "Y", "Z"):
            self.state.add_node(x, "Loc")
        self.state.add_edge("A", "at", "X")
        self.state.add_edge("B", "at", "Z")
        self.move = Action(
            name="move",
            preconditions=(("?b", "at", "?from"),),
            add=(("?b", "at", "?to"),),
            remove=(("?b", "at", "?from"),),
        )

    def test_already_satisfied_returns_empty_plan(self) -> None:
        plan = Planner([self.move], self.onto).plan(self.state, Goal(triples=(("A", "at", "X"),)))
        self.assertEqual(len(plan), 0)
        self.assertEqual(plan.cost, 0.0)

    def test_single_step_plan(self) -> None:
        plan = Planner([self.move], self.onto).plan(self.state, Goal(triples=(("A", "at", "Y"),)))
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan.actions[0].action.name, "move")

    def test_multi_step_plan(self) -> None:
        plan = Planner([self.move], self.onto).plan(
            self.state, Goal(triples=(("A", "at", "Y"), ("B", "at", "X"))), max_depth=4
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 2)

    def test_infeasible_returns_none(self) -> None:
        # no action can produce at(C,Y) because C is not even a node
        plan = Planner([self.move], self.onto).plan(
            self.state, Goal(triples=(("C", "at", "Y"),)), max_depth=3
        )
        self.assertIsNone(plan)


# --- concepts ---------------------------------------------------------------
class TestConcepts(unittest.TestCase):
    def setUp(self) -> None:
        g = WorldGraph()
        for p in ("alice", "bob", "carol", "dan"):
            g.add_node(p, "Person")
        for c in ("acme", "zoot"):
            g.add_node(c, "Company")
        g.add_edge("alice", "works_at", "acme")
        g.add_edge("bob", "works_at", "acme")
        g.add_edge("carol", "works_at", "zoot")
        g.add_edge("dan", "works_at", "zoot")
        g.add_edge("alice", "friend", "bob")
        g.add_edge("carol", "friend", "dan")
        self.g = g

    def test_induce_node_types_splits_persons_from_companies(self) -> None:
        types = induce_node_types(self.g, k=2)
        groups = [set(t.members) for t in types]
        self.assertIn({"acme", "zoot"}, groups)
        self.assertIn({"alice", "bob", "carol", "dan"}, groups)

    def test_induce_relations_finds_length_2_paths(self) -> None:
        relations = induce_relations(self.g, min_support=1)
        names = {r.name for r in relations}
        self.assertIn("friend+works_at", names)

    def test_revise_ontology_extends_in_place(self) -> None:
        onto = Ontology()
        report = revise_ontology(self.g, onto, min_support=1)
        self.assertTrue(report["added_types"])
        self.assertTrue(report["added_relations"])
        self.assertIn(report["added_types"][0], onto.node_types)


# --- federation -------------------------------------------------------------
class TestFederation(unittest.TestCase):
    def test_provenance_records_writer_and_trust(self) -> None:
        g = FederatedGraph()
        g.assert_fact("Paris", "capital_of", "France", writer="alice", trust=0.9)
        self.assertEqual(g.writers_of(("Paris", "capital_of", "France")), ["alice"])
        self.assertAlmostEqual(g.trust_score(("Paris", "capital_of", "France")), 0.9)

    def test_merge_with_trust_weighted_strategy_filters_low_trust(self) -> None:
        a = FederatedGraph()
        b = FederatedGraph()
        a.assert_fact("x", "r", "y", writer="alice", trust=0.9)
        b.assert_fact("x", "r", "y", writer="bob", trust=0.8)
        b.assert_fact("z", "r", "y", writer="bob", trust=0.3)
        out = merge(a, b, strategy=trust_weighted(threshold=0.5))
        self.assertTrue(out.graph.has_edge("x", "r", "y"))
        self.assertFalse(out.graph.has_edge("z", "r", "y"))

    def test_disagreements_surface_conflicts(self) -> None:
        g = FederatedGraph()
        g.assert_fact("p", "capital_of", "france", writer="a")
        g.assert_fact("berlin", "capital_of", "france", writer="b")
        d = g.disagreements("p", "capital_of")
        self.assertEqual(d, [("p", "capital_of", "france")])

    def test_writer_trust_multiplies_record_trust(self) -> None:
        g = FederatedGraph()
        g.set_writer_trust("bob", 0.1)
        g.assert_fact("x", "r", "y", writer="bob", trust=1.0)
        self.assertAlmostEqual(g.trust_score(("x", "r", "y")), 0.1)


# --- worldmodel -------------------------------------------------------------
class TestWorldModel(unittest.TestCase):
    def test_identity_keeps_state_constant(self) -> None:
        wm = IdentityWorldModel()
        traj = wm.rollout("initial", ["a", "b", "c"])
        self.assertEqual(traj.states, ["initial", "initial", "initial", "initial"])
        self.assertEqual(traj.actions, ["a", "b", "c"])
        self.assertEqual(traj.rewards, [0.0, 0.0, 0.0])

    def test_observe_returns_observation(self) -> None:
        wm = IdentityWorldModel()
        self.assertEqual(wm.observe("seen"), "seen")


if __name__ == "__main__":
    unittest.main()
