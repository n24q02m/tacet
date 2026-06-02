"""Rule-mining precision evaluation (proposed / installed / world-correct).

A reviewer asked us to quantify the miner's over-production: it forms many
candidate bodies, the confidence/support threshold installs a subset, and only
*some* of those installed rules are actually true in the world. This test pins
the two new instruments:

* ``rule_world_precision(rule, ground_truth_graph)`` -- the fraction of (x, y)
  pairs satisfying a rule's body that also satisfy its head, measured over the
  FULL ground-truth graph (every entity, not only teacher-answered heads). A
  genuinely-valid composition scores ~1.0; a spurious co-occurrence scores low.
* ``mine_rules_with_stats`` -- surfaces the PROPOSED candidate count alongside
  the installed rules, without altering the mining behaviour.
"""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.core.symbolic import Rule
from tacet.distill.distill import mine_rules, mine_rules_with_stats
from tacet.eval.rule_precision import rule_world_precision


def _family_graph() -> WorldGraph:
    """A grandparent graph with a genuine composition and a spurious overlap.

    Genuine: ``grandparent <= parent . parent`` -- exactly the 2-hop closure of
    ``parent_of``, so its world precision is 1.0.

    Spurious: every grandparent in this toy also happens to ``lives_in`` a city
    that the grandchild lives in (a co-occurrence with no causal meaning); the
    1-hop rule ``grandparent <= ~lives_in . lives_in`` (people sharing a city)
    over-fires to many non-grandparent pairs, so its world precision is low.
    """
    g = WorldGraph(name="family-toy")
    # 6 independent grandparent chains: gp_i -> p_i -> c_i.
    for i in range(6):
        gp, p, c = f"gp{i}", f"p{i}", f"c{i}"
        g.add_edge(gp, "parent_of", p)
        g.add_edge(p, "parent_of", c)
        g.add_edge(gp, "grandparent_of", c)  # the gold head edges

    # City co-occurrence: pack everyone into two shared cities so that the
    # "same city" rule binds many (x, y) pairs but only a handful are
    # grandparent pairs -> low precision for the spurious rule.
    members = [f"gp{i}" for i in range(6)] + [f"c{i}" for i in range(6)]
    for idx, person in enumerate(members):
        g.add_edge(person, "lives_in", f"city_{idx % 2}")
    return g


class TestRuleWorldPrecision(unittest.TestCase):
    def setUp(self) -> None:
        self.g = _family_graph()
        # genuine: parent_of . parent_of => grandparent_of
        self.valid = Rule(
            name="syn:grandparent_of<=parent_of.parent_of",
            body=(("?x", "parent_of", "?z"), ("?z", "parent_of", "?y")),
            head=("?x", "grandparent_of", "?y"),
            distinct=(("?x", "?y"),),
        )
        # spurious: shared-city co-occurrence => grandparent_of
        self.spurious = Rule(
            name="syn:grandparent_of<=lives_in.~lives_in",
            body=(("?x", "lives_in", "?z"), ("?y", "lives_in", "?z")),
            head=("?x", "grandparent_of", "?y"),
            distinct=(("?x", "?y"),),
        )

    def test_valid_rule_scores_high(self) -> None:
        prec = rule_world_precision(self.valid, self.g)
        self.assertGreaterEqual(prec, 0.9)
        self.assertEqual(prec, 1.0)

    def test_spurious_rule_scores_low(self) -> None:
        prec = rule_world_precision(self.spurious, self.g)
        self.assertLess(prec, 0.5)

    def test_precision_in_unit_interval(self) -> None:
        for rule in (self.valid, self.spurious):
            prec = rule_world_precision(rule, self.g)
            self.assertGreaterEqual(prec, 0.0)
            self.assertLessEqual(prec, 1.0)


class TestProposedVsInstalled(unittest.TestCase):
    def test_proposed_at_least_installed(self) -> None:
        g = _family_graph()
        teacher_facts = {(f"gp{i}", "grandparent_of", f"c{i}") for i in range(6)}
        heads = {f"gp{i}" for i in range(6)}
        rules, n_proposed = mine_rules_with_stats(
            g,
            teacher_facts,
            "grandparent_of",
            min_confidence=0.95,
            min_support=3,
            complete_heads=heads,
            allowed_body={"parent_of", "lives_in"},
        )
        self.assertGreaterEqual(n_proposed, len(rules))
        self.assertGreater(n_proposed, 0)

    def test_stats_wrapper_matches_plain_mine_rules(self) -> None:
        """The wrapper must not change which rules are installed."""
        g = _family_graph()
        teacher_facts = {(f"gp{i}", "grandparent_of", f"c{i}") for i in range(6)}
        heads = {f"gp{i}" for i in range(6)}
        kwargs = dict(
            min_confidence=0.95,
            min_support=3,
            complete_heads=heads,
            allowed_body={"parent_of", "lives_in"},
        )
        plain = mine_rules(g, teacher_facts, "grandparent_of", **kwargs)
        rules, _ = mine_rules_with_stats(g, teacher_facts, "grandparent_of", **kwargs)
        self.assertEqual([m.rule.name for m in plain], [m.rule.name for m in rules])

    def test_genuine_composition_is_installed_and_world_correct(self) -> None:
        g = _family_graph()
        teacher_facts = {(f"gp{i}", "grandparent_of", f"c{i}") for i in range(6)}
        heads = {f"gp{i}" for i in range(6)}
        rules, n_proposed = mine_rules_with_stats(
            g,
            teacher_facts,
            "grandparent_of",
            min_confidence=0.95,
            min_support=3,
            complete_heads=heads,
            allowed_body={"parent_of"},
        )
        names = {m.rule.name for m in rules}
        self.assertIn("syn:grandparent_of<=parent_of.parent_of", names)
        installed = next(
            m for m in rules if m.rule.name == "syn:grandparent_of<=parent_of.parent_of"
        )
        self.assertEqual(rule_world_precision(installed.rule, g), 1.0)
        self.assertGreaterEqual(n_proposed, len(rules))


if __name__ == "__main__":
    unittest.main()
