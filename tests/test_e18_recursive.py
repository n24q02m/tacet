"""E18 — genuinely recursive targets: the registered boundary of E16's ban.

The E16 ledger entry (2026-07-24, written before the result) states the
limitation: forbidding the target in its own length-2 body "forecloses
genuinely recursive targets (``ancestor <= ancestor.parent``)", so a POSITIVE
E16 result licenses "safe on non-recursive targets", never "always turn it
off". E18 tests exactly that case at the miner level: on a target defined in
terms of itself, the true rule NEEDS the target in its body, so E16's flag
must destroy learnability, and the refined guard (ban only the pure
all-target body, keep mixed target-base legs) must separate the sound
recursive rule from unsound pure self-loop shortcuts on non-transitive
recursive shapes.
"""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.distill.distill import mine_rules, mine_rules_with_stats


def _dag_graph() -> WorldGraph:
    """A small DAG: base edges ``parent`` point forward, so ``anc`` (the
    transitive closure) is a genuinely recursive relation: anc(x, y) iff
    y is reachable from x. Write-back anc edges are present, exactly as the
    cascade would have written them, so all candidate shapes are grounded."""
    g = WorldGraph()
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3), (3, 5), (5, 6), (6, 7)]
    anc: set[tuple[int, int]] = set()
    for a, b in edges:
        g.add_edge(f"n{a}", "parent", f"n{b}")
    # transitive closure over the DAG
    changed = True
    closure: dict[int, set[int]] = {i: set() for i in range(8)}
    for a, b in edges:
        closure[a].add(b)
    while changed:
        changed = False
        for a in range(8):
            for b in list(closure[a]):
                for c in closure[b]:
                    if c not in closure[a]:
                        closure[a].add(c)
                        changed = True
    for a, bs in closure.items():
        for b in bs:
            anc.add((a, b))
            g.add_edge(f"n{a}", "anc", f"n{b}")
    return g


class TestRecursiveTargetForeclosure(unittest.TestCase):
    """E16's ban forecloses the true rule on a genuinely recursive target."""

    KW = dict(min_confidence=0.9, min_support=2, allowed_body={"parent"})

    def test_true_recursive_rule_needs_target_in_body(self) -> None:
        # The gold recursive rule anc <= parent.anc (and anc <= anc.parent)
        # exists in the data: mining without the ban finds at least one rule
        # whose body mixes the target with the base relation.
        rules = mine_rules(_dag_graph(), set(), "anc", allow_target_in_body=True, **self.KW)
        mixed = [
            r
            for r in rules
            if any(rel == "anc" for _s, rel, _o in r.rule.body)
            and any(rel == "parent" for _s, rel, _o in r.rule.body)
        ]
        self.assertTrue(mixed, "no mixed recursive candidate installed at all")

    def test_e16_ban_destroys_every_recursive_candidate(self) -> None:
        # With E16's flag, NO candidate may mention the target in its body —
        # so the true recursive rule is unlearnable by construction.
        rules = mine_rules(_dag_graph(), set(), "anc", allow_target_in_body=False, **self.KW)
        for r in rules:
            self.assertNotIn("anc", {rel for _s, rel, _o in r.rule.body})

    def test_refined_guard_keeps_mixed_and_drops_pure_self_loop(self) -> None:
        graph = _dag_graph()
        kw = dict(min_confidence=0.0, min_support=1, allowed_body={"parent"})
        refined = mine_rules(
            graph,
            set(),
            "anc",
            allow_target_in_body=True,
            forbid_target_self_loop=True,
            **kw,
        )
        for r in refined:
            rels = [rel for _s, rel, _o in r.rule.body]
            self.assertFalse(
                len(rels) == 2 and rels[0] == "anc" and rels[1] == "anc",
                f"pure self-loop survived the refined guard: {r.rule.name}",
            )
        names = {r.rule.name for r in refined}
        mixed_recursive = [n for n in names if "anc" in n.split("<=", 1)[1]]
        self.assertTrue(mixed_recursive, "refined guard lost the recursive rule too")


class TestRefinedGuardEquivalence(unittest.TestCase):
    """The refined guard must differ from the published behaviour ONLY by the
    pure all-target candidates, and must dominate the E16 ban on candidates."""

    def test_only_pure_self_loops_are_removed(self) -> None:
        graph = _dag_graph()
        kw = dict(min_confidence=0.0, min_support=1)
        default = mine_rules(graph, set(), "anc", **kw)
        refined = mine_rules(graph, set(), "anc", forbid_target_self_loop=True, **kw)
        dropped = {r.rule.name for r in default} - {r.rule.name for r in refined}
        for name in dropped:
            body = name.split("<=", 1)[1]
            legs = body.split(".")
            self.assertEqual(
                {leg.lstrip("~") for leg in legs},
                {"anc"},
                f"dropped a non-pure candidate: {name}",
            )

    def test_guard_is_stricter_than_e16_never_looser(self) -> None:
        graph = _dag_graph()
        kw = dict(min_confidence=0.0, min_support=1)
        e16 = mine_rules(graph, set(), "anc", allow_target_in_body=False, **kw)
        refined = mine_rules(graph, set(), "anc", forbid_target_self_loop=True, **kw)
        e16_names = {r.rule.name for r in e16}
        refined_names = {r.rule.name for r in refined}
        self.assertEqual(e16_names & refined_names, e16_names)

    def test_default_replay_unchanged_by_flag_default(self) -> None:
        graph = _dag_graph()
        kw = dict(min_confidence=0.5, min_support=2)
        implicit = mine_rules(graph, set(), "anc", **kw)
        explicit = mine_rules(
            graph, set(), "anc", allow_target_in_body=True, forbid_target_self_loop=False, **kw
        )
        self.assertEqual(
            [(r.rule.name, r.confidence, r.support) for r in implicit],
            [(r.rule.name, r.confidence, r.support) for r in explicit],
        )


class TestStatsSurfaceGuard(unittest.TestCase):
    def test_proposed_count_drops_monotonically(self) -> None:
        graph = _dag_graph()
        kw = dict(min_confidence=0.0, min_support=1)
        _, n_default = mine_rules_with_stats(graph, set(), "anc", **kw)
        _, n_refined = mine_rules_with_stats(
            graph, set(), "anc", forbid_target_self_loop=True, **kw
        )
        _, n_e16 = mine_rules_with_stats(graph, set(), "anc", allow_target_in_body=False, **kw)
        self.assertLessEqual(n_refined, n_default)
        self.assertLessEqual(n_e16, n_refined)


if __name__ == "__main__":
    unittest.main()
