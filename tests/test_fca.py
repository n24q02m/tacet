"""Tests for the Formal Concept Analysis module (G1.4)."""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.distill.fca import FormalContext, summarise_lattice


def _toy_graph() -> WorldGraph:
    """Mini KG: 3 people, 3 companies with a clear lattice pattern."""
    g = WorldGraph(name="toy-fca")
    for s, r, t in [
        ("alice", "works_at", "acme"),
        ("bob", "works_at", "acme"),
        ("carol", "works_at", "acme"),
        ("bob", "works_at", "zoot"),
        ("carol", "works_at", "zoot"),
        ("alice", "speaks", "en"),
        ("bob", "speaks", "en"),
        ("carol", "speaks", "fr"),
    ]:
        g.add_edge(s, r, t)
    return g


class TestFormalContext(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = FormalContext.from_graph(_toy_graph(), min_support=1)

    def test_context_built(self) -> None:
        # objects = {alice, bob, carol}; attributes = the 4 distinct
        # (relation, tail) pairs: (works_at, acme), (works_at, zoot),
        # (speaks, en), (speaks, fr).
        self.assertEqual(set(self.ctx.objects), {"alice", "bob", "carol"})
        self.assertEqual(len(self.ctx.attributes), 4)

    def test_galois_closure_idempotent(self) -> None:
        # ((A')')' = A' for any extent A
        a = frozenset(["alice", "bob"])
        intent = self.ctx.attrs_of(a)
        ext, intent2 = self.ctx.closure_of_intent(intent)
        intent3 = self.ctx.attrs_of(ext)
        self.assertEqual(intent2, intent3)

    def test_concept_count_in_expected_range(self) -> None:
        # For the toy graph above (3 obj × 5 attr) we expect 5-9 concepts
        # (top + bottom + a handful of non-trivial ones).
        cs = self.ctx.concepts()
        self.assertGreaterEqual(len(cs), 4)
        self.assertLessEqual(len(cs), 16)  # safety upper bound

    def test_extents_form_set_lattice(self) -> None:
        # The full extent (all objects) must be present as a concept
        # (the top of the lattice).
        cs = self.ctx.concepts()
        extents = [ext for ext, _ in cs]
        self.assertIn(frozenset(self.ctx.objects), extents)

    def test_min_support_filter_removes_rare_attrs(self) -> None:
        # speaks/fr only covers carol → filtered if min_support=2.
        ctx2 = FormalContext.from_graph(_toy_graph(), min_support=2)
        fr_attr = ("speaks", "fr")
        self.assertNotIn(fr_attr, ctx2.attributes)
        en_attr = ("speaks", "en")
        self.assertIn(en_attr, ctx2.attributes)

    def test_summarise_returns_expected_keys(self) -> None:
        s = summarise_lattice(self.ctx)
        for key in ("n_objects", "n_attributes", "n_concepts", "n_non_trivial"):
            self.assertIn(key, s)
        self.assertEqual(s["n_objects"], 3)


if __name__ == "__main__":
    unittest.main()
