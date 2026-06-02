"""Regression test for the 2-hop composition rule the distiller must recover.

The paper's distillation-beats-caching claim rests on the miner synthesising the
*genuine base-relation composition* for a composed (2-hop) target relation — e.g.
``directors of the movies an actor starred in`` is

    ~starred_actors(x, z) & directed_by(z, y) => target(x, y)

an inverse-then-forward V-shape over the BASE relations. A prior run instead
mined a degenerate SELF-rule ``target(x,z) & target(z,y) => target(x,y)``
(``target∘target``); that happens only when the mining graph the miner sees
contains the written-back ``target`` edges but NOT the base relations, so the
sole candidate bodies are self-compositions. These tests pin both halves:

1. with the base edges present and whitelisted, the miner recovers the real
   inverse-path composition and does NOT emit the self-rule;
2. with ONLY target edges visible (base relations absent), the degenerate
   self-rule is the only thing recoverable — the exact failure mode to avoid.
"""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.distill.distill import mine_rules

TARGET = "q2_directors_of_acted"


def _toy_movie_graph() -> tuple[WorldGraph, set, set]:
    """3 actors, 6 movies, 2 directors over the MetaQA base-edge direction.

    Base edges (movie -> person), as in MetaQA: ``movie starred_actors actor``
    and ``movie directed_by director``. The composed target
    (actor -> director: "directors of movies the actor starred in") is
    ``~starred_actors . directed_by``.

    Returns the graph (base edges only), the set of composed ``target`` teacher
    facts, and the set of complete heads.
    """
    g = WorldGraph(name="toy-movies")
    # movie: (actor, director)
    movies = {
        "m1": ("a1", "d1"),
        "m2": ("a1", "d2"),
        "m3": ("a2", "d1"),
        "m4": ("a2", "d2"),
        "m5": ("a3", "d1"),
        "m6": ("a3", "d2"),
    }
    for mv, (actor, director) in movies.items():
        g.add_edge(mv, "starred_actors", actor)
        g.add_edge(mv, "directed_by", director)

    # composed gold: actor -> {directors of movies the actor starred in}
    actor_movies: dict[str, set[str]] = {}
    for mv, (actor, _d) in movies.items():
        actor_movies.setdefault(actor, set()).add(mv)
    teacher_facts: set[tuple[str, str, str]] = set()
    heads: set[str] = set()
    for actor, mvs in actor_movies.items():
        directors = {movies[m][1] for m in mvs}
        directors.discard(actor)
        for d in directors:
            teacher_facts.add((actor, TARGET, d))
        heads.add(actor)
    return g, teacher_facts, heads


class TestCompositionMining(unittest.TestCase):
    def test_recovers_inverse_path_composition_not_self_rule(self) -> None:
        g, teacher_facts, heads = _toy_movie_graph()
        rules = mine_rules(
            g,
            teacher_facts,
            TARGET,
            min_confidence=0.95,
            min_support=3,
            complete_heads=heads,
            allowed_body={"starred_actors", "directed_by"},
        )
        names = {m.rule.name for m in rules}
        # The genuine base-relation composition is recovered ...
        self.assertIn(f"syn:{TARGET}<=~starred_actors.directed_by", names)
        # ... and NO degenerate self-rule (target∘target in any leg orientation).
        for m in rules:
            for s, rel, o in m.rule.body:  # noqa: B007
                self.assertNotEqual(
                    rel,
                    TARGET,
                    f"degenerate self-rule mined: {m.rule.name} (body atom on {TARGET})",
                )

    def test_recovered_rule_is_exact(self) -> None:
        g, teacher_facts, heads = _toy_movie_graph()
        rules = mine_rules(
            g,
            teacher_facts,
            TARGET,
            min_confidence=0.95,
            min_support=3,
            complete_heads=heads,
            allowed_body={"starred_actors", "directed_by"},
        )
        # The composition is exact on this graph: confidence 1.0.
        top = next(m for m in rules if m.rule.name.endswith("~starred_actors.directed_by"))
        self.assertEqual(top.confidence, 1.0)
        self.assertEqual(
            top.rule.body,
            (("?z", "starred_actors", "?x"), ("?z", "directed_by", "?y")),
        )

    def test_without_base_relations_only_degenerate_self_rule_is_recoverable(self) -> None:
        """The exact failure mode: base edges absent => only target∘target.

        When the miner can see only the written-back ``target`` edges (no base
        relations in the graph / whitelist), the genuine composition is
        unrecoverable and the only candidate bodies are ``target``-on-``target``
        self-compositions — the degenerate rule the regression guards against.
        """
        g, teacher_facts, heads = _toy_movie_graph()
        # Drop every base edge: leave a graph with no base relations at all.
        bare = WorldGraph(name="no-base")
        for n in g.nodes:
            bare.add_node(n.id, n.type, **n.props)
        rules = mine_rules(
            bare,
            teacher_facts,
            TARGET,
            min_confidence=0.0,
            min_support=1,
            complete_heads=heads,
            allowed_body=set(),  # no base relations whitelisted
        )
        # Whatever is recoverable here is a self-rule (or nothing); the genuine
        # base composition can NOT appear because the base edges are gone.
        for m in rules:
            self.assertTrue(
                all(rel == TARGET for _s, rel, _o in m.rule.body),
                f"unexpected non-self body without base edges: {m.rule.name}",
            )
        self.assertNotIn(
            f"syn:{TARGET}<=~starred_actors.directed_by",
            {m.rule.name for m in rules},
        )


if __name__ == "__main__":
    unittest.main()
