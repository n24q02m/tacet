"""Regression tests for the installed-rule world-precision scoring graph (E11).

The controlled runner scores an installed rule's *world precision* against a
ground-truth graph. For a multi-hop composition that graph must materialise the
composed relation's TRUE edges for EVERY head — the full ``_compose_gold`` map —
not just the workload pool's teacher-answered heads. The pool is a shuffled,
truncated, ``max_answer``-filtered subset, so scoring against pool-only composed
edges leaves the head side sparse while the rule body fires across the whole KB:
the ratio collapses far below 1.0 even for a perfect oracle whose single
installed rule IS the target composition (paper ``\\oracleRuleMeanPrec = 1.000``).

All fixtures are TINY and SYNTHETIC: MetaQA is never loaded or run here.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_real_kg_amortization import COMPOSITIONS, _compose_gold  # noqa: E402
from run_real_kg_controlled import _scoring_graph, run_controlled  # noqa: E402

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.core.symbolic import Rule  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark  # noqa: E402
from tacet.eval.rule_precision import rule_world_precision  # noqa: E402

#: The registered 2-hop composition ``q2 := ~starred_actors . directed_by``
#: (directors of the movies an actor starred in).
_COMP = "directors_of_acted"
_SPEC = COMPOSITIONS[_COMP]
_REL = _SPEC["kg_relation"]


def _composition_kg() -> WorldGraph:
    """A KG whose latent composition the miner recovers, with a pool/KB gap.

    14 clean actors each starred in one movie with one director -> composed answer
    set size 1, so all 14 enter the workload pool. One HUB actor starred in 30
    movies with 30 distinct directors -> composed answer set size 30 > ``max_answer``
    (25), so the hub is DROPPED from the pool yet the rule body still fires on it.
    The composed relation therefore has 15 true heads but the pool covers only 14:
    scoring against pool-only composed edges under-counts the head side.
    """
    kg = WorldGraph(name="comp-toy")
    for i in range(14):
        kg.add_edge(f"CM{i}", "starred_actors", f"A{i}")
        kg.add_edge(f"CM{i}", "directed_by", f"CD{i}")
    for j in range(30):
        kg.add_edge(f"HM{j}", "starred_actors", "A_hub")
        kg.add_edge(f"HM{j}", "directed_by", f"HD{j}")
    return kg


def _composition_bench() -> MetaQABenchmark:
    kg = _composition_kg()
    return MetaQABenchmark(
        name="comp-toy",
        hop=1,  # the KB is hop-agnostic; run_controlled's ``hop`` arg drives routing
        split="test",
        kg=kg,
        questions=[],
        entities=set(kg.entities()),
        relations=kg.relations(),
    )


def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


def _correct_rule() -> Rule:
    """The genuine composition ``~starred_actors . directed_by => q2`` the miner installs."""
    return Rule(
        name="syn:composition",
        body=(("?z", "starred_actors", "?x"), ("?z", "directed_by", "?y")),
        head=("?x", _REL, "?y"),
        distinct=(("?x", "?y"),),
    )


# --------------------------------------------------------------- end-to-end (Test 2)
def test_controlled_hop2_reports_world_precision_one_for_correct_rule():
    """run_controlled (oracle, error_rate=0, low gamma) on a synthetic composition
    the miner recovers: the reported ``rule_world_precision`` must be exactly 1.0.

    Against the pool-only scoring graph this reads ~0.318 (14 pool heads / 44 body
    firings) even though the single installed rule is the exact target composition
    — the bug this fixes.
    """
    report = run_controlled(
        hop=2,
        split="test",
        limit=150,
        zipf_a=1.2,
        seed=0,
        composition=_COMP,
        oracle_error_rate=0.0,
        gamma=0.5,
        settings=_oracle_settings(),
        bench=_composition_bench(),
        verbose=False,
    )
    v = report["verdict"]
    # exactly the genuine inverse-then-forward composition is installed
    assert v["rule_installed"] is True
    assert v["synthesised_rules"] == [
        "syn:q2_directors_of_movies_acted_in_by<=~starred_actors.directed_by"
    ]
    # the world precision of that single installed rule is exactly 1.0 (mean over 1)
    assert v["rule_world_precision"] == 1.0
    assert v["rule_world_precision"] > 0.5


# ------------------------------------------------------ scoring-graph unit (Test 1)
def test_pool_limited_scoring_graph_degenerates_full_scores_one():
    """Pins the bug: the CORRECT composition rule scores far below 1.0 when the
    scoring graph materialises only the pool's composed edges, and exactly 1.0 when
    it materialises the full ``_compose_gold`` map over every head.
    """
    kg = _composition_kg()
    full_gold = _compose_gold(kg, _SPEC)  # 15 true composed heads (incl. the hub)
    pool_gold = {h: full_gold[h] for h in ("A0", "A1")}  # a couple of pool heads

    gt_pool = _scoring_graph(
        kg, hop=2, pool_gold={}, composed_gold=pool_gold, composed_relation=_REL
    )
    gt_full = _scoring_graph(
        kg, hop=2, pool_gold={}, composed_gold=full_gold, composed_relation=_REL
    )

    prec_pool = rule_world_precision(_correct_rule(), gt_pool)
    prec_full = rule_world_precision(_correct_rule(), gt_full)

    # the pool-limited scoring graph collapses the precision far below 1.0 (2 of 44
    # body firings land on a materialised head) -- the bug
    assert prec_pool < 0.5
    # the full composed gold scores the genuine composition at exactly 1.0
    assert prec_full == 1.0


# ------------------------------------------------ spurious rule scores low (Test 3)
def test_spurious_rule_scores_low_against_full_gold_graph():
    """A rule that is NOT world-correct scores well below 1.0 against the FULL gold
    graph -- so the guard is load-bearing: "a lower gamma installs more rules"
    cannot masquerade as a win when the extra rules are spurious.

    The spurious body drops the join, pairing every actor with every director
    (a cross product), so it over-fires massively: only 44 of its firings are true
    ``q2`` edges.
    """
    kg = _composition_kg()
    full_gold = _compose_gold(kg, _SPEC)
    gt_full = _scoring_graph(
        kg, hop=2, pool_gold={}, composed_gold=full_gold, composed_relation=_REL
    )

    spurious = Rule(
        name="syn:spurious-overfire",
        body=(("?z", "starred_actors", "?x"), ("?z2", "directed_by", "?y")),
        head=("?x", _REL, "?y"),
        distinct=(("?x", "?y"),),
    )
    prec = rule_world_precision(spurious, gt_full)
    assert prec < 0.5
    # the genuine composition, by contrast, is world-correct on the same graph
    assert rule_world_precision(_correct_rule(), gt_full) == 1.0


# ------------------------------------------------------- hop==1 unchanged (Test 4)
def test_hop1_scoring_graph_is_the_kb_unchanged():
    """hop==1 has no composed relation and its gold relations already live in the
    KB, so ``_scoring_graph`` adds only edges the KB already holds: the resulting
    graph's triple set equals the KB's (the pre-fix behaviour, exactly).
    """
    kg = WorldGraph(name="hop1-toy")
    for i in range(5):
        kg.add_edge(f"M{i}", "directed_by", f"D{i}")
        kg.add_edge(f"M{i}", "has_genre", "drama")
    # a pool gold map keyed "head\trelation" over a subset of real KB edges
    pool_gold = {f"M{i}\tdirected_by": frozenset({f"D{i}"}) for i in range(3)}

    gt = _scoring_graph(kg, hop=1, pool_gold=pool_gold, composed_gold=None, composed_relation=None)

    # materialising pool gold that already lives in the KB is a no-op -> gt == KB
    assert set(gt.triples()) == set(kg.triples())
