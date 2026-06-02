"""Simplified AMIE+ rule miner for comparison with TACET rule synthesis (G1.4).

AMIE+ (Galárraga et al. 2015) is a popular baseline for rule mining
on KGs.  Unlike ``tacet.concepts.induce_relations`` (length-2 path
mining + support threshold), AMIE+ computes **PCA confidence** —
confidence under the Partial Completeness Assumption — which measures:

> Among the (x, y) pairs where the rule body fires **and** x already
> has at least one edge carrying the head relation, what fraction are
> actual head edges in the graph?

This is a "fairer" confidence than naive confidence when the KG is
incomplete (PCA assumes the KG is only missing facts, not stating
wrong facts).

We implement only **length-2 closed rules** of the form

    H(x, y) ← B1(x, z) ∧ B2(z, y)

(sufficient to closely track TACET's current rule synthesizer, which
is also length ≤ 2).  Extending to full AMIE+ requires PRA-like
reasoning and body length ≥ 3 — out of scope.

API::

    from tacet.distill.amie import mine_amie_plus_rules
    rules = mine_amie_plus_rules(graph, min_support=20, min_pca=0.8)
    # rules: list[AMIERule]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tacet.core.graph import WorldGraph


@dataclass(frozen=True)
class AMIERule:
    """A length-2 closed rule with the typical AMIE+ metrics."""

    body1: str
    body2: str
    head: str
    support: int  # number of (x, y) pairs with body1(x,z) ∧ body2(z,y)
    head_coverage: float  # support / |H(_, _)| in the graph
    std_confidence: float  # support / |{(x,y): body1∧body2 fire}|
    pca_confidence: float  # support / |{(x,y): body1∧body2 fire ∧ ∃y' H(x,y')}|


def _edge_index(
    graph: WorldGraph,
) -> tuple[
    dict[tuple[str, str], list[str]],  # (relation, head) -> tails
    dict[str, list[tuple[str, str]]],  # relation -> [(head, tail), ...]
    dict[tuple[str, str], int],  # (relation, head) -> count
    dict[str, int],  # relation -> count
]:
    head_tails: dict[tuple[str, str], list[str]] = {}
    relation_pairs: dict[str, list[tuple[str, str]]] = {}
    head_counts: dict[tuple[str, str], int] = {}
    relation_counts: dict[str, int] = {}
    for edge in graph.edges:
        key = (edge.relation, edge.source)
        head_tails.setdefault(key, []).append(edge.target)
        relation_pairs.setdefault(edge.relation, []).append((edge.source, edge.target))
        head_counts[key] = head_counts.get(key, 0) + 1
        relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
    return head_tails, relation_pairs, head_counts, relation_counts


def mine_amie_plus_rules(
    graph: WorldGraph,
    *,
    min_support: int = 10,
    min_pca: float = 0.5,
    min_head_coverage: float = 0.01,
    max_rules: int | None = None,
) -> list[AMIERule]:
    """Mine length-2 closed rules with PCA confidence ≥ ``min_pca``.

    ``min_support`` filters out rules with < N instances on the current
    graph.  ``min_head_coverage`` filters out rules that rarely predict
    the head relation.  ``max_rules`` caps the output (ranked highest by
    PCA × support).
    """
    head_tails, relation_pairs, head_counts, relation_counts = _edge_index(graph)
    relations = list(relation_counts.keys())
    rules: list[AMIERule] = []

    # Materialise body1(x,z) ∧ body2(z,y) → derived (x, y) pairs and
    # check membership in head relation r3.
    for r1 in relations:
        pairs1 = relation_pairs[r1]
        # Index r1 by z to chain with r2 quickly: z -> {x with (x, r1, z)}.
        z_to_xs: dict[str, set[str]] = {}
        for x, z in pairs1:
            z_to_xs.setdefault(z, set()).add(x)
        for r2 in relations:
            # Skip identity body1 == body2 chains that degenerate to
            # paths-through-self when x == y; the TACET rule miner
            # filters these with inequality guards, so we do the same.
            derived: dict[tuple[str, str], int] = {}
            for z, y in relation_pairs[r2]:
                xs = z_to_xs.get(z)
                if not xs:
                    continue
                for x in xs:
                    if x == y:
                        continue
                    derived[(x, y)] = derived.get((x, y), 0) + 1
            if not derived:
                continue
            for r3 in relations:
                support = sum(
                    1 for (x, y) in derived if (r3, x) in head_tails and y in head_tails[(r3, x)]
                )
                if support < min_support:
                    continue
                std_conf = support / len(derived)
                # PCA confidence: only count derived (x, y) where x has
                # at least one edge with relation r3 in the graph (the
                # PCA assumption: missing edges are missing-from-KG, not
                # negative; absence of *any* r3 edge for x doesn't tell
                # us the rule is wrong on x).
                derived_with_head_x = sum(1 for (x, y) in derived if (r3, x) in head_tails)
                pca_conf = (support / derived_with_head_x) if derived_with_head_x else 0.0
                hc = support / relation_counts[r3]
                if pca_conf < min_pca or hc < min_head_coverage:
                    continue
                rules.append(
                    AMIERule(
                        body1=r1,
                        body2=r2,
                        head=r3,
                        support=support,
                        head_coverage=hc,
                        std_confidence=std_conf,
                        pca_confidence=pca_conf,
                    )
                )

    rules.sort(key=lambda r: (-r.pca_confidence, -r.support))
    if max_rules is not None:
        rules = rules[:max_rules]
    return rules


def compare_with_induced(graph: WorldGraph, induced_rules: list, **kwargs) -> dict:
    """Compare AMIE+ output with rules produced by ``concepts.induce_relations``.

    Returns a dict with the overlap, TACET's recall relative to the AMIE+
    baseline, and the mean accuracy (PCA confidence) over the
    overlapping rules. This module is an optional comparison baseline and is
    not part of the shipped cascade (which mines rules via
    ``tacet.distill.distill.mine_rules``).
    """
    amie = mine_amie_plus_rules(graph, **kwargs)
    amie_signatures = {(r.body1, r.body2, r.head) for r in amie}
    induced_signatures = set()
    for ir in induced_rules:
        # InducedRelation may use either field name: tuple body / single body
        body = getattr(ir, "body_relations", None) or getattr(ir, "body", None)
        head = getattr(ir, "head_relation", None) or getattr(ir, "head", None)
        if body and head and len(body) == 2:
            induced_signatures.add((body[0], body[1], head))
    overlap = amie_signatures & induced_signatures
    return {
        "n_amie_rules": len(amie),
        "n_induced_rules": len(induced_signatures),
        "n_overlap": len(overlap),
        "amie_recall_of_induced": (
            len(overlap) / len(induced_signatures) if induced_signatures else 0.0
        ),
        "induced_recall_of_amie": (len(overlap) / len(amie_signatures) if amie_signatures else 0.0),
        "mean_pca_overlap": (
            sum(r.pca_confidence for r in amie if (r.body1, r.body2, r.head) in overlap)
            / len(overlap)
        )
        if overlap
        else 0.0,
    }


__all__ = ["AMIERule", "mine_amie_plus_rules", "compare_with_induced"]
