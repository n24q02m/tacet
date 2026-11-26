"""Tier-4 entailment, proof-coverage, grounding, and dual precision metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tacet.core.graph import WorldGraph
from tacet.core.symbolic import Rule
from tacet.eval.rule_precision import _body_pairs


def _get_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _is_entailed(engine: Any, head: str, relation: str, tail: str) -> bool:
    if hasattr(engine, "entails"):
        res = engine.entails(head, relation, tail)
        return bool(res.entailed) if hasattr(res, "entailed") else bool(res)
    return (head, relation, tail) in engine.closure


def _extract_answers(record: Any) -> list[str]:
    raw = _get_field(record, "answers", None)
    if raw is None:
        single = _get_field(record, "answer", None)
        if single is None or single == "":
            return []
        return [str(single)]
    if isinstance(raw, str):
        return [raw] if raw else []
    return [str(a) for a in raw if a]


def entailment_rate(records: Iterable[Any], engine: Any) -> float:
    """Fraction of non-empty answer assertions ``(h, r, a)`` entailed by ``engine``."""
    total = 0
    entailed = 0
    for rec in records:
        head = _get_field(rec, "head", None)
        relation = _get_field(rec, "relation", None)
        if head is None or relation is None:
            continue
        for ans in _extract_answers(rec):
            total += 1
            if _is_entailed(engine, str(head), str(relation), ans):
                entailed += 1
    return entailed / total if total else 0.0


def proof_coverage_tier4(records: Iterable[Any]) -> float:
    """Fraction of Tier-4 records (``tier == 4``) carrying a non-empty proof."""
    tier4_total = 0
    with_proof = 0
    for rec in records:
        if _get_field(rec, "tier", None) != 4:
            continue
        tier4_total += 1
        proof = _get_field(rec, "proof", ())
        if isinstance(proof, str):
            has_proof = bool(proof.strip())
        elif isinstance(proof, Iterable):
            has_proof = any(str(line).strip() for line in proof)
        else:
            has_proof = bool(proof)
        if has_proof:
            with_proof += 1
    return with_proof / tier4_total if tier4_total else 0.0


def grounded_fraction(records: Iterable[Any]) -> float:
    """Fraction of records with ``grounded == True``."""
    total = 0
    grounded = 0
    for rec in records:
        total += 1
        if bool(_get_field(rec, "grounded", False)):
            grounded += 1
    return grounded / total if total else 0.0


def _relation_map(
    graph: WorldGraph,
    truth: Mapping[tuple[str, str], Sequence[str]] | WorldGraph,
    relation: str,
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    if isinstance(truth, WorldGraph):
        for h, r, t in truth.triples():
            if r == relation:
                mapping.setdefault(h, set()).add(t)
    else:
        for (h, r), tails in truth.items():
            if r == relation:
                mapping.setdefault(h, set()).update(tails)
    for h, r, t in graph.triples():
        if r == relation and h not in mapping:
            mapping.setdefault(h, set()).add(t)
    return mapping


def colleague_precision_dual(
    rule: Rule,
    graph: WorldGraph,
    truth: Mapping[tuple[str, str], Sequence[str]] | WorldGraph,
) -> dict[str, float]:
    """Evaluate ``rule`` on ``graph`` under both strict and relaxed colleague contracts.

    * ``strict_precision`` — fraction of irreflexive (``x != y``) predicted pairs
      present in ``truth`` for ``"colleague_of"``.
    * ``relaxed_precision`` — fraction of all predicted pairs satisfied under
      reflexive (``x == y``), symmetric ``colleague_of`` closure, or shared
      ``works_at`` / ``member_of`` affiliation.
    """
    facts = set(graph.triples())
    predicted_all = _body_pairs(rule, facts)
    predicted_strict = {(x, y) for x, y in predicted_all if x != y}

    if isinstance(truth, WorldGraph):
        strict_true = {(h, t) for h, r, t in truth.triples() if r == "colleague_of" and h != t}
    else:
        strict_true = {
            (h, t)
            for (h, r), tails in truth.items()
            if r == "colleague_of"
            for t in tails
            if h != t
        }

    sym_colleague = strict_true | {(y, x) for x, y in strict_true}
    works_at_map = _relation_map(graph, truth, "works_at")
    dept_map = _relation_map(graph, truth, "member_of")

    strict_hits = sum(1 for pair in predicted_strict if pair in strict_true)
    relaxed_hits = sum(
        1
        for x, y in predicted_all
        if x == y
        or (x, y) in sym_colleague
        or bool(works_at_map.get(x, set()) & works_at_map.get(y, set()))
        or bool(dept_map.get(x, set()) & dept_map.get(y, set()))
    )

    strict_precision = strict_hits / len(predicted_strict) if predicted_strict else 0.0
    relaxed_precision = relaxed_hits / len(predicted_all) if predicted_all else 0.0
    return {
        "strict_precision": strict_precision,
        "relaxed_precision": relaxed_precision,
        "predicted_pairs_strict": float(len(predicted_strict)),
        "predicted_pairs_all": float(len(predicted_all)),
        "true_pairs": float(len(strict_true)),
    }
