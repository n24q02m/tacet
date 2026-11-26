"""3-arm Tier-4 SFT dataset exporter with fail-closed entity-leakage checks.

Exports training records from a symbolic ``RuleEngine`` and knowledge graph
across three experimental arms:

* ``"A"`` — answer-only targets ``(head, relation) -> answers`` (``proof=()``).
* ``"B"`` — proof-guided targets with grounded Datalog proof lines.
* ``"C"`` — Arm ``"B"`` plus deterministic hard-negative / abstain queries.

All exports enforce strict zero overlap between ``train_entities`` and
``eval_entities`` across entity sets, heads, answers, proof lines, and
underlying derivation premises, failing closed with :class:`EntityLeakageError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tacet.core.graph import WorldGraph

Triple = tuple[str, str, str]
Arm = Literal["A", "B", "C"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]+")


class EntityLeakageError(ValueError):
    """Raised when eval entities overlap or leak into exported training data."""


@dataclass(frozen=True)
class ExportRecord:
    """One deterministic SFT record for Tier-4 distillation."""

    head: str
    relation: str
    answers: tuple[str, ...]
    proof: tuple[str, ...]
    arm: str
    abstain: bool = False
    grounded: bool = False


@dataclass(frozen=True)
class ExportManifest:
    """Integrity manifest describing an exported Tier-4 JSONL dataset."""

    arm: str
    seed: int
    record_count: int
    positive_count: int
    negative_count: int
    sha256: str
    train_entities_count: int
    eval_entities_count: int
    leakage_checked: bool
    jsonl_path: str
    manifest_path: str


def _entails(engine: Any, head: str, relation: str, tail: str) -> bool:
    if hasattr(engine, "entails"):
        res = engine.entails(head, relation, tail)
        return bool(res.entailed) if hasattr(res, "entailed") else bool(res)
    return (head, relation, tail) in engine.closure


def _derivations_map(engine: Any, *, grounded_only: bool) -> dict[Triple, Any]:
    if hasattr(engine, "derivations"):
        raw = engine.derivations(grounded_only=grounded_only)
        if isinstance(raw, Mapping):
            return dict(raw)
        return {d.triple: d for d in raw}
    return dict(getattr(engine, "_derivations", {}))


def _is_grounded(engine: Any, triple: Triple, grounded_derivs: Mapping[Triple, Any]) -> bool:
    if not _entails(engine, *triple):
        return False
    if hasattr(engine, "proof_is_grounded"):
        return bool(engine.proof_is_grounded(triple))
    # Duck-typed Phase-2 engine without proof_is_grounded: derived triples must
    # appear in grounded_derivs; base facts (absent from all derivations) are grounded.
    all_derivs = _derivations_map(engine, grounded_only=False)
    if triple in all_derivs:
        return triple in grounded_derivs
    return True


def _explain_triple(engine: Any, triple: Triple) -> list[str]:
    if hasattr(engine, "explain"):
        return list(engine.explain(triple))
    if hasattr(engine, "_explain"):
        return list(engine._explain(triple))
    if hasattr(engine, "query"):
        return list(engine.query(triple[0], triple[1]).proof)
    return []


def _parse_proof_endpoints(line: str) -> tuple[str, str] | None:
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        return None
    payload = parts[1]
    if payload.rstrip().endswith("]") and "[" in payload:
        payload = payload[: payload.rindex("[")]
    if " -" not in payload or "-> " not in payload:
        return None
    head, rest = payload.split(" -", 1)
    _, tail = rest.split("-> ", 1)
    return head.strip(), tail.strip()


def _check_premise_leakage(
    triple: Triple,
    eval_set: set[str],
    deriv_map: Mapping[Triple, Any],
) -> None:
    seen: set[Triple] = set()
    stack = [triple]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur[0] in eval_set or cur[2] in eval_set:
            raise EntityLeakageError(f"eval entity referenced in derivation triple {cur!r}")
        deriv = deriv_map.get(cur)
        if deriv is not None:
            for premise in getattr(deriv, "premises", ()):
                stack.append(premise)


def _check_record_leakage(record: ExportRecord, eval_set: set[str]) -> None:
    if record.head in eval_set:
        raise EntityLeakageError(f"eval entity {record.head!r} leaked as record head")
    for ans in record.answers:
        if ans in eval_set:
            raise EntityLeakageError(
                f"eval entity {ans!r} leaked in answers for ({record.head}, {record.relation})"
            )
    for line in record.proof:
        endpoints = _parse_proof_endpoints(line)
        if endpoints is not None and (endpoints[0] in eval_set or endpoints[1] in eval_set):
            raise EntityLeakageError(f"eval entity leaked in proof line: {line!r}")
        for token in _TOKEN_RE.findall(line):
            if token in eval_set:
                raise EntityLeakageError(f"eval entity {token!r} leaked in proof line: {line!r}")


def _has_any_entailment(
    engine: Any, head: str, relation: str, candidate_tails: Sequence[str]
) -> bool:
    if hasattr(engine, "query"):
        return bool(engine.query(head, relation).answered)
    return any(_entails(engine, head, relation, tail) for tail in candidate_tails)


def _relation_admits_head(engine: Any, graph: WorldGraph, head: str, relation: str) -> bool:
    onto = getattr(engine, "ontology", None)
    if onto is None or not getattr(onto, "relation_types", None):
        return True
    rt = onto.relation(relation)
    if rt is None or "*" in rt.domain:
        return True
    node = graph.node(head)
    if node is None:
        return True
    return node.type in rt.domain


def _collect_positive_queries(
    engine: Any,
    truth: Mapping[tuple[str, str], Sequence[str]],
    train_set: set[str],
    eval_set: set[str],
) -> list[tuple[str, str, list[str]]]:
    queries: list[tuple[str, str, list[str]]] = []
    if truth:
        for (head, relation), raw_answers in sorted(truth.items()):
            if head not in train_set:
                continue
            unique_raw = sorted({a for a in raw_answers if a})
            if not unique_raw:
                continue
            for ans in unique_raw:
                if ans in eval_set:
                    raise EntityLeakageError(
                        f"eval entity {ans!r} leaked in truth for ({head}, {relation})"
                    )
            queries.append((head, relation, unique_raw))
        return queries

    by_head: dict[tuple[str, str], set[str]] = {}
    for h, r, t in sorted(engine.closure):
        if h in train_set:
            by_head.setdefault((h, r), set()).add(t)
    for (head, relation), tails in sorted(by_head.items()):
        queries.append((head, relation, sorted(tails)))
    return queries


def export_tier4_dataset(
    graph: WorldGraph,
    engine: Any,
    truth: Mapping[tuple[str, str], Sequence[str]],
    train_entities: Iterable[str],
    eval_entities: Iterable[str],
    out_dir: str | Path,
    *,
    arm: Arm,
    seed: int = 0,
    negative_ratio: float = 0.25,
) -> ExportManifest:
    """Export deterministic Tier-4 SFT records and manifest for ``arm`` (A/B/C)."""
    if arm not in {"A", "B", "C"}:
        raise ValueError(f"arm must be one of 'A', 'B', 'C', got {arm!r}")
    if not math.isfinite(negative_ratio) or negative_ratio < 0.0:
        raise ValueError(f"negative_ratio must be finite and >= 0.0, got {negative_ratio!r}")

    train_set = set(train_entities)
    eval_set = set(eval_entities)
    overlap = train_set & eval_set
    if overlap:
        raise EntityLeakageError(
            f"train_entities and eval_entities overlap: {sorted(overlap)[:10]}"
        )

    grounded_derivs = _derivations_map(engine, grounded_only=True)
    all_derivs = _derivations_map(engine, grounded_only=False)
    queries = _collect_positive_queries(engine, truth, train_set, eval_set)

    positive_records: list[ExportRecord] = []
    positive_keys: set[tuple[str, str]] = set()
    for head, relation, candidate_answers in queries:
        grounded_answers = sorted(
            a
            for a in candidate_answers
            if _is_grounded(engine, (head, relation, a), grounded_derivs)
        )
        if not grounded_answers:
            continue

        proof_lines: list[str] = []
        for tail in grounded_answers:
            triple = (head, relation, tail)
            _check_premise_leakage(triple, eval_set, all_derivs)
            if arm in {"B", "C"}:
                proof_lines.extend(_explain_triple(engine, triple))

        rec = ExportRecord(
            head=head,
            relation=relation,
            answers=tuple(grounded_answers),
            proof=tuple(proof_lines) if arm in {"B", "C"} else (),
            arm=arm,
            abstain=False,
            grounded=True,
        )
        _check_record_leakage(rec, eval_set)
        positive_records.append(rec)
        positive_keys.add((head, relation))

    negative_records: list[ExportRecord] = []
    if arm == "C" and negative_ratio > 0.0 and positive_records:
        target_neg = max(1, round(len(positive_records) * negative_ratio))
        known_relations = sorted(
            graph.relations()
            | {r for _, r, _ in engine.closure}
            | {r for _, r in truth}
            | {r.head[1] for r in getattr(engine, "rules", ())}
        )
        all_tails = sorted(train_set | set(graph.entities()))
        neg_candidates: list[tuple[str, str]] = []
        for head in sorted(train_set):
            for relation in known_relations:
                if (head, relation) in positive_keys:
                    continue
                if not _relation_admits_head(engine, graph, head, relation):
                    continue
                if _has_any_entailment(engine, head, relation, all_tails):
                    continue
                neg_candidates.append((head, relation))

        rng = random.Random(seed)
        k = min(target_neg, len(neg_candidates))
        chosen = sorted(rng.sample(neg_candidates, k)) if k > 0 else []
        for head, relation in chosen:
            rec = ExportRecord(
                head=head,
                relation=relation,
                answers=(),
                proof=(),
                arm=arm,
                abstain=True,
                grounded=False,
            )
            _check_record_leakage(rec, eval_set)
            negative_records.append(rec)

    records = positive_records + negative_records
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / f"tier4_arm_{arm}_s{seed}.jsonl"
    manifest_path = out_path / f"manifest_arm_{arm}_s{seed}.json"

    jsonl_payload = "".join(
        json.dumps(asdict(rec), sort_keys=True, ensure_ascii=False) + "\n" for rec in records
    ).encode("utf-8")
    jsonl_path.write_bytes(jsonl_payload)
    digest = hashlib.sha256(jsonl_payload).hexdigest()

    manifest = ExportManifest(
        arm=arm,
        seed=seed,
        record_count=len(records),
        positive_count=len(positive_records),
        negative_count=len(negative_records),
        sha256=digest,
        train_entities_count=len(train_set),
        eval_entities_count=len(eval_set),
        leakage_checked=True,
        jsonl_path=str(jsonl_path),
        manifest_path=str(manifest_path),
    )
    manifest_bytes = (
        json.dumps(asdict(manifest), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest
