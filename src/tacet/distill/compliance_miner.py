"""Conjunctive rule mining for the compliance domain.

Mines AMIE-style rules from TEACHER-labelled cases (never gold labels):
conjunctions of normalised slot atoms imply either a violated article or a
``permit`` verdict. Mined rules are standard engine ``Rule`` objects --
body atoms share one case variable, with category constants -- so the sound
Tier-1 engine evaluates them with replayable proof trees and the ontology
gate vets every derived edge.

Search is apriori over the closed vocabulary's (slot, category) atoms with
support pruning; patterns are kept most-general-first (a superset pattern is
dropped when a subset already reaches the confidence bar for the same
target).
"""

from __future__ import annotations

from dataclasses import dataclass

from tacet.core.symbolic import Rule
from tacet.data.privaci_graph import SLOT_RELATIONS

Atom = tuple[str, str]  # (slot, category)

PERMIT_TARGET = "permit"


@dataclass(frozen=True)
class LabeledCase:
    """One teacher-labelled case: normalised atoms + the teacher's answer."""

    case_id: str
    atoms: frozenset[Atom]
    verdict: str  # "permit" | "prohibit"
    articles: tuple[str, ...]  # e.g. ("art6", "art32"); empty for permit


@dataclass(frozen=True)
class MinedComplianceRule:
    rule: Rule
    target: str  # "artN" or "permit"
    confidence: float
    support: int


def _body(pattern: tuple[Atom, ...]) -> tuple[tuple[str, str, str], ...]:
    out = []
    for slot, category in pattern:
        rel, target_type = SLOT_RELATIONS[slot]
        out.append(("?c", rel, f"{target_type}:{category}"))
    return tuple(out)


def _head(target: str) -> tuple[str, str, str]:
    if target == PERMIT_TARGET:
        return ("?c", "verdict", "verdict:permit")
    return ("?c", "violates", f"article:{target}")


def _rule_name(target: str, pattern: tuple[Atom, ...]) -> str:
    body = "__".join(f"{s}-{c}" for s, c in pattern)
    return f"mined_{target}__{body}"


def mine_compliance_rules(
    labeled: list[LabeledCase],
    *,
    min_support: int = 5,
    min_confidence: float = 0.9,
    max_atoms: int = 3,
) -> list[MinedComplianceRule]:
    """Mine conjunctive (pattern -> target) rules from teacher-labelled cases."""
    if not labeled:
        return []

    # ----- candidate patterns: apriori with support pruning ---------------
    atom_support: dict[Atom, int] = {}
    atom_index: dict[Atom, set[int]] = {}  # inverted index for fast matching
    for i, case in enumerate(labeled):
        for atom in case.atoms:
            atom_support[atom] = atom_support.get(atom, 0) + 1
            if atom not in atom_index:
                atom_index[atom] = set()
            atom_index[atom].add(i)
    frequent_atoms = sorted(a for a, n in atom_support.items() if n >= min_support)

    pattern_matches: dict[tuple[Atom, ...], set[int]] = {}
    for a in frequent_atoms:
        pattern_matches[(a,)] = atom_index[a]

    levels: list[list[tuple[Atom, ...]]] = [[(a,) for a in frequent_atoms]]
    for _ in range(2, max_atoms + 1):
        prev = levels[-1]
        nxt = set()
        for p in prev:
            p_matches = pattern_matches[p]
            p_last = p[-1]
            for a in frequent_atoms:
                if a > p_last:  # canonical order -> no duplicate combos
                    new_matches = p_matches & atom_index[a]
                    if len(new_matches) >= min_support:
                        new_p = (*p, a)
                        pattern_matches[new_p] = new_matches
                        nxt.add(new_p)
        levels.append(sorted(nxt))

    # ----- score every (pattern, target) pair -----------------------------
    candidates: list[tuple[tuple[Atom, ...], str, float, int]] = []

    case_targets: list[list[str]] = [[] for _ in range(len(labeled))]
    for i, c in enumerate(labeled):
        if c.verdict == "permit":
            case_targets[i].append(PERMIT_TARGET)
        for art in c.articles:
            case_targets[i].append(art)

    for level in levels:
        for pattern in level:
            matched_indices = pattern_matches.get(pattern)
            if not matched_indices or len(matched_indices) < min_support:
                continue
            target_counts: dict[str, int] = {}
            for i in matched_indices:
                for target in case_targets[i]:
                    target_counts[target] = target_counts.get(target, 0) + 1

            matched_len = len(matched_indices)
            for target, hits in target_counts.items():
                if hits >= min_support:
                    conf = hits / matched_len
                    if conf >= min_confidence:
                        candidates.append((pattern, target, conf, hits))

    # ----- most-general-first pruning --------------------------------------
    candidates.sort(key=lambda c: (len(c[0]), -c[2], -c[3], c[1], c[0]))
    kept: list[tuple[tuple[Atom, ...], str, float, int]] = []
    kept_by_target: dict[str, list[tuple[set[Atom], float]]] = {}

    for pattern, target, conf, hits in candidates:
        pat = set(pattern)
        dominated = False

        if target in kept_by_target:
            for k_pattern, k_conf in kept_by_target[target]:
                if k_conf >= conf and k_pattern < pat:
                    dominated = True
                    break

        if not dominated:
            kept.append((pattern, target, conf, hits))
            if target not in kept_by_target:
                kept_by_target[target] = []
            kept_by_target[target].append((pat, conf))

    return [
        MinedComplianceRule(
            rule=Rule(name=_rule_name(target, pattern), body=_body(pattern), head=_head(target)),
            target=target,
            confidence=conf,
            support=hits,
        )
        for pattern, target, conf, hits in kept
    ]


__all__ = ["PERMIT_TARGET", "Atom", "LabeledCase", "MinedComplianceRule", "mine_compliance_rules"]
