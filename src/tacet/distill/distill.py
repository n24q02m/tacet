"""Distillation — moving knowledge from the LLM teacher down to cheap tiers.

Three mechanisms compile expensive Tier-3 reasoning into reusable assets:

1. **Fact write-back** — a teacher answer becomes a graph edge, so the
   *identical* query is answered by Tier 1 forever after.
2. **KGE augmentation** — teacher facts join the Tier-2 training set, so the
   embedding model generalises from them on the next consolidation.
3. **Rule synthesis** — the interesting one. An AMIE-style miner (Galárraga
   et al., 2013) induces high-confidence Horn rules from the graph plus the
   accumulated teacher facts, so a *whole relational pattern* — not just one
   fact — is absorbed into the sound symbolic tier.

Rule synthesis is what makes the cascade beat a cache: a synthesised rule
answers *unseen* heads, whereas a cache only answers exact repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph
from tacet.core.symbolic import Rule

Triple = tuple[str, str, str]
Pair = tuple[str, str]


@dataclass
class MinedRule:
    rule: Rule
    confidence: float
    support: int


def _pairs_index(facts: set[Triple]) -> dict[str, set[Pair]]:
    idx: dict[str, set[Pair]] = {}
    for h, r, t in facts:
        idx.setdefault(r, set()).add((h, t))
    return idx


def _directed(pairs: set[Pair], inverse: bool) -> set[Pair]:
    return {(t, h) for h, t in pairs} if inverse else set(pairs)


def _adj(pairs: set[Pair]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for h, t in pairs:
        adj.setdefault(h, set()).add(t)
    return adj


def mine_rules(
    graph: WorldGraph,
    teacher_facts: set[Triple],
    target: str,
    min_confidence: float = 0.95,
    min_support: int = 3,
    max_rules: int = 8,
    complete_heads: set[str] | None = None,
    allowed_body: set[str] | None = None,
) -> list[MinedRule]:
    """Induce Horn rules `body => target(x,y)` from data (body length 1-2).

    Confidence is the AMIE-style confidence: the fraction of distinct (x,y)
    pairs satisfying the body that also satisfy the head. It is estimated only
    over heads with a *complete* ground-truth answer (`complete_heads` — the
    heads the teacher has answered), since the teacher returns the full answer
    set for each query, so a correct rule is not penalised for unobserved
    heads. `allowed_body` restricts body atoms to a relation whitelist (the
    base relations), which keeps the search well-posed and avoids chaining
    one synthesised relation into another.
    """
    rules, _ = mine_rules_with_stats(
        graph,
        teacher_facts,
        target,
        min_confidence,
        min_support,
        max_rules,
        complete_heads,
        allowed_body,
    )
    return rules


def mine_rules_with_stats(
    graph: WorldGraph,
    teacher_facts: set[Triple],
    target: str,
    min_confidence: float = 0.95,
    min_support: int = 3,
    max_rules: int = 8,
    complete_heads: set[str] | None = None,
    allowed_body: set[str] | None = None,
) -> tuple[list[MinedRule], int]:
    """`mine_rules` plus the count of candidate rules *proposed* before filtering.

    Returns ``(installed_rules, n_proposed)`` where ``installed_rules`` is
    exactly what :func:`mine_rules` returns (passed the confidence/support
    threshold, de-duplicated, capped at ``max_rules``) and ``n_proposed`` is the
    number of distinct candidate bodies the miner forms over the complete heads
    — every length-1 and length-2 body with at least ``min_support`` groundings —
    *before* the confidence/support/cap cut. ``n_proposed >= len(installed)`` by
    construction; the gap is the over-production the confidence filter removes.

    This surfaces the count only; the installed-rule behaviour is unchanged.
    """
    facts: set[Triple] = set(graph.triples()) | teacher_facts
    idx = _pairs_index(facts)
    target_pairs = idx.get(target, set())
    if len(target_pairs) < min_support:
        return [], 0

    def keep(pairs: set[Pair]) -> set[Pair]:
        if complete_heads is None:
            return pairs
        return {(x, y) for x, y in pairs if x in complete_heads}

    if allowed_body is not None:
        relations = sorted(set(idx) & (allowed_body | {target}))
    else:
        relations = sorted(idx)
    candidates: list[MinedRule] = []
    proposed_names: set[str] = set()

    def atom(rel: str, inv: bool, a: str, b: str) -> tuple[str, str, str]:
        return (b, rel, a) if inv else (a, rel, b)

    # ---- length-1 body: R1(x,y) => target(x,y) --------------------------
    for r1 in relations:
        if r1 == target:
            continue
        for inv1 in (False, True):
            # exclude reflexive (x, x) pairs from the confidence denominator, as
            # the length-2 branch does: the installed rule carries a distinct
            # guard and can never fire on (x, x), so confidence must be measured
            # over the non-reflexive pairs the rule can actually derive.
            body = keep({(x, y) for x, y in _directed(idx[r1], inv1) if x != y})
            if len(body) < min_support:
                continue
            # A well-formed candidate (a body of sufficient size): proposed.
            name = f"syn:{target}<={'~' if inv1 else ''}{r1}"
            proposed_names.add(name)
            support = len(body & target_pairs)
            conf = support / len(body)
            if conf >= min_confidence and support >= min_support:
                rule = Rule(
                    name=name,
                    body=(atom(r1, inv1, "?x", "?y"),),
                    head=("?x", target, "?y"),
                    distinct=(("?x", "?y"),),
                )
                candidates.append(MinedRule(rule, conf, support))

    # Pre-compute adjacency maps to avoid O(|R|^2) rebuilds
    adj_maps: dict[tuple[str, bool], dict[str, set[str]]] = {}
    for r in relations:
        for inv in (False, True):
            adj_maps[(r, inv)] = _adj(_directed(idx[r], inv))

    # ---- length-2 body: R1(x,z) & R2(z,y) => target(x,y) ----------------
    for r1 in relations:
        for inv1 in (False, True):
            p1 = adj_maps[(r1, inv1)]
            if not p1:
                continue
            for r2 in relations:
                for inv2 in (False, True):
                    p2 = adj_maps[(r2, inv2)]
                    if not p2:
                        continue
                    raw: set[Pair] = set()
                    for x, zs in p1.items():
                        if complete_heads is not None and x not in complete_heads:
                            continue
                        for z in zs:
                            for y in p2.get(z, ()):
                                if x != y:
                                    raw.add((x, y))
                    body_pairs = raw if complete_heads is not None else keep(raw)
                    if len(body_pairs) < min_support:
                        continue
                    name = f"syn:{target}<={'~' if inv1 else ''}{r1}.{'~' if inv2 else ''}{r2}"
                    # A well-formed length-2 candidate body: proposed.
                    proposed_names.add(name)
                    support = len(body_pairs & target_pairs)
                    conf = support / len(body_pairs)
                    if conf >= min_confidence and support >= min_support:
                        rule = Rule(
                            name=name,
                            body=(atom(r1, inv1, "?x", "?z"), atom(r2, inv2, "?z", "?y")),
                            head=("?x", target, "?y"),
                            distinct=(("?x", "?y"),),
                        )
                        candidates.append(MinedRule(rule, conf, support))

    candidates.sort(key=lambda m: (m.confidence, m.support), reverse=True)
    seen: set[str] = set()
    unique: list[MinedRule] = []
    for m in candidates:
        if m.rule.name not in seen:
            seen.add(m.rule.name)
            unique.append(m)
    return unique[:max_rules], len(proposed_names)


@dataclass
class Distiller:
    """Accumulates teacher knowledge and triggers the three distillation paths."""

    synth_trigger: int = 10  # complete heads on a relation before mining
    min_confidence: float = 0.95
    min_support: int = 3
    base_relations: set[str] = field(default_factory=set)
    teacher_facts: set[Triple] = field(default_factory=set)
    _complete_heads: dict[str, set[str]] = field(default_factory=dict)
    _synthesised: set[str] = field(default_factory=set)

    def record(self, head: str, relation: str, answers: list[str]) -> list[Triple]:
        """Register a teacher answer; returns the facts to write back to the graph."""
        facts = [(head, relation, t) for t in answers]
        for f in facts:
            self.teacher_facts.add(f)
        # the teacher returns the complete answer set for `head`, so `head` now
        # has fully-known ground truth for `relation`.
        self._complete_heads.setdefault(relation, set()).add(head)
        return facts

    def ready_to_synthesise(self, relation: str) -> bool:
        return (
            relation not in self._synthesised
            and len(self._complete_heads.get(relation, set())) >= self.synth_trigger
        )

    def synthesise(self, graph: WorldGraph, relation: str) -> list[MinedRule]:
        """Mine rules for `relation` from accumulated knowledge (once per relation)."""
        self._synthesised.add(relation)
        return mine_rules(
            graph,
            self.teacher_facts,
            relation,
            self.min_confidence,
            self.min_support,
            complete_heads=self._complete_heads.get(relation, set()),
            allowed_body=self.base_relations or None,
        )

    def kge_augmentation(self) -> list[Triple]:
        """Teacher facts to add to the Tier-2 training set on next consolidation."""
        return sorted(self.teacher_facts)
