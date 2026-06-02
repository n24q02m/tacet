"""World-precision of a synthesised rule against a full ground-truth graph.

A mined rule passes the cascade's *confidence* filter, which is estimated only
over the heads the teacher has answered. That is a noisy, partial-data estimate;
this module answers a different, harder question:

    Is the rule actually *true in the world*?

``rule_world_precision(rule, ground_truth_graph)`` returns the fraction of
distinct ``(x, y)`` pairs that satisfy the rule's body — joined over the FULL
ground-truth graph, every entity — that also satisfy its head. A genuinely-valid
composition (e.g. ``parent_of . parent_of => grandparent_of``) scores ~1.0; a
spurious co-occurrence the confidence filter happened to admit scores low. It is
the reviewer-requested check on whether the confidence threshold yields rules
that generalise correctly rather than merely fitting the answered heads.

The body join reuses the symbolic engine's relational-join semantics (the same
``RuleEngine._join`` the closure is built from), so the pairs counted here are
exactly the pairs the rule would fire on, including inverse atoms and the
``distinct`` inequality guard.
"""

from __future__ import annotations

from tacet.core.graph import WorldGraph
from tacet.core.symbolic import Rule, RuleEngine

Triple = tuple[str, str, str]


def _body_pairs(rule: Rule, facts: set[Triple]) -> set[tuple[str, str]]:
    """Distinct ``(?x, ?y)`` bindings of the rule's body over `facts`.

    Uses the engine's join (relation-indexed, bound-side narrowed) so the
    semantics match what the Tier-1 engine would derive, then applies the
    rule's ``distinct`` inequality guards (e.g. ``?x != ?y``).
    """
    idx_all: dict[str, list[Triple]] = {}
    idx_subj: dict[tuple[str, str], list[Triple]] = {}
    idx_obj: dict[tuple[str, str], list[Triple]] = {}
    for fact in facts:
        h, r, t = fact
        idx_all.setdefault(r, []).append(fact)
        idx_subj.setdefault((r, h), []).append(fact)
        idx_obj.setdefault((r, t), []).append(fact)

    head_x, _rel, head_y = rule.head
    pairs: set[tuple[str, str]] = set()
    for binding in RuleEngine._join(rule.body, idx_all, idx_subj, idx_obj):
        if any(binding.get(a) == binding.get(b) for a, b in rule.distinct):
            continue
        x = binding.get(head_x, head_x)
        y = binding.get(head_y, head_y)
        pairs.add((x, y))
    return pairs


def rule_world_precision(rule: Rule, ground_truth_graph: WorldGraph) -> float:
    """Precision of ``body => head`` over the full ground-truth graph.

    Returns ``|body_pairs ∩ head_pairs| / |body_pairs|`` where ``head_pairs`` is
    the set of ``(x, y)`` for which the head relation holds in the ground-truth
    graph, computed over ALL entities (not only teacher-answered heads). Returns
    ``0.0`` when the body never fires (no pair to be precise about).
    """
    facts: set[Triple] = set(ground_truth_graph.triples())
    head_rel = rule.head[1]
    head_pairs = {(h, t) for h, r, t in facts if r == head_rel}

    body_pairs = _body_pairs(rule, facts)
    if not body_pairs:
        return 0.0
    correct = len(body_pairs & head_pairs)
    return correct / len(body_pairs)
