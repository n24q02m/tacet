"""Goal-directed agency — TACET as the memory of a planner.

Adds three first-class concepts that turn the substrate from a QA system
into something that can *do things*:

* ``Action``      — a STRIPS-style operator with variable preconditions,
                    add-effects, remove-effects and a unit cost.
* ``Goal``        — a conjunction of triple patterns that must hold.
* ``Planner``     — best-first / cost-bounded search over graph states; uses
                    TACET's symbolic engine to ground actions and check
                    goals (so axioms, given rules and synthesised rules all
                    participate in plan validation).

The planner is intentionally classical (Datalog state + STRIPS operators).
For continuous-state agents and POMDPs the ``worldmodel`` interface is the
hook; this module covers the symbolic-planning leg of the agent loop.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.core.symbolic import Pattern, Rule, RuleEngine, _ground, _is_var, _unify

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class Action:
    """A STRIPS-style operator with variable patterns.

    Variables (``?x`` syntax) shared across pre / add / remove are co-bound
    by the same substitution. ``cost`` is the planner's edge weight.
    """

    name: str
    preconditions: tuple[Pattern, ...] = ()
    add: tuple[Pattern, ...] = ()
    remove: tuple[Pattern, ...] = ()
    cost: float = 1.0


@dataclass(frozen=True)
class GroundedAction:
    """An `Action` whose variables have all been bound to concrete entities."""

    action: Action
    bindings: tuple[tuple[str, str], ...]  # frozenset-equivalent

    def signature(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.bindings)
        return f"{self.action.name}({args})"


@dataclass
class Goal:
    """A conjunction of triple patterns to be entailed by the final state."""

    triples: tuple[Pattern, ...]


@dataclass
class Plan:
    actions: list[GroundedAction]
    cost: float

    def __len__(self) -> int:
        return len(self.actions)

    def __str__(self) -> str:
        return " → ".join(a.signature() for a in self.actions) or "(empty plan)"


# ----------------------------------------------------------------------------
def _ground_action(
    action: Action, facts: set[Triple], entities: list[str] | None = None
) -> Iterable[GroundedAction]:
    """Enumerate all ways to ground `action` against the current fact set.

    Variables that appear in ``add`` / ``remove`` but not in any
    precondition (a STRIPS "free output") are enumerated over `entities`
    (defaults to all entities seen in the current facts).
    """
    if entities is None:
        entities = sorted({x for f in facts for x in (f[0], f[2])})
    bindings: list[dict[str, str]] = [{}]
    for pat in action.preconditions:
        nxt: list[dict[str, str]] = []
        for b in bindings:
            for fact in facts:
                merged = _unify(pat, fact, b)
                if merged is not None:
                    nxt.append(merged)
        bindings = nxt
        if not bindings:
            return
    # collect every variable mentioned anywhere in the action
    used = {
        tok
        for pat in action.preconditions + action.add + action.remove
        for tok in pat
        if _is_var(tok)
    }
    for b in bindings:
        free = [v for v in used if v not in b]
        # cartesian product over free variables
        stack: list[tuple[dict[str, str], list[str]]] = [(b, free)]
        while stack:
            curr, remaining = stack.pop()
            if not remaining:
                yield GroundedAction(action=action, bindings=tuple(sorted(curr.items())))
                continue
            v = remaining[0]
            rest = remaining[1:]
            for e in entities:
                stack.append(({**curr, v: e}, rest))


def _apply(facts: frozenset[Triple], ga: GroundedAction) -> frozenset[Triple]:
    b = dict(ga.bindings)
    add = {_ground(p, b) for p in ga.action.add}
    rm = {_ground(p, b) for p in ga.action.remove}
    return frozenset((facts - rm) | add)


def _goal_satisfied(facts: set[Triple], goal: Goal) -> bool:
    for pat in goal.triples:
        bindings: list[dict[str, str]] = [{}]
        for atom in (pat,):
            nxt: list[dict[str, str]] = []
            for b in bindings:
                for f in facts:
                    merged = _unify(atom, f, b)
                    if merged is not None:
                        nxt.append(merged)
            bindings = nxt
        if not bindings:
            return False
    return True


# ----------------------------------------------------------------------------
class Planner:
    """Best-first STRIPS planner over the deductive closure of the graph.

    At every search node the closure is recomputed under the rule engine, so
    rules / ontology axioms participate in goal checking — a goal can ask
    for a *derived* triple (e.g.\\ ``ancestor_of(a, z)``) and the planner
    finds a sequence of actions that makes it true.
    """

    def __init__(
        self, actions: list[Action], ontology: Ontology, rules: list[Rule] | None = None
    ) -> None:
        self.actions = actions
        self._engine = RuleEngine(ontology, rules)

    def plan(
        self, start: WorldGraph, goal: Goal, max_depth: int = 6, max_nodes: int = 5000
    ) -> Plan | None:
        # free action variables are enumerated over every entity in the
        # graph (not only those mentioned in facts), so plans can reach
        # entities never yet touched by an asserted relation.
        entities = sorted(start.entities())
        facts = self._closure_facts(start)
        if _goal_satisfied(facts, goal):
            return Plan(actions=[], cost=0.0)
        # priority queue ordered by accumulated cost; nodes are
        # (cost, depth, counter, facts, plan_prefix)
        seen: set[frozenset[Triple]] = {facts}
        queue: list[tuple[float, int, int, frozenset[Triple], list[GroundedAction]]] = []
        heapq.heappush(queue, (0.0, 0, 0, facts, []))
        counter = 0
        while queue and counter < max_nodes:
            counter += 1
            cost, depth, _, state, prefix = heapq.heappop(queue)
            if depth >= max_depth:
                continue
            for action in self.actions:
                for ga in _ground_action(action, set(state), entities):
                    next_state = _apply(state, ga)
                    if next_state in seen:
                        continue
                    seen.add(next_state)
                    new_prefix = [*prefix, ga]
                    next_facts = self._closure_facts_from_set(next_state)
                    if _goal_satisfied(next_facts, goal):
                        return Plan(actions=new_prefix, cost=cost + action.cost)
                    heapq.heappush(
                        queue, (cost + action.cost, depth + 1, counter, next_state, new_prefix)
                    )
        return None

    def _closure_facts(self, graph: WorldGraph) -> frozenset[Triple]:
        return frozenset(self._engine.materialise(graph))

    def _closure_facts_from_set(self, facts: frozenset[Triple]) -> frozenset[Triple]:
        g = WorldGraph()
        for h, r, t in facts:
            g.add_edge(h, r, t)
        return frozenset(self._engine.materialise(g))


__all__ = [
    "Action",
    "Goal",
    "GroundedAction",
    "Plan",
    "Planner",
]
