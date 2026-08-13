"""Tier 1 — symbolic reasoning (a sound, explainable Datalog-style engine).

`RuleEngine` computes the *deductive closure* of a graph: every triple
derivable from the base facts plus (a) ontology axioms (symmetric, transitive,
inverse relations) and (b) IF-THEN rules — hand-written or synthesised by the
Tier-3 teacher.

The rules are function-free and range-restricted (every variable in the head
also appears in the body), so forward chaining reaches a least fixpoint that is
**exactly** the set of triples entailed under Datalog semantics. Two
consequences the cascade relies on:

* *Soundness* — every triple the engine returns is entailed by (facts ∪ rules);
  Tier 1 never guesses. When it cannot derive an answer it abstains.
* *Explainability* — each derived triple keeps provenance, so every answer
  unfolds into a full proof tree down to base facts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology

Triple = tuple[str, str, str]
Pattern = tuple[str, str, str]  # tokens beginning with '?' are variables


def _is_var(token: str) -> bool:
    return token.startswith("?")


def save_rules_json(rules: list[Rule], path) -> None:  # noqa: ANN001
    """Persist a list of rules to a JSON file (G2.3)."""
    import json
    from pathlib import Path

    out = [
        {
            "name": r.name,
            "body": [list(p) for p in r.body],
            "head": list(r.head),
            "distinct": [list(d) for d in r.distinct],
        }
        for r in rules
    ]
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


def load_rules_json(path) -> list[Rule]:  # noqa: ANN001
    """Inverse of ``save_rules_json``; raises on a missing or malformed file."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rules: list[Rule] = []
    for r in data:
        body = tuple(tuple(p) for p in r["body"])
        head = tuple(r["head"])
        distinct = tuple(tuple(d) for d in r.get("distinct", []))
        rules.append(Rule(name=r["name"], body=body, head=head, distinct=distinct))
    return rules


@dataclass(frozen=True)
class Rule:
    """An IF-THEN rule: a conjunction of body patterns implies one head pattern.

    Variables begin with '?'; relations are constants. Range-restricted: every
    head variable must occur in the body. `distinct` lists variable pairs that
    must bind to different entities — used to forbid spurious self-loop
    derivations (e.g. `ancestor_of(x, x)`) from compositional rules.
    """

    name: str
    body: tuple[Pattern, ...]
    head: Pattern
    distinct: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        body_vars = {tok for pat in self.body for tok in pat if _is_var(tok)}
        head_vars = {tok for tok in self.head if _is_var(tok)}
        if not head_vars <= body_vars:
            raise ValueError(f"rule {self.name!r} is not range-restricted")


@dataclass
class Derivation:
    triple: Triple
    rule: str
    premises: tuple[Triple, ...]


@dataclass
class SymbolicResult:
    answered: bool
    answers: list[str] = field(default_factory=list)
    proof: list[str] = field(default_factory=list)


def _unify(pattern: Pattern, triple: Triple, binding: dict[str, str]) -> dict[str, str] | None:
    # ⚡ Bolt Optimization: Unpack 3-tuples and delay dict allocation to avoid overhead in hot path
    p0, p1, p2 = pattern
    t0, t1, t2 = triple

    if not p0.startswith("?"):
        if p0 != t0:
            return None
    elif p0 in binding and binding[p0] != t0:
        return None

    if not p1.startswith("?"):
        if p1 != t1:
            return None
    elif p1 == p0:
        if t1 != t0:
            return None
    elif p1 in binding and binding[p1] != t1:
        return None

    if not p2.startswith("?"):
        if p2 != t2:
            return None
    elif p2 == p0:
        if t2 != t0:
            return None
    elif p2 == p1:
        if t2 != t1:
            return None
    elif p2 in binding and binding[p2] != t2:
        return None

    out = binding.copy()
    if p0.startswith("?"):
        out[p0] = t0
    if p1.startswith("?"):
        out[p1] = t1
    if p2.startswith("?"):
        out[p2] = t2
    return out


def _ground(pattern: Pattern, binding: dict[str, str]) -> Triple:
    # ⚡ Bolt Optimization: Avoid generator expression for 3-tuple
    p0, p1, p2 = pattern
    return (
        binding.get(p0, p0),
        binding.get(p1, p1),
        binding.get(p2, p2),
    )


class RuleEngine:
    """Forward-chaining engine; materialises the closure to a fixpoint."""

    def __init__(
        self, ontology: Ontology, rules: list[Rule] | None = None, max_iterations: int = 100
    ) -> None:
        self.ontology = ontology
        self.rules: list[Rule] = list(rules or [])
        self.max_iterations = max_iterations
        self._closure: set[Triple] = set()
        self._by_head: dict[tuple[str, str], set[str]] = {}
        self._derivations: dict[Triple, Derivation] = {}

    def add_rule(self, rule: Rule) -> bool:
        """Register a rule (used by the distillation loop). Returns True if new.

        Enforces the ontology-preservation premise of Theorem 1: a rule is only
        registered if it is *type-consistent* with the ontology, so the closure
        the engine materialises can never contain an edge the ontology rejects.
        A rule is rejected when (a) it names a relation the ontology does not
        declare, or (b) its body/head atoms impose contradictory types on a
        shared variable (e.g. a variable forced to be both a City and a Person).
        An *untyped* ontology (no declared relation types) constrains nothing and
        so accepts every range-restricted rule, leaving the distillation loop and
        schema-free graphs unaffected.
        """
        if any(r.name == rule.name for r in self.rules):
            return False
        if not self._ontology_consistent(rule):
            return False
        self.rules.append(rule)
        return True

    def _ontology_consistent(self, rule: Rule) -> bool:
        """True iff `rule` preserves the ontology's relation and type schema.

        Enforces the *sufficient* condition for Theorem 1's ontology-preservation
        premise: for every head variable, the node types the rule body can bind
        it to must be a **subset** of the types the head relation admits at that
        position. A mere non-empty intersection (that *some* shared type exists)
        is necessary but not sufficient -- if a body relation's domain is wider
        than the head's (e.g. ``manages`` ranging over ``{Person, Robot}`` feeding
        a ``boss`` head restricted to ``{Person}``), the body can still bind the
        head variable to an off-type entity and leak a type-violating fact into
        the closure.

        Body- and head-induced types are tracked separately. Each is the
        intersection of the ``RelationType.domain`` / ``range`` constraints over
        the atoms a variable occurs in (the same primitive ``Ontology.validate``
        uses for edges); a wildcard ``"*"`` position imposes no constraint. The
        rule is rejected when (a) it names a relation the ontology does not
        declare, (b) body and head force a shared variable into an empty type set
        (an unsatisfiable rule), or (c) a head variable's body-induced types are
        not contained in the types its head position requires. An *untyped*
        ontology (no declared relation types) constrains nothing and so accepts
        every range-restricted rule.
        """
        relation_types = self.ontology.relation_types
        if not relation_types:  # untyped ontology constrains nothing
            return True

        # Admissible node types per variable, kept separate for body vs head so
        # the body-types-subset-of-head-types check below is well defined. A
        # variable absent from a map is unconstrained there (wildcard / unseen).
        body_types: dict[str, frozenset[str]] = {}
        head_types: dict[str, frozenset[str]] = {}

        def narrow(into: dict[str, frozenset[str]], var: str, allowed: frozenset[str]) -> None:
            if not _is_var(var) or "*" in allowed:  # wildcard imposes no constraint
                return
            into[var] = allowed if var not in into else (into[var] & allowed)

        for s, r, o in rule.body:
            rt = relation_types.get(r)
            if rt is None:  # relation unknown to the ontology -> inconsistent
                return False
            narrow(body_types, s, rt.domain)
            narrow(body_types, o, rt.range)

        hs, hr, ho = rule.head
        head_rt = relation_types.get(hr)
        if head_rt is None:
            return False
        narrow(head_types, hs, head_rt.domain)
        narrow(head_types, ho, head_rt.range)

        # (b) Unsatisfiable rule: a variable forced into an empty type set by the
        # combined body+head constraints can never bind, so the rule is malformed.
        for var in body_types.keys() | head_types.keys():
            constraints = [s for s in (body_types.get(var), head_types.get(var)) if s is not None]
            if constraints and not frozenset.intersection(*constraints):
                return False

        # (c) Ontology preservation: every head variable's body-induced types must
        # be contained in the types its head position admits. If the body leaves a
        # head variable unconstrained while the head requires a specific type, the
        # body can bind an off-type entity -> reject.
        for var, required in head_types.items():
            induced = body_types.get(var)
            if induced is None or not (induced <= required):
                return False
        return True

    # ---------------------------------------------------------- axiom rules
    def _axiom_rules(self) -> list[Rule]:
        rules: list[Rule] = []
        for rt in self.ontology.relation_types.values():
            r = rt.name
            if rt.symmetric:
                rules.append(Rule(f"ax:sym:{r}", (("?x", r, "?y"),), ("?y", r, "?x")))
            if rt.transitive:
                rules.append(
                    Rule(f"ax:trans:{r}", (("?x", r, "?y"), ("?y", r, "?z")), ("?x", r, "?z"))
                )
            if rt.inverse_of:
                inv = rt.inverse_of
                rules.append(Rule(f"ax:inv:{r}:{inv}", (("?x", r, "?y"),), ("?y", inv, "?x")))
        return rules

    # ---------------------------------------------------------- matching
    @staticmethod
    def _join(
        body: tuple[Pattern, ...],
        idx_all: dict[str, list[Triple]],
        idx_subj: dict[tuple[str, str], list[Triple]],
        idx_obj: dict[tuple[str, str], list[Triple]],
    ) -> Iterator[dict[str, str]]:
        """Relational join of the body patterns, produced one binding at a time.

        Each atom is matched only against facts of its relation, and — when its
        subject or object is already bound — against the index keyed on that
        value, so a dense relation does not force a quadratic scan.

        Bindings are yielded lazily. Materialising a level at a time is what
        made a high-fanout relation exhaust memory during rule mining: the
        intermediate result set is the product of the fanouts, while the caller
        only ever needs one row. Depth-first traversal emits rows in the same
        order the level-by-level version did, so the derivation a fact is
        recorded with — and therefore its proof tree — is unchanged.
        """
        # ⚡ Bolt Optimization: Pre-compute static checks outside the recursive generator
        is_var_s = [_is_var(p[0]) for p in body]
        is_var_o = [_is_var(p[2]) for p in body]

        def extend(depth: int, binding: dict[str, str]) -> Iterator[dict[str, str]]:
            if depth == len(body):
                yield binding
                return
            s, r, o = body[depth]
            s_val = binding.get(s) if is_var_s[depth] else s
            o_val = binding.get(o) if is_var_o[depth] else o

            # ⚡ Bolt Optimization: Avoid allocating empty lists on miss via direct .get()
            if s_val is not None:
                candidates: list[Triple] | None = idx_subj.get((r, s_val))
            elif o_val is not None:
                candidates = idx_obj.get((r, o_val))
            else:
                candidates = idx_all.get(r)

            if candidates is not None:
                for fact in candidates:
                    merged = _unify((s, r, o), fact, binding)
                    if merged is not None:
                        yield from extend(depth + 1, merged)

        return extend(0, {})

    # ---------------------------------------------------------- closure
    def materialise(self, graph: WorldGraph) -> set[Triple]:
        """Compute the deductive closure of `graph`. Idempotent."""
        facts: set[Triple] = set(graph.triples())
        self._derivations = {}
        all_rules = self._axiom_rules() + self.rules

        converged = False
        for _ in range(self.max_iterations):
            added = False
            idx_all: dict[str, list[Triple]] = {}
            idx_subj: dict[tuple[str, str], list[Triple]] = {}
            idx_obj: dict[tuple[str, str], list[Triple]] = {}
            for fact in facts:
                h, r, t = fact

                # ⚡ Bolt Optimization: explicit containment checks instead of setdefault
                # avoid redundant empty list allocations for every dictionary lookup
                if r in idx_all:
                    idx_all[r].append(fact)
                else:
                    idx_all[r] = [fact]

                k_subj = (r, h)
                if k_subj in idx_subj:
                    idx_subj[k_subj].append(fact)
                else:
                    idx_subj[k_subj] = [fact]

                k_obj = (r, t)
                if k_obj in idx_obj:
                    idx_obj[k_obj].append(fact)
                else:
                    idx_obj[k_obj] = [fact]

            for rule in all_rules:
                for binding in self._join(rule.body, idx_all, idx_subj, idx_obj):
                    if any(binding.get(a) == binding.get(b) for a, b in rule.distinct):
                        continue
                    derived = _ground(rule.head, binding)
                    if derived in facts:
                        continue
                    facts.add(derived)
                    self._derivations[derived] = Derivation(
                        derived, rule.name, tuple(_ground(p, binding) for p in rule.body)
                    )
                    added = True
            if not added:
                converged = True
                break

        if not converged:
            # The closure has not reached a least fixpoint within the iteration
            # cap, so it may be incomplete. Returning it silently would violate
            # the completeness the cascade relies on (Proposition 1: Tier 1
            # abstains only when an answer is not entailed), so we fail loudly
            # instead. Shipped workloads (ontology axioms + length-<=2 mined
            # rules) converge in a few passes; deep hand-authored recursive rule
            # sets should raise ``max_iterations``.
            raise RuntimeError(
                "RuleEngine.materialise did not reach a fixpoint within "
                f"max_iterations={self.max_iterations}; the closure may be "
                "incomplete. Increase max_iterations for deeper recursive rules."
            )

        self._closure = facts
        index: dict[tuple[str, str], set[str]] = {}
        for h, r, t in facts:
            k_idx = (h, r)
            if k_idx in index:
                index[k_idx].add(t)
            else:
                index[k_idx] = {t}
        self._by_head = index
        return facts

    # ---------------------------------------------------------- query
    def query(self, head: str, relation: str) -> SymbolicResult:
        """Answer 'tails of (head, relation)?' from the closure. Abstain if empty."""
        answers = sorted(self._by_head.get((head, relation), ()))
        if not answers:
            return SymbolicResult(answered=False)
        proof: list[str] = []
        for tail in answers:
            proof.extend(self._explain((head, relation, tail)))
        return SymbolicResult(answered=True, answers=answers, proof=proof)

    def add_fact(self, triple: Triple) -> bool:
        """Insert a base fact directly into the closure (cheap, no re-derivation).

        Used for Tier-3 fact write-back so an identical query immediately hits
        Tier 1. Full consequences are recomputed by the next `materialise`.
        """
        if triple in self._closure:
            return False
        self._closure.add(triple)
        h, r, t = triple
        k_head = (h, r)
        if k_head in self._by_head:
            self._by_head[k_head].add(t)
        else:
            self._by_head[k_head] = {t}
        return True

    def known_rule_names(self) -> set[str]:
        """Names of every rule the engine may legitimately cite in a proof.

        This is the union of the synthesised/axiom rule names (derived from the
        ontology) and the explicitly registered rules. A proof step is only
        *grounded* when it is a base fact or names a rule in this set.
        """
        return {r.name for r in self._axiom_rules()} | {r.name for r in self.rules}

    def proof_is_grounded(self, triple: Triple) -> bool:
        """True if every step of `triple`'s proof reduces to a base fact or a
        known rule (no dangling / unsupported derivation).

        Walks the proof tree structurally via ``self._derivations``: a triple is
        grounded when it is a base leaf (absent from ``_derivations``) or when its
        derivation cites a known rule and every premise is itself grounded. This
        checks provenance *integrity*, not ground-truth correctness of the answer.
        """
        known = self.known_rule_names()
        seen: set[Triple] = set()

        def _walk(t: Triple) -> bool:
            if t in seen:  # cycle guard (closure is finite)
                return True
            seen.add(t)
            deriv = self._derivations.get(t)
            if deriv is None:  # base fact leaf
                return True
            if deriv.rule not in known:  # dangling / unknown-rule step
                return False
            return all(_walk(p) for p in deriv.premises)

        return _walk(triple)

    def _explain(
        self, triple: Triple, depth: int = 0, seen: frozenset[Triple] = frozenset()
    ) -> list[str]:
        h, r, t = triple
        pad = "  " * depth
        deriv = self._derivations.get(triple)
        if deriv is None:
            return [f"{pad}FACT     {h} -{r}-> {t}"]
        if triple in seen:  # cycle guard (mirrors proof_is_grounded): the
            # provenance can be cyclic (e.g. symmetric rules), so stop
            # unfolding an already-visited triple instead of recursing forever.
            return [f"{pad}DERIVED  {h} -{r}-> {t}   [{deriv.rule}] (cycle)"]
        lines = [f"{pad}DERIVED  {h} -{r}-> {t}   [{deriv.rule}]"]
        seen = seen | {triple}
        for premise in deriv.premises:
            lines.extend(self._explain(premise, depth + 1, seen))
        return lines

    @property
    def closure(self) -> set[Triple]:
        return set(self._closure)
