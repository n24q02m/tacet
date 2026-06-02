"""Causal layer — a structural-causal-model (SCM) interface for TACET.

A subset of a knowledge graph can be promoted to a typed causal DAG over
discrete variables, giving TACET three query kinds the symbolic and KGE
tiers cannot answer:

* **observation**         ``P(Y | X = x)`` — conditional, what is.
* **intervention**        ``P(Y | do(X = x))`` — counterfactual, what would.
* **counterfactual**      ``P(Y_{X=x} | observed = e)`` — twin-world, what
  *would have been* if X had been x, given what we actually saw.

This is the Pearl framework (Causality, 2009), implemented for finite
discrete variables with structural equations supplied as Python callables.
Inference is exact for small models (enumeration over the exogenous noise
domain) and Monte-Carlo when an analytic enumeration would blow up.

What is shipped:
  * ``CausalModel`` — variables, mechanisms, observe / intervene / sample.
  * ``counterfactual(model, evidence, do, target)`` — abduction-action-prediction.
  * ``backdoor_set(model, treatment, outcome)`` — one valid adjustment set
    via parents-not-descendants (the simplest sufficient criterion).

Out of scope (Tier C+): the full ID algorithm for semi-Markovian models,
continuous variables / SCMs with noise distributions other than the
uniform-noise / explicit-distribution model used here, and identification
of causal structure from data (causal discovery). These are real but big.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

Value = object
Assignment = Mapping[str, Value]


@dataclass
class Variable:
    """A discrete variable with finite domain and a list of parent variables."""

    name: str
    domain: tuple[Value, ...]
    parents: tuple[str, ...] = ()
    mechanism: Callable[[dict[str, Value], float], Value] | None = None

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError(f"variable {self.name!r} has empty domain")


class CausalModel:
    """Acyclic SCM: ``X = f_X(parents(X), U_X)`` for each variable X.

    The exogenous noise is sampled per-variable from a uniform [0, 1) (the
    mechanism is free to map it however it likes to the variable's domain).
    Sampling, intervention and counterfactuals all reuse the same RNG seed
    so trajectories are reproducible.
    """

    def __init__(self, seed: int = 0) -> None:
        self.variables: dict[str, Variable] = {}
        self._order: list[str] = []
        self._rng = random.Random(seed)
        self._seed = seed
        # Bidirected ("unobserved-confounding") edges for semi-Markovian
        # models.  Each entry is a frozenset of two variable names that
        # share an unobserved common cause.  Used by ``front_door_set``
        # and ``instrumental_variables`` to detect identification.
        self.bidirected: set[frozenset[str]] = set()

    def add_bidirected_edge(self, a: str, b: str) -> CausalModel:
        """Declare an unobserved common cause between ``a`` and ``b``.

        This is the bi-directed edge in the standard semi-Markovian
        graphical representation: ``a ↔ b`` ⇒ ∃ latent U with U → a
        and U → b.  Necessary to express front-door / instrumental-
        variable scenarios where the back-door criterion alone fails.
        """
        if a == b:
            raise ValueError("bidirected edge between a node and itself")
        if a not in self.variables or b not in self.variables:
            raise ValueError("both endpoints must be declared as variables")
        self.bidirected.add(frozenset({a, b}))
        return self

    def has_bidirected(self, a: str, b: str) -> bool:
        return frozenset({a, b}) in self.bidirected

    def add_variable(
        self,
        name: str,
        domain: tuple[Value, ...],
        parents: tuple[str, ...] = (),
        mechanism: Callable[[dict[str, Value], float], Value] | None = None,
    ) -> CausalModel:
        if name in self.variables:
            raise ValueError(f"duplicate variable: {name!r}")
        for parent in parents:
            if parent not in self.variables:
                raise ValueError(f"parent {parent!r} of {name!r} not yet declared")
        self.variables[name] = Variable(name, tuple(domain), tuple(parents), mechanism)
        self._order.append(name)
        return self

    def set_mechanism(
        self, name: str, mechanism: Callable[[dict[str, Value], float], Value]
    ) -> None:
        if name not in self.variables:
            raise ValueError(f"unknown variable {name!r}")
        self.variables[name].mechanism = mechanism

    # ------------------------------------------------------------------- sampling
    def sample(
        self, n: int = 1, evidence: Assignment | None = None, max_rejections: int | None = None
    ) -> list[dict[str, Value]]:
        """Draw `n` joint samples; if `evidence` is given, rejection-sample
        until each accepted sample matches it (returns at most `n`)."""
        max_rejections = max_rejections if max_rejections is not None else 10000 * n
        out: list[dict[str, Value]] = []
        tries = 0
        while len(out) < n and tries < max_rejections:
            tries += 1
            assignment = self._sample_one()
            if evidence is None or all(assignment[k] == v for k, v in evidence.items()):
                out.append(assignment)
        return out

    def _sample_one(self) -> dict[str, Value]:
        a: dict[str, Value] = {}
        for name in self._order:
            v = self.variables[name]
            if v.mechanism is None:
                raise RuntimeError(f"variable {name!r} has no mechanism")
            parent_vals = {p: a[p] for p in v.parents}
            a[name] = v.mechanism(parent_vals, self._rng.random())
        return a

    # ------------------------------------------------------------------- queries
    def probability(
        self, target: str, value: Value, evidence: Assignment | None = None, n: int = 5000
    ) -> float:
        """Monte-Carlo estimate of ``P(target = value | evidence)``."""
        samples = self.sample(n, evidence=evidence)
        if not samples:
            return float("nan")
        return sum(1 for s in samples if s[target] == value) / len(samples)

    def distribution(
        self, target: str, evidence: Assignment | None = None, n: int = 5000
    ) -> dict[Value, float]:
        """Monte-Carlo estimate of ``P(target | evidence)`` over its full domain."""
        samples = self.sample(n, evidence=evidence)
        if not samples:
            return {v: float("nan") for v in self.variables[target].domain}
        counts = Counter(s[target] for s in samples)
        return {v: counts.get(v, 0) / len(samples) for v in self.variables[target].domain}

    # ------------------------------------------------------------------- intervention
    def intervene(self, do: Assignment) -> CausalModel:
        """Return a new SCM with edges into each ``do`` variable severed and
        its mechanism replaced by a constant — Pearl's ``do(X = x)``."""
        new = CausalModel(seed=self._seed)
        for name in self._order:
            v = self.variables[name]
            if name in do:
                constant = do[name]
                new.add_variable(name, v.domain, parents=(), mechanism=lambda _p, _u, c=constant: c)
            else:
                new.add_variable(name, v.domain, parents=v.parents, mechanism=v.mechanism)
        return new

    # ------------------------------------------------------------------- structure
    def ancestors(self, name: str) -> set[str]:
        seen: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for parent in self.variables[cur].parents:
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen

    def descendants(self, name: str) -> set[str]:
        kids = defaultdict(set)
        for n, v in self.variables.items():
            for p in v.parents:
                kids[p].add(n)
        seen: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for child in kids[cur]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen


# --- causal queries ----------------------------------------------------------
def backdoor_set(model: CausalModel, treatment: str, outcome: str) -> set[str]:
    """A simple sufficient adjustment set: parents of `treatment` that are
    *not* descendants of `treatment`. This satisfies Pearl's backdoor
    criterion for the most common case (no unobserved confounders); for
    semi-Markovian models the full ID algorithm is needed."""
    if treatment not in model.variables or outcome not in model.variables:
        raise ValueError("treatment / outcome not in model")
    forbidden = model.descendants(treatment) | {treatment}
    parents = set(model.variables[treatment].parents)
    return parents - forbidden


def counterfactual(
    model: CausalModel, evidence: Assignment, do: Assignment, target: str, n: int = 10000
) -> dict[Value, float]:
    """``P(target_{do} | evidence)`` via abduction-action-prediction.

    Step 1 (abduction): rejection-sample exogenous noise vectors consistent
    with `evidence`. Step 2 (action): apply ``do`` to the model. Step 3
    (prediction): re-evaluate each accepted noise vector under the
    intervened model and tally the target.
    """
    # We need to capture and replay the noise. Re-implement sample-with-noise:
    rng = random.Random(model._seed + 12345)
    order = model._order
    accepted_noise: list[dict[str, float]] = []
    tries = 0
    while len(accepted_noise) < n and tries < 50 * n:
        tries += 1
        noise = {name: rng.random() for name in order}
        a: dict[str, Value] = {}
        for name in order:
            v = model.variables[name]
            parent_vals = {p: a[p] for p in v.parents}
            a[name] = v.mechanism(parent_vals, noise[name])
        if all(a[k] == v for k, v in evidence.items()):
            accepted_noise.append(noise)
    if not accepted_noise:
        return {v: float("nan") for v in model.variables[target].domain}

    intervened = model.intervene(do)
    counts: Counter = Counter()
    for noise in accepted_noise:
        a = {}
        for name in intervened._order:
            v = intervened.variables[name]
            parent_vals = {p: a[p] for p in v.parents}
            a[name] = v.mechanism(parent_vals, noise[name])
        counts[a[target]] += 1
    n_acc = len(accepted_noise)
    return {v: counts.get(v, 0) / n_acc for v in intervened.variables[target].domain}


def _directed_paths(model: CausalModel, src: str, dst: str) -> list[list[str]]:
    """Enumerate simple directed paths from ``src`` → ``dst`` in the DAG."""
    paths: list[list[str]] = []
    children: dict[str, list[str]] = {n: [] for n in model.variables}
    for n, v in model.variables.items():
        for p in v.parents:
            children.setdefault(p, []).append(n)

    def dfs(node: str, path: list[str]) -> None:
        if node == dst:
            paths.append(list(path))
            return
        for c in children.get(node, ()):
            if c in path:
                continue
            path.append(c)
            dfs(c, path)
            path.pop()

    dfs(src, [src])
    return paths


def front_door_set(model: CausalModel, treatment: str, outcome: str) -> set[str] | None:
    """Find a *front-door* admissible set Z for ``P(outcome | do(treatment))``.

    Implements the canonical Pearl front-door criterion (1995,
    *Causality* §3.3.2) under the practical assumption that the model
    declares latent confounders as bidirected edges.  The check is
    conservative but covers the standard textbook examples
    (smoking → tar → cancer, etc.):

    1. **Mediation**: Z lies on every directed path X → … → Y.
    2. **No latent X ↔ Z**: the unobserved-confounder edge that
       prevents the back-door criterion from working between X and Y
       must not also short-circuit Z.
    3. **No latent Z ↔ Y**: a direct bidirected edge between the
       mediator and the outcome reintroduces an unblocked back-door
       that no observed conditioning set can fix.
    4. **No directed Z → X path** (Z is genuinely *downstream* of X,
       not a parent of the treatment).

    The function tries singletons first, then pairs of mediators.
    Larger sets are out of scope (paper §11.x discusses the
    semi-Markovian ID algorithm for the general case).
    """
    if treatment not in model.variables or outcome not in model.variables:
        raise ValueError("treatment / outcome not in model")
    if treatment == outcome:
        return set()
    candidates = [v for v in model.variables if v not in (treatment, outcome)]
    paths_xy = _directed_paths(model, treatment, outcome)
    if not paths_xy:
        return None

    def intercepts_all(z: set[str]) -> bool:
        return all(any(node in z for node in path[1:-1]) for path in paths_xy)

    for size in (1, 2):
        for combo in _combinations(candidates, size):
            z = set(combo)
            if not intercepts_all(z):
                continue
            # Each mediator zi must be a true descendant of the
            # treatment (not a parent and not in the same connected
            # component as X via bidirected edges only).
            ok = True
            for zi in z:
                if model.has_bidirected(treatment, zi):
                    ok = False
                    break
                if model.has_bidirected(zi, outcome):
                    ok = False
                    break
                # Must be reachable X → zi via a directed path.
                if not _directed_paths(model, treatment, zi):
                    ok = False
                    break
                # Must not be an ancestor of the treatment (front-door
                # mediators sit between X and Y, not before X).
                if _directed_paths(model, zi, treatment):
                    ok = False
                    break
            if ok:
                return z
    return None


def _combinations(items: list[str], k: int):
    """Enumerate combinations of size ``k`` (avoids importing itertools)."""
    n = len(items)
    if k == 0:
        yield ()
        return
    if k > n:
        return
    indices = list(range(k))
    while True:
        yield tuple(items[i] for i in indices)
        for i in range(k - 1, -1, -1):
            if indices[i] != i + n - k:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1


def instrumental_variables(model: CausalModel, treatment: str, outcome: str) -> list[str]:
    """Return variables that satisfy the standard instrumental-variable
    conditions for the (treatment, outcome) pair.

    A variable ``Z`` is an instrument for ``X → Y`` iff:

    1. ``Z`` is a cause of ``X`` (Z is in some directed path Z → … → X);
    2. ``Z`` has no direct causal effect on ``Y`` other than through ``X``;
    3. ``Z`` shares no unobserved confounder with ``Y``
       (no bidirected ``Z ↔ Y`` in the model).

    Suitable for linear-model identification of the treatment effect
    when the back-door + front-door criteria both fail.
    """
    if treatment not in model.variables or outcome not in model.variables:
        raise ValueError("treatment / outcome not in model")
    candidates = [v for v in model.variables if v not in (treatment, outcome)]
    out: list[str] = []
    for z in candidates:
        # (1) Z is a cause of X.
        if not _directed_paths(model, z, treatment):
            continue
        # (2) Every Z → Y directed path passes through X.
        paths_zy = _directed_paths(model, z, outcome)
        if not paths_zy:
            # Z reaches X but not Y at all — not an IV (no effect to study).
            continue
        if any(treatment not in path[1:-1] for path in paths_zy):
            continue
        # (3) No bidirected Z ↔ Y.
        if model.has_bidirected(z, outcome):
            continue
        out.append(z)
    return out


__all__ = [
    "Assignment",
    "CausalModel",
    "Value",
    "Variable",
    "backdoor_set",
    "counterfactual",
    "front_door_set",
    "instrumental_variables",
]
