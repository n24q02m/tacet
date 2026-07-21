"""Concept formation — induce node types and new relations from graph data.

Two complementary inductions:

* ``induce_node_types`` — clusters nodes by their *structural signature*
  (which relations they participate in, in which direction). Nodes whose
  signature is the same are probably the same kind of thing. Useful when
  ingesting a corpus with no declared ontology, or when the declared
  ontology is incomplete and the analyst wants to discover a missing class.
* ``induce_relations`` — finds frequent length-2 *path types* and proposes
  them as candidate new relations: ``R(x, y) := R_1(x, z) ∧ R_2(z, y)``.
  This is the unsupervised counterpart of ``tacet.distill.mine_rules``
  (which mines rules over a *given* target); here we let the data tell us
  *what target* is worth naming.

Both are deliberately conservative: a hand-on-the-wheel analyst inspects
the proposals before extending the ontology. ``revise_ontology`` is the
convenience that applies both inductions to an existing ``Ontology``.

Out of scope: discriminative type learning (training a classifier), the
hard "concept" question in cognitive science, and adversarial-data
robustness — these are open research directions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType


@dataclass
class InducedType:
    name: str
    members: list[str]
    signature: dict[str, int]  # (relation, direction) -> degree

    def describe(self) -> str:
        top = sorted(self.signature.items(), key=lambda kv: -kv[1])[:5]
        sig = ", ".join(f"{r}:{n}" for r, n in top)
        return f"{self.name} ({len(self.members)} nodes; {sig})"


@dataclass
class InducedRelation:
    name: str  # e.g. "R1+R2"
    body: tuple[tuple[str, str, str], ...]  # the path pattern
    support: int  # distinct (x,y) it covers


# ---------------------------------------------------------------------------
def _node_signature(graph: WorldGraph, node_id: str) -> dict[tuple[str, str], int]:
    """Structural signature: (relation, direction) -> degree."""
    sig: dict[tuple[str, str], int] = {}
    # ⚡ Bolt Optimization: Traverse only populated adjacency entries using
    # new public accessors instead of querying O(|Relations|) times.
    # Expected impact: Reduces complexity from O(|Relations| * Degree) to O(Degree).
    for rel, targets in graph.out_relations(node_id).items():
        if targets:
            sig[(rel, "out")] = len(targets)
    for rel, sources in graph.in_relations(node_id).items():
        if sources:
            sig[(rel, "in")] = len(sources)
    return sig


def induce_node_types(
    graph: WorldGraph, k: int | None = None, max_k: int = 8, seed: int = 0
) -> list[InducedType]:
    """k-means over structural signatures; picks ``k`` by simple elbow if not given.

    Returns one ``InducedType`` per discovered cluster.
    """
    rng = np.random.default_rng(seed)
    nodes = graph.entities()
    if not nodes:
        return []
    relations = sorted(graph.relations())
    cols = [(r, d) for r in relations for d in ("out", "in")]
    if not cols:
        return [InducedType(name="Cluster_0", members=nodes, signature={})]
    col_index = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(nodes), len(cols)))
    for row, n in enumerate(nodes):
        sig = _node_signature(graph, n)
        for (rel, direc), deg in sig.items():
            if (rel, direc) in col_index:
                X[row, col_index[(rel, direc)]] = deg
    # presence vector (binary) often clusters node TYPES better than raw degree
    X = (X > 0).astype(float)
    chosen_k = k or _elbow_k(X, max_k=max_k, rng=rng)
    labels, centroids = _kmeans(X, chosen_k, rng=rng)
    out: list[InducedType] = []
    for cluster in range(chosen_k):
        members = [nodes[i] for i in range(len(nodes)) if labels[i] == cluster]
        if not members:
            continue
        cluster_sig: dict[str, int] = {}
        for (rel, direc), idx in col_index.items():
            if float(centroids[cluster, idx]) >= 0.5:
                cluster_sig[f"{direc}.{rel}"] = 1
        out.append(InducedType(name=f"Cluster_{cluster}", members=members, signature=cluster_sig))
    return out


def _kmeans(X: np.ndarray, k: int, n_iter: int = 30, rng=None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng if rng is not None else np.random.default_rng(0)
    if len(X) <= k:
        labels = np.arange(len(X)) % k
        centroids = np.zeros((k, X.shape[1]))
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
        return labels, centroids
    # k-means++ initial seeding (uniform fallback when all points coincide).
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(len(X))]
    for i in range(1, k):
        d2 = np.min(((X[:, None] - centroids[:i]) ** 2).sum(-1), axis=1)
        total = float(d2.sum())
        if total <= 1e-12:
            centroids[i] = X[rng.integers(len(X))]
        else:
            probs = d2 / total
            probs = probs / probs.sum()  # renormalise to defeat fp drift
            centroids[i] = X[rng.choice(len(X), p=probs)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None] - centroids) ** 2).sum(-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return labels, centroids


def _elbow_k(X: np.ndarray, max_k: int, rng) -> int:
    """Pick k by where inertia stops dropping fast. Cheap heuristic."""
    inertias: list[float] = []
    for k in range(1, min(max_k, len(X)) + 1):
        labels, centroids = _kmeans(X, k, rng=rng)
        d2 = ((X - centroids[labels]) ** 2).sum(-1)
        inertias.append(float(d2.sum()))
    if len(inertias) < 3:
        return len(inertias)
    deltas = -np.diff(inertias)
    # k where the next drop is < 30% of the previous one
    for i in range(1, len(deltas)):
        if deltas[i] < 0.3 * deltas[i - 1]:
            return i + 1
    return len(inertias)


# ---------------------------------------------------------------------------
def induce_relations(
    graph: WorldGraph, min_support: int = 10, max_proposals: int = 10
) -> list[InducedRelation]:
    """Mine length-2 path patterns ``R1(x,z) ∧ R2(z,y)`` with high support.

    Proposals are sorted by support descending. The caller decides whether
    to *name* and *promote* the pattern into the ontology — this function
    only surfaces candidates.
    """
    rel_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for e in graph.edges:
        rel_pairs[e.relation].add((e.source, e.target))

    counts: Counter = Counter()
    bodies: dict[str, tuple[tuple[str, str, str], ...]] = {}
    relations = sorted(rel_pairs)

    # Pre-compute forward adjacency maps for all relations to avoid O(N^2) recreation
    forward_maps: dict[str, dict[str, set[str]]] = {}
    for r in relations:
        fwd = defaultdict(set)
        for s, t in rel_pairs[r]:
            fwd[s].add(t)
        forward_maps[r] = fwd

    for r1 in relations:
        forward1 = forward_maps[r1]
        for r2 in relations:
            forward2 = forward_maps[r2]
            covered: set[tuple[str, str]] = set()
            for x, zs in forward1.items():
                for z in zs:
                    for y in forward2.get(z, ()):
                        if x != y:
                            covered.add((x, y))
            if len(covered) >= min_support:
                name = f"{r1}+{r2}"
                counts[name] = len(covered)
                bodies[name] = (("?x", r1, "?z"), ("?z", r2, "?y"))

    out: list[InducedRelation] = []
    for name, support in counts.most_common(max_proposals):
        out.append(InducedRelation(name=name, body=bodies[name], support=support))
    return out


# ---------------------------------------------------------------------------
def revise_ontology(
    graph: WorldGraph,
    ontology: Ontology,
    *,
    apply_types: bool = True,
    apply_relations: bool = True,
    min_support: int = 10,
) -> dict[str, list[str]]:
    """Extend `ontology` with discovered types / relation candidates.

    Returns ``{"added_types": [...], "added_relations": [...]}``. Conservative
    by default: declared types / relations are never overwritten.
    """
    added_types: list[str] = []
    added_relations: list[str] = []
    if apply_types:
        for it in induce_node_types(graph):
            if it.name not in ontology.node_types:
                ontology.add_node_type(NodeType(it.name))
                added_types.append(it.name)
    if apply_relations:
        for ir in induce_relations(graph, min_support=min_support):
            if ir.name not in ontology.relation_types:
                ontology.add_relation_type(RelationType(name=ir.name))
                added_relations.append(ir.name)
    return {"added_types": added_types, "added_relations": added_relations}


__all__ = [
    "InducedRelation",
    "InducedType",
    "induce_node_types",
    "induce_relations",
    "revise_ontology",
]
