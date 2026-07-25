"""Formal Concept Analysis for ontology revision (TACET §11.x / G1.4).

This module provides tools for formal concept lattice analysis on the
world graph, offering a fundamentally different principle from the
k-means clustering in ``tacet.distill.concepts``:

* **k-means** (currently used): partitions entities into *k* disjoint
  groups based on vector signatures.  Simple and scalable, but it
  distorts concept boundaries and cannot recover hierarchy (Person ⊇
  Employee ⊇ Manager).
* **FCA** (Wille 1982): every set of entities sharing the same set of
  attributes forms a *concept* ``(extent, intent)``.  The set of all
  concepts forms a **complete lattice** with a natural parent-child
  ordering (concept A ≤ B iff extent A ⊆ extent B).  This lattice is
  precisely the concept hierarchy by structure.

In the KG context:

* **Object** ``g ∈ G`` = an entity of the world graph.
* **Attribute** ``(r, t) ∈ M`` = "has an edge r to entity t".
* **Incidence** ``g I (r, t)`` ↔ edge ``(g, r, t)`` exists in the graph.

The lattice of the context ``(G, M, I)`` yields every entity cluster
that is "defined" by a shared attribute pattern.  The top concept
(∅ intent → all entities) and bottom concept (∅ extent → all
attributes) are the two extremes; in between sit the "natural"
concepts.

Main API:

* ``FormalContext.from_graph(graph)`` — build a context from a WorldGraph.
* ``ctx.closure_of_extent(A)`` / ``closure_of_intent(B)`` — closure
  operators (Galois connection).
* ``ctx.concepts()`` — enumerate all concepts (NextClosure, stable for
  contexts of a few hundred objects × a few hundred attributes;
  small-to-medium KGs).
* ``ctx.lattice_edges()`` — list of direct-cover edges (Hasse diagram)
  for visualisation or downstream reasoning.

The computation is polynomial in the number of concepts, but the
number of concepts is worst-case exponential in ``|G| + |M|``.  For
KG contexts with a few thousand entities, filtering attributes
(min_support) before calling ``concepts()`` is recommended.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid mandatory import cycle for type-only use
    from tacet.core.graph import WorldGraph


Attribute = tuple[str, str]  # (relation, tail)
ExtentIntent = tuple[frozenset[str], frozenset[Attribute]]


@dataclass
class FormalContext:
    """Triplet (objects, attributes, incidence) for FCA.

    ``incidence`` is wrapped read-only after construction. ``objects_of`` caches
    an inverted index against the mapping's identity, which handles the mapping
    being *replaced* but could serve a stale index if it were edited in place;
    making in-place edits raise removes that case rather than documenting it.
    (Freezing the dataclass would not: it forbids rebinding the attribute, which
    the cache already handles, and still permits mutating the dict behind it.)
    """

    objects: list[str]
    attributes: list[Attribute]
    # incidence[obj] -> set of attribute indices the object has
    incidence: Mapping[str, frozenset[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.incidence = MappingProxyType(dict(self.incidence))

    def __setattr__(self, name: str, value: object) -> None:
        if name == "incidence" and not isinstance(value, MappingProxyType):
            value = MappingProxyType(dict(value))  # type: ignore[arg-type]
        super().__setattr__(name, value)

    # --- factories ----------------------------------------------------
    @classmethod
    def from_graph(
        cls, graph: WorldGraph, *, min_support: int = 1, max_attributes: int | None = None
    ) -> FormalContext:
        """Build a formal context (entity × (relation, tail)) from a WorldGraph.

        ``min_support`` filters out attributes shared by fewer than that
        many entities — without it the lattice for any reasonable KG is
        unmanageably large.  ``max_attributes`` truncates the attribute
        list to the most-frequent N attributes if set (lattice size is
        exponential in the column count, so this is a hard cap when
        scaling).
        """
        # Objects of the formal context are *subjects*: only entities
        # that appear on the left of at least one edge can be described
        # by attributes ``(relation, tail)``.  Target-only nodes
        # (companies / languages / locations referenced but not
        # described) are attribute *values*, not objects in the FCA
        # sense.
        ents = sorted({e.source for e in graph.edges})
        # collect (relation, tail) frequencies
        counts: dict[Attribute, int] = {}
        memberships: dict[str, set[Attribute]] = {e: set() for e in ents}
        for edge in graph.edges:
            attr = (edge.relation, edge.target)
            counts[attr] = counts.get(attr, 0) + 1
            memberships.setdefault(edge.source, set()).add(attr)
        kept = sorted(
            [a for a, c in counts.items() if c >= min_support], key=lambda a: (-counts[a], a)
        )
        if max_attributes is not None:
            kept = kept[:max_attributes]
        kept_idx = {a: i for i, a in enumerate(kept)}
        incidence = {
            e: frozenset(kept_idx[a] for a in memberships[e] if a in kept_idx) for e in ents
        }
        return cls(objects=ents, attributes=kept, incidence=incidence)

    # --- Galois operators --------------------------------------------
    def _attr_extents(self) -> dict[int, frozenset[str]]:
        """Inverted index attribute -> objects, cached against ``incidence``.

        Caching on the identity of ``incidence`` rather than building once in
        ``__post_init__`` means a context whose incidence is replaced after
        construction cannot serve a stale index.
        """
        cached = getattr(self, "_extents_cache", None)
        if cached is not None and cached[0] is self.incidence:
            return cached[1]
        extents: dict[int, set[str]] = {}
        for g, idxs in self.incidence.items():
            for i in idxs:
                extents.setdefault(i, set()).add(g)
        index = {i: frozenset(s) for i, s in extents.items()}
        object.__setattr__(self, "_extents_cache", (self.incidence, index))
        return index

    def attrs_of(self, extent: frozenset[str]) -> frozenset[int]:
        """A' = intersection over g ∈ A of g's attribute set."""
        if not extent:
            return frozenset(range(len(self.attributes)))
        it = iter(extent)
        first = self.incidence.get(next(it), frozenset())
        common = set(first)
        for g in it:
            common &= self.incidence.get(g, frozenset())
            if not common:
                break
        return frozenset(common)

    def objects_of(self, intent: frozenset[int]) -> frozenset[str]:
        """B' = {g ∈ G | B ⊆ g's attribute set}.

        Intersects the per-attribute extents rather than testing every object,
        which is O(|B| x |extent|) instead of O(|G| x |B|). ``_extents`` is
        derived from ``incidence``, so it is built on demand and dropped
        whenever ``incidence`` is reassigned — see ``_attr_extents``.
        """
        if not intent:
            # B = {} holds of every object that HAS an attribute row, which is
            # what the object-wise formulation returned; `objects` may list
            # entities that were filtered out of `incidence`.
            return frozenset(self.incidence)
        extents = self._attr_extents()
        it = iter(intent)
        common = set(extents.get(next(it), frozenset()))
        for attr in it:
            common &= extents.get(attr, frozenset())
            if not common:
                break
        return frozenset(common)

    def closure_of_extent(self, extent: frozenset[str]) -> ExtentIntent:
        intent = self.attrs_of(extent)
        return self.objects_of(intent), intent

    def closure_of_intent(self, intent: frozenset[int]) -> ExtentIntent:
        ext = self.objects_of(intent)
        return ext, self.attrs_of(ext)

    # --- concept enumeration -----------------------------------------
    def concepts(self) -> list[ExtentIntent]:
        """Enumerate all formal concepts (NextClosure, Ganter 1984).

        Returns list of (extent, intent) pairs in lectic order on the
        intent.  Complexity is polynomial per concept; the total count
        is bounded above by 2^|M| so this is intended for small-to-
        medium contexts (≤ a few hundred objects × a few hundred
        attributes after ``min_support`` filtering).
        """
        n_attr = len(self.attributes)
        # All attribute indices sorted ascending; lectic order operates
        # on integer subsets via this order.
        bottom_int = self.attrs_of(frozenset(self.objects))
        # Start from the bottom concept (extent=all objects, intent=A').
        concepts: list[ExtentIntent] = [(frozenset(self.objects), bottom_int)]
        current = set(bottom_int)
        while True:
            nxt = self._next_intent(current, n_attr)
            if nxt is None:
                break
            ext = self.objects_of(frozenset(nxt))
            concepts.append((ext, frozenset(nxt)))
            current = nxt
        return concepts

    def _next_intent(self, B: set[int], n_attr: int) -> set[int] | None:
        """Lectically-next closed intent after ``B`` (Ganter's NextClosure)."""
        for m in range(n_attr - 1, -1, -1):
            if m in B:
                continue
            candidate = (B - {k for k in B if k > m}) | {m}
            closure = self.attrs_of(self.objects_of(frozenset(candidate)))
            # The lectic-next-closure step requires the closure to
            # introduce no attribute smaller than ``m`` that was not
            # already in B.
            if all(k >= m or k in B for k in (closure - candidate)):
                return set(closure)
        return None

    # --- lattice cover (Hasse diagram) ------------------------------
    def lattice_edges(self) -> list[tuple[ExtentIntent, ExtentIntent]]:
        """Direct covers (parent, child) in the concept lattice."""
        concepts = self.concepts()
        edges: list[tuple[ExtentIntent, ExtentIntent]] = []
        # parent c1 covers c2 iff extent(c2) ⊂ extent(c1) and no concept
        # sits strictly between them.
        by_size = sorted(concepts, key=lambda c: len(c[0]))
        for i, child in enumerate(by_size):
            for j in range(i + 1, len(by_size)):
                parent = by_size[j]
                if not child[0].issubset(parent[0]) or parent[0] == child[0]:
                    continue
                # Direct cover iff no intermediate concept exists.
                cover = True
                for k in range(i + 1, j):
                    mid = by_size[k]
                    if child[0] < mid[0] < parent[0]:
                        cover = False
                        break
                if cover:
                    edges.append((parent, child))
        return edges


def summarise_lattice(ctx: FormalContext) -> dict:
    """Summary statistics suitable for paper §11.x table."""
    cs = ctx.concepts()
    return {
        "n_objects": len(ctx.objects),
        "n_attributes": len(ctx.attributes),
        "n_concepts": len(cs),
        # non-trivial: excludes top (∅ intent) and bottom (∅ extent)
        "n_non_trivial": sum(
            1 for ext, int_ in cs if ext and int_ and ext != frozenset(ctx.objects)
        ),
    }


__all__ = ["FormalContext", "summarise_lattice", "Attribute", "ExtentIntent"]
