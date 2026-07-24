"""Typed labeled-property graph — the world state the reasoning engine operates over.

`WorldGraph` is an in-memory directed multigraph keyed by string entity ids.
It uses a flexible property-graph schema (a node has a
`type`, an `id` and a free-form property dict) and adds the indices the
symbolic tier needs for O(degree) one-hop traversal in either direction.

Loaders accept the formats a real deployment encounters — raw triples, the
JSON export used by `tacet/datasets/`, and column CSV — so the framework runs on
arbitrary knowledge graphs, not a single hard-coded world.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

Triple = tuple[str, str, str]


def _edge_valid_at(props: dict, time: float) -> bool:
    """True if an edge with these props is valid at the given time."""
    vf = props.get("valid_from")
    vt = props.get("valid_to")
    if vf is None and vt is None:
        return True
    if vf is not None and time < vf:
        return False
    return not (vt is not None and time >= vt)


def _edge_overlaps(props: dict, start: float, end: float) -> bool:
    """True if the edge's validity interval overlaps [start, end)."""
    vf = props.get("valid_from")
    vt = props.get("valid_to")
    if vf is None and vt is None:
        return True
    edge_start = -float("inf") if vf is None else vf
    edge_end = float("inf") if vt is None else vt
    return edge_start < end and start < edge_end


@dataclass
class Node:
    """An entity. `id` is unique within a graph; `type` is its ontology class."""

    id: str
    type: str = "Entity"
    props: dict = field(default_factory=dict)


@dataclass
class Edge:
    """A directed, typed relation between two entities."""

    source: str
    relation: str
    target: str
    props: dict = field(default_factory=dict)

    @property
    def triple(self) -> Triple:
        return (self.source, self.relation, self.target)


class WorldGraph:
    """Directed multigraph with typed nodes and relations.

    Edges are indexed forward and backward by relation, so `out`/`into` are
    O(degree). Node ids are assumed unique (the standard KG assumption).
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._triple_to_edge: dict[Triple, Edge] = {}
        self._out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._in: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    # ------------------------------------------------------------- mutation
    def add_node(self, node_id: str, type: str = "Entity", **props: object) -> Node:
        """Upsert a node; re-adding merges properties (idempotent)."""
        node = self._nodes.get(node_id)
        if node is None:
            node = Node(id=node_id, type=type, props=dict(props))
            self._nodes[node_id] = node
        else:
            if type != "Entity":
                node.type = type
            node.props.update(props)
        return node

    def add_edge(self, source: str, relation: str, target: str, **props: object) -> Edge:
        """Upsert a directed edge; auto-creates missing endpoints."""
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)
        triple = (source, relation, target)
        if triple in self._triple_to_edge:
            edge = self._triple_to_edge[triple]
            edge.props.update(props)
            return edge
        edge = Edge(source, relation, target, dict(props))
        self._edges.append(edge)
        self._triple_to_edge[triple] = edge
        self._out[source][relation].add(target)
        self._in[target][relation].add(source)
        return edge

    # ------------------------------------------------------------- accessors
    def node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, source: str, relation: str, target: str) -> bool:
        return (source, relation, target) in self._triple_to_edge

    def slice_at(self, time: float) -> WorldGraph:
        """Return a graph containing only edges valid at `time`.

        An edge is considered valid at `t` when (a) it carries no temporal
        annotation, or (b) `valid_from <= t < valid_to` (half-open interval,
        with `None` endpoints meaning unbounded). Nodes are kept regardless.
        """
        sliced = WorldGraph(name=f"{self.name}@{time}")
        for n in self._nodes.values():
            sliced.add_node(n.id, n.type, **n.props)
        for e in self._edges:
            if _edge_valid_at(e.props, time):
                sliced.add_edge(e.source, e.relation, e.target, **e.props)
        return sliced

    def slice_between(self, start: float, end: float) -> WorldGraph:
        """Return a graph of edges whose validity overlaps the interval `[start, end)`."""
        sliced = WorldGraph(name=f"{self.name}@[{start},{end})")
        for n in self._nodes.values():
            sliced.add_node(n.id, n.type, **n.props)
        for e in self._edges:
            if _edge_overlaps(e.props, start, end):
                sliced.add_edge(e.source, e.relation, e.target, **e.props)
        return sliced

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    def entities(self) -> list[str]:
        return list(self._nodes)

    def relations(self) -> set[str]:
        return {e.relation for e in self._edges}

    def types(self) -> set[str]:
        return {n.type for n in self._nodes.values()}

    def nodes_of_type(self, type: str) -> list[str]:
        return [n.id for n in self._nodes.values() if n.type == type]

    def triples(self) -> list[Triple]:
        return [e.triple for e in self._edges]

    # ------------------------------------------------------------- traversal
    def out_relations(self, node_id: str) -> dict[str, set[str]]:
        """Forward adjacency of one node: relation -> neighbours.

        The mapping is the live index, so callers must treat it as read-only.
        Iterating it costs O(degree); `out()` per relation costs O(|R|).
        """
        return self._out.get(node_id, {})

    def in_relations(self, node_id: str) -> dict[str, set[str]]:
        """Backward adjacency of one node: relation -> neighbours (read-only)."""
        return self._in.get(node_id, {})

    def out(self, node_id: str, relation: str | None = None) -> set[str]:
        """Forward one-hop neighbours, optionally filtered by relation."""
        rels = self._out.get(node_id, {})
        if relation is not None:
            return set(rels.get(relation, ()))
        return {t for ts in rels.values() for t in ts}

    def into(self, node_id: str, relation: str | None = None) -> set[str]:
        """Backward one-hop neighbours, optionally filtered by relation."""
        rels = self._in.get(node_id, {})
        if relation is not None:
            return set(rels.get(relation, ()))
        return {s for ss in rels.values() for s in ss}

    def degree(self, node_id: str) -> int:
        return len(self.out(node_id)) + len(self.into(node_id))

    # ------------------------------------------------------------- stats
    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "types": len(self.types()),
            "relations": len(self.relations()),
        }

    # ------------------------------------------------------------- loaders
    @classmethod
    def from_triples(
        cls,
        triples: list[Triple],
        types: dict[str, str] | None = None,
        name: str = "graph",
    ) -> WorldGraph:
        """Build a graph from (head, relation, tail) triples."""
        g = cls(name=name)
        types = types or {}
        for h, r, t in triples:
            g.add_node(h, types.get(h, "Entity"))
            g.add_node(t, types.get(t, "Entity"))
            g.add_edge(h, r, t)
        return g

    @classmethod
    def from_json(cls, path: str | Path) -> WorldGraph:
        """Load the TACET JSON KG format (keys: name, nodes, edges)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = cls(name=data.get("name", "graph"))
        for n in data.get("nodes", []):
            g.add_node(n["id"], n.get("type", "Entity"), **n.get("props", {}))
        for e in data.get("edges", []):
            g.add_edge(e["source"], e["relation"], e["target"], **e.get("props", {}))
        return g

    @classmethod
    def from_csv(cls, path: str | Path, name: str = "graph") -> WorldGraph:
        """Load a 3-column CSV with header `source,relation,target`."""
        g = cls(name=name)
        with Path(path).open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                g.add_edge(row["source"], row["relation"], row["target"])
        return g

    def copy(self) -> WorldGraph:
        """A deep copy — systems that mutate the graph (write-back) work on their own."""
        g = WorldGraph(name=self.name)
        for n in self.nodes:
            g.add_node(n.id, n.type, **n.props)
        for e in self.edges:
            g.add_edge(e.source, e.relation, e.target, **e.props)
        return g

    def to_json(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "nodes": [{"id": n.id, "type": n.type, "props": n.props} for n in self.nodes],
            "edges": [
                {"source": e.source, "relation": e.relation, "target": e.target, "props": e.props}
                for e in self.edges
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
