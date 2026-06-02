"""Typed ontology — node/relation types, structural axioms, and validation.

The ontology turns the C1 *flexible* schema into a C2 *typed* one. It serves
three roles in the reasoning engine:

1. integrity checking — `validate` flags type violations deterministically;
2. a verification gate — Tier 2 (KGE) predictions that violate domain/range
   or functionality are rejected before they are ever surfaced;
3. a source of structural axioms (symmetric / transitive / inverse) that the
   symbolic tier compiles into rules.

`Ontology.induce` learns a serviceable ontology straight from graph data, so
the framework runs on any knowledge graph without a hand-written schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph


@dataclass
class NodeType:
    name: str
    required_props: tuple[str, ...] = ()


@dataclass
class RelationType:
    """A typed relation with optional structural axioms.

    symmetric  : r(a,b) => r(b,a)
    transitive : r(a,b) & r(b,c) => r(a,c)
    functional : each head has at most one tail under r
    inverse_of : r(a,b) <=> inverse_of(b,a)
    """

    name: str
    domain: frozenset[str] = frozenset({"*"})
    range: frozenset[str] = frozenset({"*"})
    symmetric: bool = False
    transitive: bool = False
    functional: bool = False
    inverse_of: str | None = None

    def accepts(self, source_type: str, target_type: str) -> bool:
        ok_s = "*" in self.domain or source_type in self.domain
        ok_t = "*" in self.range or target_type in self.range
        return ok_s and ok_t


@dataclass
class Violation:
    kind: str
    detail: str


@dataclass
class Ontology:
    node_types: dict[str, NodeType] = field(default_factory=dict)
    relation_types: dict[str, RelationType] = field(default_factory=dict)

    # ------------------------------------------------------------- building
    def add_node_type(self, nt: NodeType) -> Ontology:
        self.node_types[nt.name] = nt
        return self

    def add_relation_type(self, rt: RelationType) -> Ontology:
        self.relation_types[rt.name] = rt
        return self

    def relation(self, name: str) -> RelationType | None:
        return self.relation_types.get(name)

    # ------------------------------------------------------------- gate
    def allows(self, graph: WorldGraph, source: str, relation: str, target: str) -> bool:
        """True iff adding (source, relation, target) keeps the graph well-typed.

        Used as the verification gate for Tier-2 predictions.
        """
        rt = self.relation_types.get(relation)
        if rt is None:
            return False
        s, t = graph.node(source), graph.node(target)
        if s is None or t is None:
            return False
        if not rt.accepts(s.type, t.type):
            return False
        if rt.functional:
            existing = graph.out(source, relation)
            if existing and target not in existing:
                return False
        return True

    # ------------------------------------------------------------- checking
    def validate(self, graph: WorldGraph) -> list[Violation]:
        """Return every type / structural violation in the current graph."""
        out: list[Violation] = []
        for node in graph.nodes:
            nt = self.node_types.get(node.type)
            if nt is None:
                continue
            for prop in nt.required_props:
                if prop not in node.props:
                    out.append(Violation("missing_property", f"{node.id} lacks '{prop}'"))
        for edge in graph.edges:
            rt = self.relation_types.get(edge.relation)
            if rt is None:
                out.append(Violation("unknown_relation", edge.relation))
                continue
            s, t = graph.node(edge.source), graph.node(edge.target)
            if s and t and not rt.accepts(s.type, t.type):
                out.append(
                    Violation(
                        "domain_range",
                        f"{edge.relation}: {s.type}->{t.type} not in "
                        f"{set(rt.domain)}->{set(rt.range)}",
                    )
                )
        for rt in self.relation_types.values():
            if not rt.functional:
                continue
            for node in graph.nodes:
                if len(graph.out(node.id, rt.name)) > 1:
                    out.append(Violation("functional", f"{node.id} has >1 {rt.name}"))
        return out

    # ------------------------------------------------------------- induction
    @classmethod
    def induce(cls, graph: WorldGraph, symmetric_thresh: float = 0.6) -> Ontology:
        """Induce a serviceable ontology directly from graph data.

        Learns domain/range from observed type usage, detects functional and
        symmetric relations statistically. Transitivity is *not* induced (it is
        unreliable to detect) and must be declared explicitly.
        """
        onto = cls()
        for t in graph.types():
            onto.add_node_type(NodeType(t))

        by_relation: dict[str, list[tuple[str, str]]] = {}
        for e in graph.edges:
            by_relation.setdefault(e.relation, []).append((e.source, e.target))

        for relation, pairs in by_relation.items():
            domain = {graph.node(s).type for s, _ in pairs if graph.node(s)}
            rng = {graph.node(t).type for _, t in pairs if graph.node(t)}
            functional = all(len(graph.out(s, relation)) <= 1 for s in {s for s, _ in pairs})
            present = set(pairs)
            mirrored = sum(1 for s, t in pairs if (t, s) in present)
            symmetric = len(pairs) > 0 and mirrored / len(pairs) >= symmetric_thresh
            onto.add_relation_type(
                RelationType(
                    name=relation,
                    domain=frozenset(domain or {"*"}),
                    range=frozenset(rng or {"*"}),
                    symmetric=symmetric,
                    functional=functional and not symmetric,
                )
            )
        return onto
