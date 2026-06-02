#!/usr/bin/env python3
"""TACET on a real knowledge graph — world geography.

Loads `tacet/datasets/worldgeo.json` (15 European countries: capitals, regions,
languages, currencies, borders), withholds a few `uses_currency` edges, and
answers a mixed query workload through the cascade:

* Tier 1 — transitive `located_in` (country -> subregion -> continent) and
  symmetric `borders`, resolved by deterministic rules with a proof;
* Tier 2 — a withheld `uses_currency` edge, inferred by the KGE from the
  currency homophily of neighbouring countries;
* Tier 3 — `has_capital`, an idiosyncratic fact only the teacher knows.

    python examples/real_kg_demo.py
"""

from __future__ import annotations

from pathlib import Path

from tacet import TACET, CascadeConfig, WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.kge.kge import KGEConfig
from tacet.llm.teacher import OracleTeacher

DATA = Path(__file__).resolve().parents[1] / "src" / "tacet" / "datasets" / "worldgeo.json"


def build_ontology() -> Ontology:
    onto = Ontology()
    for t in ("Country", "City", "Subregion", "Continent", "Language", "Currency"):
        onto.add_node_type(NodeType(t))
    onto.add_relation_type(
        RelationType(
            "located_in",
            frozenset({"Country", "Subregion"}),
            frozenset({"Subregion", "Continent"}),
            transitive=True,
        )
    )
    onto.add_relation_type(
        RelationType("borders", frozenset({"Country"}), frozenset({"Country"}), symmetric=True)
    )
    onto.add_relation_type(
        RelationType("has_capital", frozenset({"Country"}), frozenset({"City"}), functional=True)
    )
    onto.add_relation_type(
        RelationType(
            "official_language", frozenset({"Country"}), frozenset({"Language"}), functional=True
        )
    )
    onto.add_relation_type(
        RelationType(
            "uses_currency", frozenset({"Country"}), frozenset({"Currency"}), functional=True
        )
    )
    return onto


def main() -> None:
    full = WorldGraph.from_json(DATA)
    print(f"loaded {full.name}: {full.stats()}")

    # the teacher's oracle knows the *complete* graph.
    truth: dict[tuple[str, str], list[str]] = {}
    for e in full.edges:
        truth.setdefault((e.source, e.relation), []).append(e.target)

    def oracle(head: str, relation: str) -> list[str]:
        return list(truth.get((head, relation), []))

    # the observed graph withholds some currency edges and one capital.
    no_currency = {"Austria", "Belgium", "Portugal"}
    no_capital = {"Norway"}
    observed = WorldGraph(name=full.name)
    for n in full.nodes:
        observed.add_node(n.id, n.type, **n.props)
    for e in full.edges:
        if e.relation == "uses_currency" and e.source in no_currency:
            continue
        if e.relation == "has_capital" and e.source in no_capital:
            continue
        observed.add_edge(e.source, e.relation, e.target, **e.props)

    # expected answers account for transitive `located_in` and symmetric `borders`.
    def expected(head: str, relation: str) -> set[str]:
        if relation == "located_in":
            seen, stack = set(), list(truth.get((head, "located_in"), []))
            while stack:
                node = stack.pop()
                if node not in seen:
                    seen.add(node)
                    stack.extend(truth.get((node, "located_in"), []))
            return seen
        if relation == "borders":
            return {t for t in truth.get((head, "borders"), [])} | {
                s for (s, r), ts in truth.items() if r == "borders" and head in ts
            }
        return set(truth.get((head, relation), []))

    onto = build_ontology()
    teacher = OracleTeacher(oracle, entity_pool=full.entities())
    tacet = TACET(
        observed, onto, teacher, config=CascadeConfig(l2_threshold=0.4, kge=KGEConfig(epochs=300))
    )
    tacet.warmup()

    workload = [
        ("France", "located_in"),  # Tier 1 — transitive
        ("Switzerland", "located_in"),  # Tier 1 — transitive
        ("Belgium", "borders"),  # Tier 1 — symmetric closure
        ("Austria", "uses_currency"),  # Tier 2/3 — withheld edge
        ("Belgium", "uses_currency"),  # Tier 2/3 — withheld edge
        ("Portugal", "uses_currency"),  # Tier 2/3 — withheld edge
        ("Norway", "has_capital"),  # Tier 3 — idiosyncratic
    ]
    print()
    for head, relation in workload:
        ans = tacet.ask(head, relation)
        ok = set(ans.answers) == expected(head, relation)
        print(
            f"  T{ans.tier}  {head:12s} {relation:18s} -> "
            f"{', '.join(ans.answers) or '(none)':24s} "
            f"{'OK' if ok else 'MISS'}  ${ans.cost:.4f}"
        )
    print(f"\n  report: {tacet.report()}")


if __name__ == "__main__":
    main()
