"""Compliance benchmark assembly: PrivaCI cases as a typed world graph.

Each case becomes a ``case`` node with one edge per normalised slot value;
category values are shared nodes, so structurally similar cases share
neighbourhoods — the substrate the rule miner generalises over. Gold labels
NEVER enter the graph: they live only in the returned oracle, so no arm can
leak them.

The ``violates`` relation (case -> article) is declared in the ontology but
not populated here — the cascade writes those edges back from teacher
verdicts at run time, gated by ``Ontology.allows``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.data.privaci import PrivaCICase
from tacet.data.privaci_vocab import normalize_case

#: slot -> (relation name, target node type)
SLOT_RELATIONS: dict[str, tuple[str, str]] = {
    "information_type": ("information_type", "info_type"),
    "purpose": ("purpose", "purpose"),
    "sender_role": ("sender_role", "role"),
    "recipient_role": ("recipient_role", "role"),
    "subject_role": ("subject_role", "role"),
    "consent_form": ("consent_form", "consent"),
}


@dataclass(frozen=True)
class ComplianceBenchmark:
    graph: WorldGraph
    ontology: Ontology
    workload: tuple[str, ...]  # case ids in dataset order
    oracle: dict[str, tuple[str, tuple[str, ...]]]  # case_id -> (verdict, articles)
    case_content: dict[str, str]  # case_id -> raw scenario text (teacher input)


def build_compliance_ontology() -> Ontology:
    onto = Ontology()
    for nt in ("case", "info_type", "purpose", "role", "consent", "article", "verdict"):
        onto.add_node_type(NodeType(name=nt))
    for slot, (rel, target_type) in SLOT_RELATIONS.items():
        onto.add_relation_type(
            RelationType(
                name=rel,
                domain=frozenset({"case"}),
                range=frozenset({target_type}),
                functional=(slot == "consent_form"),
            )
        )
    onto.add_relation_type(
        RelationType(name="violates", domain=frozenset({"case"}), range=frozenset({"article"}))
    )
    onto.add_relation_type(
        RelationType(name="verdict", domain=frozenset({"case"}), range=frozenset({"verdict"}))
    )
    return onto


def build_compliance_benchmark(
    cases: list[PrivaCICase], vocab: dict | None = None
) -> ComplianceBenchmark:
    onto = build_compliance_ontology()
    graph = WorldGraph()
    workload: list[str] = []
    oracle: dict[str, tuple[str, tuple[str, ...]]] = {}
    content: dict[str, str] = {}

    for case in cases:
        slots = normalize_case(case, vocab)
        graph.add_node(case.case_id, type="case")
        for slot, values in slots.items():
            rel, target_type = SLOT_RELATIONS[slot]
            for value in values:
                if value == "none" and slot != "consent_form":
                    continue
                value_id = f"{target_type}:{value}"
                graph.add_node(value_id, type=target_type)
                graph.add_edge(case.case_id, rel, value_id)
        workload.append(case.case_id)
        oracle[case.case_id] = (case.norm_type, case.violated_articles)
        content[case.case_id] = case.case_content

    return ComplianceBenchmark(
        graph=graph,
        ontology=onto,
        workload=tuple(workload),
        oracle=oracle,
        case_content=content,
    )


__all__ = [
    "SLOT_RELATIONS",
    "ComplianceBenchmark",
    "build_compliance_benchmark",
    "build_compliance_ontology",
]
