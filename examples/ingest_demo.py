#!/usr/bin/env python3
"""Auto-ingestion — turn a short text corpus into a world graph.

    python examples/ingest_demo.py

Uses the deterministic regex extractor (reproducible offline). For a real
deployment, swap in `CallableExtractor(my_llm_fn)` — the rest of the
pipeline is unchanged.
"""

from __future__ import annotations

from tacet import TACET, KGBuilder, Pattern, RuleBasedExtractor
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.kge.kge import KGEConfig
from tacet.llm.teacher import OracleTeacher
from tacet.serve.config import CascadeConfig

corpus = [
    "France borders Belgium. France borders Germany. France borders Spain.",
    "Spain borders Portugal. Germany borders Austria. Austria borders Italy.",
    "The capital of France is Paris. The capital of Germany is Berlin.",
    "The capital of Spain is Madrid. The capital of Italy is Rome.",
]

ontology = Ontology()
ontology.add_node_type(NodeType("Country"))
ontology.add_node_type(NodeType("City"))
ontology.add_relation_type(
    RelationType("borders", frozenset({"Country"}), frozenset({"Country"}), symmetric=True)
)
ontology.add_relation_type(
    RelationType("has_capital", frozenset({"Country"}), frozenset({"City"}), functional=True)
)

extractor = RuleBasedExtractor(
    [
        Pattern(r"(?P<head>\w+) borders (?P<tail>\w+)", "borders", "Country", "Country"),
        Pattern(r"capital of (?P<head>\w+) is (?P<tail>\w+)", "has_capital", "Country", "City"),
    ]
)
builder = KGBuilder(
    extractor,
    ontology=ontology,
    type_hints={
        "borders": ("Country", "Country"),
        "has_capital": ("Country", "City"),
    },
)
graph, report = builder.ingest(corpus)
print(f"Ingestion: {report}")
print(f"  graph: {graph.stats()}")
print(f"  validation: {ontology.validate(graph) or 'OK'}")

# Drive the cascade on the freshly-built graph. The borders relation is
# declared symmetric in the ontology so Tier-1 derives the reverse edges
# automatically.
oracle = {("Spain", "borders"): ["France", "Portugal"], ("Italy", "borders"): ["Austria"]}
ak = TACET(
    graph,
    ontology,
    teacher=OracleTeacher(lambda h, r: oracle.get((h, r), [])),
    config=CascadeConfig(
        l2_threshold=1.01,  # turn KGE off for this micro-demo
        kge=KGEConfig(epochs=20),
    ),
)
ak.warmup()
for head, rel in [
    ("France", "borders"),
    ("Germany", "borders"),
    ("Germany", "has_capital"),
    ("Italy", "has_capital"),
]:
    ans = ak.ask(head, rel)
    print(f"  T{ans.tier}  {head:10s} {rel:14s} -> {ans.answers}")
