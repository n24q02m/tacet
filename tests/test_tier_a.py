"""Tests for the Tier A extensions: temporal, ingestion, and datasets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tacet.core.graph import WorldGraph
from tacet.core.ingest import (
    CallableExtractor,
    KGBuilder,
    Pattern,
    RuleBasedExtractor,
)
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.temporal import TemporalEngine, allen_relation, temporal_edge
from tacet.data import (
    KGDataset,
    load_kg_dataset,
    load_triples_tsv,
    load_worldgeo,
    split_triples,
    synthetic_kg_dataset,
    worldgeo_dataset,
)


# --- temporal --------------------------------------------------------------
class TestTemporalGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.g = WorldGraph()
        self.g.add_node("alice", "Person")
        for company in ("acme", "zoot", "ever"):
            self.g.add_node(company, "Company")
        self.g.add_edge("alice", "works_at", "acme", valid_from=2018.0, valid_to=2021.0)
        self.g.add_edge("alice", "works_at", "zoot", valid_from=2021.0, valid_to=2024.0)
        self.g.add_edge("alice", "works_at", "ever", valid_from=2024.0, valid_to=None)
        self.g.add_edge("alice", "knows", "bob")  # untimed -> always valid

    def test_slice_at_picks_current_edge(self) -> None:
        self.assertEqual(
            [
                (e.source, e.relation, e.target)
                for e in self.g.slice_at(2019).edges
                if e.relation == "works_at"
            ],
            [("alice", "works_at", "acme")],
        )

    def test_slice_at_handles_open_end(self) -> None:
        edges = {(e.target) for e in self.g.slice_at(2030).edges if e.relation == "works_at"}
        self.assertEqual(edges, {"ever"})

    def test_untimed_edges_always_valid(self) -> None:
        for year in (1900.0, 2025.0):
            self.assertTrue(self.g.slice_at(year).has_edge("alice", "knows", "bob"))

    def test_slice_between_collects_overlap(self) -> None:
        edges = {
            e.target for e in self.g.slice_between(2020, 2023).edges if e.relation == "works_at"
        }
        self.assertEqual(edges, {"acme", "zoot"})

    def test_temporal_edge_builder(self) -> None:
        self.assertEqual(
            temporal_edge("a", "r", "b", valid_from=1, valid_to=2), {"valid_from": 1, "valid_to": 2}
        )


class TestTemporalEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.g = WorldGraph()
        self.g.add_node("p", "Person")
        for c in ("a", "b"):
            self.g.add_node(c, "Company")
        self.g.add_edge("p", "works_at", "a", valid_from=2020.0, valid_to=2023.0)
        self.g.add_edge("p", "works_at", "b", valid_from=2023.0)
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        onto.add_node_type(NodeType("Company"))
        onto.add_relation_type(
            RelationType("works_at", frozenset({"Person"}), frozenset({"Company"}))
        )
        self.engine = TemporalEngine(onto, [])

    def test_query_at_returns_temporally_valid(self) -> None:
        self.assertEqual(self.engine.query_at(self.g, "p", "works_at", 2021).answers, ["a"])
        self.assertEqual(self.engine.query_at(self.g, "p", "works_at", 2025).answers, ["b"])

    def test_query_history_yields_trajectory(self) -> None:
        hist = self.engine.query_history(self.g, "p", "works_at", [2019, 2022, 2025])
        self.assertEqual(hist, {2019: [], 2022: ["a"], 2025: ["b"]})

    def test_query_during_unions_interval(self) -> None:
        self.assertEqual(
            set(self.engine.query_during(self.g, "p", "works_at", 2020, 2026).answers), {"a", "b"}
        )


class TestAllen(unittest.TestCase):
    def test_overlaps_and_during(self) -> None:
        self.assertEqual(
            allen_relation({"valid_from": 1, "valid_to": 3}, {"valid_from": 2, "valid_to": 4}),
            "overlaps",
        )
        self.assertEqual(
            allen_relation({"valid_from": 2, "valid_to": 3}, {"valid_from": 1, "valid_to": 4}),
            "during",
        )

    def test_before_after_and_meets(self) -> None:
        self.assertEqual(
            allen_relation({"valid_from": 1, "valid_to": 2}, {"valid_from": 3, "valid_to": 4}),
            "before",
        )
        self.assertEqual(
            allen_relation({"valid_from": 5, "valid_to": 6}, {"valid_from": 3, "valid_to": 4}),
            "after",
        )
        self.assertEqual(
            allen_relation({"valid_from": 1, "valid_to": 2}, {"valid_from": 2, "valid_to": 3}),
            "meets",
        )

    def test_equal_and_open_endpoints(self) -> None:
        self.assertEqual(
            allen_relation({"valid_from": 1, "valid_to": 2}, {"valid_from": 1, "valid_to": 2}),
            "equal",
        )
        self.assertEqual(allen_relation({}, {}), "equal")


# --- ingestion --------------------------------------------------------------
class TestIngest(unittest.TestCase):
    def test_pattern_extracts_named_groups(self) -> None:
        pat = Pattern(r"(?P<head>\w+) borders (?P<tail>\w+)", relation="borders")
        self.assertEqual(
            pat.matches("France borders Belgium and Germany borders Austria"),
            [("France", "Belgium"), ("Germany", "Austria")],
        )

    def test_rule_based_extractor_produces_triples(self) -> None:
        ext = RuleBasedExtractor(
            [
                Pattern(r"(?P<head>\w+) borders (?P<tail>\w+)", "borders"),
                Pattern(r"capital of (?P<tail>\w+) is (?P<head>\w+)", "has_capital"),
            ]
        )
        triples = ext.extract("France borders Belgium. The capital of France is Paris.")
        self.assertIn(("France", "borders", "Belgium"), triples)
        self.assertIn(("Paris", "has_capital", "France"), triples)

    def test_callable_extractor_wraps_function(self) -> None:
        ext = CallableExtractor(lambda _t: [("a", "rel", "b")])
        self.assertEqual(ext.extract("anything"), [("a", "rel", "b")])

    def test_kgbuilder_writes_to_graph(self) -> None:
        ext = RuleBasedExtractor(
            [
                Pattern(r"(?P<head>\w+) borders (?P<tail>\w+)", "borders"),
            ]
        )
        builder = KGBuilder(ext, type_hints={"borders": ("Country", "Country")})
        graph, report = builder.ingest(["France borders Belgium. France borders Germany."])
        self.assertEqual(report.edges_added, 2)
        self.assertTrue(graph.has_edge("France", "borders", "Germany"))
        self.assertEqual(graph.node("France").type, "Country")

    def test_kgbuilder_filters_by_ontology(self) -> None:
        onto = Ontology()
        onto.add_node_type(NodeType("Country"))
        onto.add_relation_type(
            RelationType("borders", frozenset({"Country"}), frozenset({"Country"}), symmetric=True)
        )
        ext = RuleBasedExtractor(
            [
                Pattern(r"(?P<head>\w+) borders (?P<tail>\w+)", "borders"),
                Pattern(r"(?P<head>\w+) loves (?P<tail>\w+)", "loves"),  # unknown relation
            ]
        )
        builder = KGBuilder(ext, ontology=onto, type_hints={"borders": ("Country", "Country")})
        _, report = builder.ingest(["A borders B. C loves D."])
        self.assertEqual(report.triples_rejected_unknown_relation, 1)
        self.assertEqual(report.edges_added, 1)


# --- datasets ---------------------------------------------------------------
class TestDatasets(unittest.TestCase):
    def test_load_worldgeo(self) -> None:
        g = load_worldgeo()
        self.assertIn("France", g.entities())
        self.assertGreater(g.stats()["edges"], 80)

    def test_load_triples_tsv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "triples.tsv"
            path.write_text("a\trel\tb\nc\trel\td\n", encoding="utf-8")
            self.assertEqual(load_triples_tsv(path), [("a", "rel", "b"), ("c", "rel", "d")])

    def test_load_kg_dataset_standard_layout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for part in ("train", "valid", "test"):
                (root / f"{part}.txt").write_text(f"{part}_h\trel\t{part}_t\n", encoding="utf-8")
            ds = load_kg_dataset(root, name="mini")
            self.assertEqual(ds.name, "mini")
            self.assertEqual(ds.stats()["train"], 1)
            self.assertEqual(ds.stats()["test"], 1)

    def test_split_triples_partitions_deterministically(self) -> None:
        triples = [(str(i), "r", str(i + 1)) for i in range(100)]
        a = split_triples(triples, (0.7, 0.2, 0.1), seed=42)
        b = split_triples(triples, (0.7, 0.2, 0.1), seed=42)
        self.assertEqual(a.train, b.train)
        total = len(a.train) + len(a.valid) + len(a.test)
        self.assertEqual(total, 100)
        seen = {*a.train, *a.valid, *a.test}
        self.assertEqual(len(seen), 100)

    def test_worldgeo_dataset_to_graph(self) -> None:
        ds = worldgeo_dataset(seed=0)
        g = ds.to_graph()
        self.assertGreater(g.stats()["nodes"], 30)

    def test_synthetic_dataset_is_reasonably_sized(self) -> None:
        ds = synthetic_kg_dataset(seed=0)
        stats = ds.stats()
        self.assertGreater(stats["train"], 1000)
        self.assertGreater(stats["entities"], 100)

    def test_kg_dataset_dataclass(self) -> None:
        ds = KGDataset("x", [("a", "r", "b")], [], [])
        self.assertEqual(ds.all_triples(), [("a", "r", "b")])


if __name__ == "__main__":
    unittest.main()
