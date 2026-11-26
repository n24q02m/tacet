"""Tests for Tier-4 3-arm SFT dataset export and leakage guarantees."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule, RuleEngine
from tacet.distill.export import (
    EntityLeakageError,
    ExportManifest,
    ExportRecord,
    _check_record_leakage,
    export_tier4_dataset,
)


def _build_family_world() -> tuple[
    WorldGraph, Ontology, list[Rule], dict[tuple[str, str], list[str]]
]:
    """Small deterministic world with two disconnected families (train vs eval)."""
    onto = Ontology()
    onto.add_node_type(NodeType("Person"))
    onto.add_node_type(NodeType("City"))
    p_set = frozenset({"Person"})
    onto.add_relation_type(RelationType("parent_of", p_set, p_set))
    onto.add_relation_type(RelationType("ancestor_of", p_set, p_set))
    onto.add_relation_type(RelationType("lives_in", p_set, frozenset({"City"}), functional=True))

    rules = [
        Rule("given:ancestor_base", (("?x", "parent_of", "?y"),), ("?x", "ancestor_of", "?y")),
        Rule(
            "given:ancestor_step",
            (("?x", "ancestor_of", "?z"), ("?z", "parent_of", "?y")),
            ("?x", "ancestor_of", "?y"),
        ),
    ]

    g = WorldGraph(name="family-export")
    for person in ("alice", "bob", "carol", "dave", "eve", "frank"):
        g.add_node(person, "Person")
    g.add_node("city_a", "City")
    g.add_node("city_b", "City")

    # Train component: alice -> bob -> carol, dave (leaf)
    g.add_edge("alice", "parent_of", "bob")
    g.add_edge("bob", "parent_of", "carol")
    g.add_edge("alice", "lives_in", "city_a")
    g.add_edge("dave", "lives_in", "city_a")

    # Eval component: eve -> frank
    g.add_edge("eve", "parent_of", "frank")
    g.add_edge("eve", "lives_in", "city_b")

    truth = {
        ("alice", "ancestor_of"): ["bob", "carol"],
        ("bob", "ancestor_of"): ["carol"],
        ("alice", "lives_in"): ["city_a"],
        ("dave", "lives_in"): ["city_a"],
        ("eve", "ancestor_of"): ["frank"],
        ("eve", "lives_in"): ["city_b"],
    }
    return g, onto, rules, truth


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class TestExportArmsAndDeterminism:
    def test_arm_a_answer_only(self, tmp_path: Path) -> None:
        g, onto, rules, truth = _build_family_world()
        train_entities = {"alice", "bob", "carol", "dave"}
        eval_entities = {"eve", "frank"}

        train_g = WorldGraph(name="train")
        for n in g.nodes:
            if n.id not in eval_entities:
                train_g.add_node(n.id, n.type, **n.props)
        for e in g.edges:
            if e.source not in eval_entities and e.target not in eval_entities:
                train_g.add_edge(e.source, e.relation, e.target, **e.props)

        engine = RuleEngine(onto, rules)
        engine.materialise(train_g)

        manifest = export_tier4_dataset(
            train_g,
            engine,
            truth,
            train_entities,
            eval_entities,
            tmp_path,
            arm="A",
            seed=7,
        )
        assert isinstance(manifest, ExportManifest)
        assert manifest.arm == "A"
        assert manifest.seed == 7
        assert manifest.positive_count == 4
        assert manifest.negative_count == 0
        assert manifest.record_count == 4
        assert manifest.train_entities_count == 4
        assert manifest.eval_entities_count == 2
        assert manifest.leakage_checked is True

        jsonl_path = Path(manifest.jsonl_path)
        manifest_path = Path(manifest.manifest_path)
        assert jsonl_path.name == "tier4_arm_A_s7.jsonl"
        assert manifest_path.name == "manifest_arm_A_s7.json"
        assert hashlib.sha256(jsonl_path.read_bytes()).hexdigest() == manifest.sha256

        rows = _read_jsonl(jsonl_path)
        assert len(rows) == 4
        for row in rows:
            assert row["arm"] == "A"
            assert row["proof"] == []
            assert row["abstain"] is False
            assert row["grounded"] is True
            assert row["head"] in train_entities
        alice_anc = next(r for r in rows if r["head"] == "alice" and r["relation"] == "ancestor_of")
        assert alice_anc["answers"] == ["bob", "carol"]

    def test_arm_b_proof_guided(self, tmp_path: Path) -> None:
        g, onto, rules, truth = _build_family_world()
        train_entities = {"alice", "bob", "carol", "dave"}
        eval_entities = {"eve", "frank"}

        train_g = WorldGraph(name="train")
        for n in g.nodes:
            if n.id not in eval_entities:
                train_g.add_node(n.id, n.type, **n.props)
        for e in g.edges:
            if e.source not in eval_entities and e.target not in eval_entities:
                train_g.add_edge(e.source, e.relation, e.target, **e.props)

        engine = RuleEngine(onto, rules)
        engine.materialise(train_g)

        manifest = export_tier4_dataset(
            train_g,
            engine,
            truth,
            train_entities,
            eval_entities,
            tmp_path,
            arm="B",
            seed=3,
        )
        assert manifest.arm == "B"
        assert manifest.negative_count == 0
        rows = _read_jsonl(Path(manifest.jsonl_path))
        alice_anc = next(r for r in rows if r["head"] == "alice" and r["relation"] == "ancestor_of")
        assert len(alice_anc["proof"]) >= 3
        assert any("given:ancestor_step" in ln for ln in alice_anc["proof"])
        assert any("FACT" in ln for ln in alice_anc["proof"])

    def test_arm_c_adds_deterministic_negatives(self, tmp_path: Path) -> None:
        g, onto, rules, truth = _build_family_world()
        train_entities = {"alice", "bob", "carol", "dave"}
        eval_entities = {"eve", "frank"}

        train_g = WorldGraph(name="train")
        for n in g.nodes:
            if n.id not in eval_entities:
                train_g.add_node(n.id, n.type, **n.props)
        for e in g.edges:
            if e.source not in eval_entities and e.target not in eval_entities:
                train_g.add_edge(e.source, e.relation, e.target, **e.props)

        engine = RuleEngine(onto, rules)
        engine.materialise(train_g)

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        m1 = export_tier4_dataset(
            train_g,
            engine,
            truth,
            train_entities,
            eval_entities,
            dir1,
            arm="C",
            seed=42,
            negative_ratio=0.5,
        )
        m2 = export_tier4_dataset(
            train_g,
            engine,
            truth,
            train_entities,
            eval_entities,
            dir2,
            arm="C",
            seed=42,
            negative_ratio=0.5,
        )
        assert m1.sha256 == m2.sha256
        assert m1.positive_count == 4
        assert m1.negative_count == 2
        assert m1.record_count == 6

        rows = _read_jsonl(Path(m1.jsonl_path))
        neg_rows = [r for r in rows if r["abstain"]]
        assert len(neg_rows) == 2
        for nr in neg_rows:
            assert nr["answers"] == []
            assert nr["proof"] == []
            assert nr["grounded"] is False
            assert nr["head"] in train_entities
            assert not engine.query(nr["head"], nr["relation"]).answered

        # Zero negative ratio produces zero negatives
        m_zero = export_tier4_dataset(
            train_g,
            engine,
            truth,
            train_entities,
            eval_entities,
            tmp_path / "zero",
            arm="C",
            seed=42,
            negative_ratio=0.0,
        )
        assert m_zero.negative_count == 0
        assert m_zero.record_count == m_zero.positive_count


class TestLeakageAndValidation:
    def test_overlap_between_train_and_eval_raises(self, tmp_path: Path) -> None:
        g, onto, rules, truth = _build_family_world()
        engine = RuleEngine(onto, rules)
        engine.materialise(g)
        with pytest.raises(EntityLeakageError, match="overlap"):
            export_tier4_dataset(
                g,
                engine,
                truth,
                {"alice", "bob"},
                {"bob", "eve"},
                tmp_path,
                arm="A",
            )

    def test_eval_entity_in_answers_raises(self, tmp_path: Path) -> None:
        g, onto, rules, truth = _build_family_world()
        # Cross-link train entity bob -> eval entity eve
        g.add_edge("bob", "parent_of", "eve")
        truth[("bob", "ancestor_of")] = ["carol", "eve"]
        engine = RuleEngine(onto, rules)
        engine.materialise(g)

        with pytest.raises(EntityLeakageError):
            export_tier4_dataset(
                g,
                engine,
                truth,
                {"alice", "bob", "carol"},
                {"eve", "frank"},
                tmp_path,
                arm="A",
            )

    def test_eval_entity_in_intermediate_proof_premise_raises(self, tmp_path: Path) -> None:
        """Even if head and answer are train entities, an eval premise must fail closed."""
        g = WorldGraph(name="bridge")
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        p_set = frozenset({"Person"})
        onto.add_relation_type(RelationType("parent_of", p_set, p_set))
        onto.add_relation_type(RelationType("ancestor_of", p_set, p_set))
        rules = [
            Rule("base", (("?x", "parent_of", "?y"),), ("?x", "ancestor_of", "?y")),
            Rule(
                "step",
                (("?x", "ancestor_of", "?z"), ("?z", "parent_of", "?y")),
                ("?x", "ancestor_of", "?y"),
            ),
        ]
        # train_a -> eval_mid -> train_b
        g.add_edge("train_a", "parent_of", "eval_mid")
        g.add_edge("eval_mid", "parent_of", "train_b")
        engine = RuleEngine(onto, rules)
        engine.materialise(g)
        truth = {("train_a", "ancestor_of"): ["train_b"]}

        for arm in ("A", "B"):
            with pytest.raises(EntityLeakageError):
                export_tier4_dataset(
                    g,
                    engine,
                    truth,
                    {"train_a", "train_b"},
                    {"eval_mid"},
                    tmp_path / arm,
                    arm=arm,
                )

    def test_prefix_entity_id_does_not_false_positive_leak(self, tmp_path: Path) -> None:
        """Entity 'person_1' in eval must not false-trigger on 'person_10' in train."""
        g = WorldGraph(name="prefix-safe")
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        p_set = frozenset({"Person"})
        onto.add_relation_type(RelationType("parent_of", p_set, p_set))
        g.add_node("person_10", "Person")
        g.add_node("person_11", "Person")
        g.add_edge("person_10", "parent_of", "person_11")
        engine = RuleEngine(onto)
        engine.materialise(g)
        truth = {("person_10", "parent_of"): ["person_11"]}

        manifest = export_tier4_dataset(
            g,
            engine,
            truth,
            {"person_10", "person_11"},
            {"person_1"},
            tmp_path,
            arm="B",
        )
        assert manifest.record_count == 1

    def test_invalid_arm_or_negative_ratio_raises(self, tmp_path: Path) -> None:
        g, onto, rules, truth = _build_family_world()
        engine = RuleEngine(onto, rules)
        engine.materialise(g)
        with pytest.raises(ValueError, match="arm"):
            export_tier4_dataset(
                g,
                engine,
                truth,
                {"alice"},
                {"eve"},
                tmp_path,
                arm="Z",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="negative_ratio"):
            export_tier4_dataset(
                g,
                engine,
                truth,
                {"alice"},
                {"eve"},
                tmp_path,
                arm="C",
                negative_ratio=-0.1,
            )


class TestPhase2ForwardCompatibilityAndFallbacks:
    def test_supports_phase2_duck_typed_engine_methods(self, tmp_path: Path) -> None:
        """Simulate Phase 2 RuleEngine having .derivations(grounded_only=True) and .entails()."""

        @dataclass(frozen=True)
        class FakeEntail:
            entailed: bool

        @dataclass(frozen=True)
        class FakeDeriv:
            triple: tuple[str, str, str]
            rule: str
            premises: tuple[tuple[str, str, str], ...]

        class Phase2Engine:
            def __init__(self) -> None:
                self.ontology = Ontology()
                self.rules = [Rule("r1", (("?x", "p", "?y"),), ("?x", "q", "?y"))]
                self._closure = {("a", "p", "b"), ("a", "q", "b")}
                self._deriv = {
                    ("a", "q", "b"): FakeDeriv(("a", "q", "b"), "r1", (("a", "p", "b"),))
                }

            @property
            def closure(self) -> set[tuple[str, str, str]]:
                return set(self._closure)

            def entails(self, h: str, r: str, t: str) -> FakeEntail:
                return FakeEntail((h, r, t) in self._closure)

            def derivations(
                self, *, grounded_only: bool = True
            ) -> dict[tuple[str, str, str], FakeDeriv]:
                return dict(self._deriv)

            def explain(self, triple: tuple[str, str, str]) -> list[str]:
                if triple == ("a", "q", "b"):
                    return ["DERIVED  a -q-> b   [r1]", "  FACT     a -p-> b"]
                return [f"FACT     {triple[0]} -{triple[1]}-> {triple[2]}"]

        g = WorldGraph.from_triples([("a", "p", "b")])
        eng = Phase2Engine()
        truth = {("a", "q"): ["b"]}
        manifest = export_tier4_dataset(
            g,
            eng,
            truth,
            {"a", "b"},
            {"z"},
            tmp_path,
            arm="B",
            seed=1,
        )
        assert manifest.record_count == 1
        rows = _read_jsonl(Path(manifest.jsonl_path))
        assert rows[0]["answers"] == ["b"]
        assert rows[0]["grounded"] is True
        assert any("DERIVED" in ln for ln in rows[0]["proof"])

    def test_empty_truth_falls_back_to_engine_closure_and_export_record_fields(
        self, tmp_path: Path
    ) -> None:
        g = WorldGraph.from_triples([("a", "rel", "b")])
        engine = RuleEngine(Ontology())
        engine.materialise(g)
        manifest = export_tier4_dataset(
            g,
            engine,
            {},
            {"a", "b"},
            {"x"},
            tmp_path,
            arm="A",
        )
        assert manifest.record_count == 1
        rec = ExportRecord(head="a", relation="rel", answers=("b",), proof=(), arm="A")
        assert rec.abstain is False
        assert rec.grounded is False

    def test_ungrounded_empty_truth_and_typed_domain_filtering(self, tmp_path: Path) -> None:
        g = WorldGraph(name="typed-arm-c")
        onto = Ontology()
        onto.add_node_type(NodeType("Person"))
        onto.add_node_type(NodeType("Company"))
        onto.add_node_type(NodeType("City"))
        onto.add_relation_type(
            RelationType("works_at", frozenset({"Person"}), frozenset({"Company"}))
        )
        onto.add_relation_type(
            RelationType("located_in", frozenset({"Company"}), frozenset({"City"}))
        )
        onto.add_relation_type(RelationType("tag", frozenset({"*"}), frozenset({"*"})))
        g.add_node("p1", "Person")
        g.add_node("p2", "Person")
        g.add_node("c1", "Company")
        g.add_node("city1", "City")
        g.add_edge("p1", "works_at", "c1")
        g.add_edge("c1", "located_in", "city1")
        g.add_edge("p1", "tag", "t1")

        engine = RuleEngine(onto)
        engine.materialise(g)
        # Include empty truth list, ungrounded truth answer, and train entities with an untyped node
        truth = {
            ("p1", "works_at"): ["c1", "missing_company"],
            ("p2", "works_at"): [],
            ("p1", "tag"): ["t1"],
        }
        m = export_tier4_dataset(
            g,
            engine,
            truth,
            {"p1", "p2", "untyped_node"},
            {"eval_z"},
            tmp_path,
            arm="C",
            seed=9,
            negative_ratio=1.0,
        )
        rows = _read_jsonl(Path(m.jsonl_path))
        neg_rels = {(r["head"], r["relation"]) for r in rows if r["abstain"]}
        assert ("p1", "located_in") not in neg_rels
        assert ("p2", "located_in") not in neg_rels

    def test_duck_typed_list_derivations_query_fallback_and_leakage_guards(
        self, tmp_path: Path
    ) -> None:
        @dataclass(frozen=True)
        class FakeDeriv:
            triple: tuple[str, str, str]
            rule: str
            premises: tuple[tuple[str, str, str], ...]

        @dataclass
        class FakeQueryRes:
            answered: bool
            proof: list[str]

        class MinimalDuckEngine:
            def __init__(self) -> None:
                self.ontology = Ontology()
                self.rules = [Rule("r1", (("?x", "p", "?y"),), ("?x", "q", "?y"))]
                self.closure = {("a", "p", "b"), ("a", "q", "b"), ("a", "q", "bad")}
                self._derivs = [FakeDeriv(("a", "q", "b"), "r1", (("a", "p", "b"),))]

            def entails(self, h: str, r: str, t: str) -> bool:
                return (h, r, t) in self.closure

            def derivations(self, *, grounded_only: bool = True) -> list[FakeDeriv]:
                if grounded_only:
                    return list(self._derivs)
                return list(self._derivs) + [FakeDeriv(("a", "q", "bad"), "unknown", ())]

            def query(self, h: str, r: str) -> FakeQueryRes:
                if (h, r) == ("a", "q"):
                    return FakeQueryRes(True, ["DERIVED  a -q-> b   [r1]", "  FACT     a -p-> b"])
                return FakeQueryRes(False, [])

        g = WorldGraph.from_triples([("a", "p", "b")])
        eng = MinimalDuckEngine()
        # 'bad' is ungrounded (absent when grounded_only=True), 'a-p-b' is base fact
        m = export_tier4_dataset(
            g,
            eng,
            {("a", "q"): ["b", "bad"], ("a", "p"): ["b"]},
            {"a", "b"},
            {"eval_z"},
            tmp_path / "ok",
            arm="B",
            seed=0,
        )
        assert m.record_count == 2

        # Direct leakage in truth answers raises EntityLeakageError
        with pytest.raises(EntityLeakageError):
            export_tier4_dataset(
                g,
                eng,
                {("a", "q"): ["eval_z"]},
                {"a", "b"},
                {"eval_z"},
                tmp_path / "leak_truth",
                arm="A",
            )

        # No-query duck engine for Arm C _has_any_entailment fallback
        class NoQueryEngine:
            ontology = Ontology()
            rules: list[Rule] = []
            closure = {("a", "p", "b")}
            _derivations: dict[tuple[str, str, str], FakeDeriv] = {}

        m_c = export_tier4_dataset(
            g,
            NoQueryEngine(),
            {("a", "p"): ["b"]},
            {"a", "b"},
            {"eval_z"},
            tmp_path / "no_query",
            arm="C",
            seed=0,
            negative_ratio=1.0,
        )
        assert m_c.negative_count >= 1

        # Record leakage guard catches head, answer, and non-standard proof token leaks
        with pytest.raises(EntityLeakageError):
            _check_record_leakage(ExportRecord("eval_z", "p", ("b",), (), "A"), {"eval_z"})
        with pytest.raises(EntityLeakageError):
            _check_record_leakage(ExportRecord("a", "p", ("eval_z",), (), "A"), {"eval_z"})
        with pytest.raises(EntityLeakageError):
            _check_record_leakage(
                ExportRecord("a", "p", ("b",), ("plain text mentioning eval_z",), "B"),
                {"eval_z"},
            )
        with pytest.raises(EntityLeakageError):
            _check_record_leakage(
                ExportRecord("a", "p", ("b",), ("SINGLE_TOKEN_eval_z", "eval_z"), "B"),
                {"eval_z"},
            )
