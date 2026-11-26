"""Tests for Tier-4 unseen benchmarks, dual colleague precision, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tacet.core.graph import WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule, RuleEngine
from tacet.distill.export import export_tier4_dataset
from tacet.eval.metrics import (
    colleague_precision_dual,
    entailment_rate,
    grounded_fraction,
    proof_coverage_tier4,
)
from tacet.eval.unseen import (
    UnseenBenchmark,
    UnseenBenchmarkConfig,
    generate_unseen_benchmark,
)


class TestUnseenBenchmark:
    def test_unseen_entity_zero_overlap_and_headroom(self) -> None:
        cfg = UnseenBenchmarkConfig(seed=0, holdout_entity_fraction=0.25)
        bench = generate_unseen_benchmark("unseen_entity", cfg)
        assert isinstance(bench, UnseenBenchmark)
        assert bench.variant == "unseen_entity"
        assert len(bench.eval_entities) > 0
        assert len(bench.train_entities) > 0
        assert bench.train_entities.isdisjoint(bench.eval_entities)

        # Strict zero entity overlap in train_graph
        train_nodes = set(bench.train_graph.entities())
        assert train_nodes.isdisjoint(bench.eval_entities)
        for e in bench.train_graph.edges:
            assert e.source not in bench.eval_entities
            assert e.target not in bench.eval_entities

        # Workload queries target eval_entities and have non-empty truth
        assert len(bench.workload) > 0
        for h, r in bench.workload:
            assert h in bench.eval_entities
            assert (h, r) in bench.truth
            assert len(bench.truth[(h, r)]) > 0

        assert bench.tier1_abstain_rate > 0.0
        assert bench.headroom > 0.0
        assert bench.noisy_teacher_accuracy is None

        # Deterministic across calls with same seed
        bench2 = generate_unseen_benchmark("unseen_entity", cfg)
        assert bench.eval_entities == bench2.eval_entities
        assert bench.workload == bench2.workload

    def test_new_relation_withholds_superior_of(self) -> None:
        cfg = UnseenBenchmarkConfig(seed=1)
        bench = generate_unseen_benchmark("new_relation", cfg)
        assert bench.variant == "new_relation"
        assert bench.train_entities.isdisjoint(bench.eval_entities)
        assert all(e.relation != "superior_of" for e in bench.train_graph.edges)
        assert all(r.head[1] != "superior_of" for r in bench.given_rules)
        assert len(bench.workload) > 0
        assert all(rel == "superior_of" for _, rel in bench.workload)
        assert bench.tier1_abstain_rate > 0.0
        assert bench.headroom > 0.0

    def test_noisy_teacher_records_accuracy_and_headroom(self) -> None:
        cfg = UnseenBenchmarkConfig(seed=2, noisy_teacher_error_rate=0.20)
        bench = generate_unseen_benchmark("noisy_teacher", cfg)
        assert bench.variant == "noisy_teacher"
        assert bench.noisy_teacher_accuracy is not None
        assert 0.0 < bench.noisy_teacher_accuracy < 1.0
        assert bench.tier1_abstain_rate > 0.0
        assert bench.headroom > 0.0
        assert len(bench.oracle) == len(bench.workload)

    def test_invalid_variant_or_config_raises(self) -> None:
        with pytest.raises(ValueError, match="variant"):
            generate_unseen_benchmark("bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            generate_unseen_benchmark(
                "unseen_entity",
                UnseenBenchmarkConfig(holdout_entity_fraction=1.0),
            )
        with pytest.raises(ValueError):
            generate_unseen_benchmark(
                "noisy_teacher",
                UnseenBenchmarkConfig(noisy_teacher_error_rate=1.5),
            )


class TestMetrics:
    def test_entailment_rate_supports_dicts_objects_and_phase2_engine(self) -> None:
        g = WorldGraph.from_triples([("a", "parent_of", "b"), ("b", "parent_of", "c")])
        onto = Ontology()
        p = frozenset({"Person"})
        onto.add_node_type(NodeType("Person"))
        onto.add_relation_type(RelationType("parent_of", p, p))
        onto.add_relation_type(RelationType("ancestor_of", p, p))
        rule = Rule("r", (("?x", "parent_of", "?y"),), ("?x", "ancestor_of", "?y"))
        engine = RuleEngine(onto, [rule])
        engine.materialise(g)

        @dataclass
        class ObjRec:
            head: str
            relation: str
            answers: tuple[str, ...]

        records = [
            {"head": "a", "relation": "ancestor_of", "answers": ["b"]},  # entailed
            ObjRec("a", "ancestor_of", ("c",)),  # not entailed (1-hop rule only)
            {"head": "b", "relation": "parent_of", "answers": []},  # empty ignored
            {"head": "b", "relation": "parent_of", "answer": "c"},  # single answer field (entailed)
            {"head": "b", "relation": "parent_of", "answers": "a"},  # string answers (not entailed)
            {"relation": "parent_of", "answers": ["c"]},  # missing head ignored
        ]
        assert entailment_rate([], engine) == 0.0
        assert entailment_rate(records, engine) == pytest.approx(0.5)

        # Duck-typed Phase 2 engine with .entails()
        @dataclass
        class EntailRes:
            entailed: bool

        class Phase2Engine:
            def entails(self, h: str, r: str, a: str) -> EntailRes:
                return EntailRes((h, r, a) in {("a", "ancestor_of", "b"), ("b", "parent_of", "c")})

        assert entailment_rate(records, Phase2Engine()) == pytest.approx(0.5)

    def test_proof_coverage_tier4_and_grounded_fraction(self) -> None:
        @dataclass
        class Tier4Rec:
            tier: int
            proof: list[str]
            grounded: bool

        assert proof_coverage_tier4([]) == 0.0
        assert grounded_fraction([]) == 0.0

        records = [
            Tier4Rec(tier=1, proof=["FACT a -r-> b"], grounded=True),
            Tier4Rec(tier=4, proof=["FACT c -r-> d"], grounded=True),
            {"tier": 4, "proof": [], "grounded": False},
            {"tier": 4, "proof": ("DERIVED e -r-> f",), "grounded": False},
            {"tier": 4, "proof": "FACT g -r-> h", "grounded": False},
            {"tier": 4, "proof": None, "grounded": False},
        ]
        # 5 Tier-4 records; 3 have non-empty proof -> 3/5
        assert proof_coverage_tier4(records) == pytest.approx(3 / 5)
        # 6 total records; 2 have grounded=True -> 2/6
        assert grounded_fraction(records) == pytest.approx(2 / 6)

        # Only Tier 1 records -> proof_coverage_tier4 returns 0.0
        assert proof_coverage_tier4([{"tier": 1, "proof": ["x"], "grounded": True}]) == 0.0

    def test_colleague_precision_dual_strict_vs_relaxed(self) -> None:
        g = WorldGraph(name="org")
        for p in ("p1", "p2", "p3"):
            g.add_node(p, "Person")
        g.add_node("c1", "Company")
        g.add_node("c2", "Company")
        g.add_node("d1", "Department")

        # p1 and p2 work at c1; p3 works at c2; all three are in d1
        g.add_edge("p1", "works_at", "c1")
        g.add_edge("p2", "works_at", "c1")
        g.add_edge("p3", "works_at", "c2")
        for p in ("p1", "p2", "p3"):
            g.add_edge(p, "member_of", "d1")

        truth = {
            ("p1", "colleague_of"): ["p2"],
            ("p2", "colleague_of"): ["p1"],
            ("p3", "colleague_of"): [],
            ("p1", "works_at"): ["c1"],
            ("p2", "works_at"): ["c1"],
            ("p3", "works_at"): ["c2"],
            ("p1", "member_of"): ["d1"],
            ("p2", "member_of"): ["d1"],
            ("p3", "member_of"): ["d1"],
        }

        # Rule without distinct guard using works_at:
        # predicts (p1,p2), (p2,p1), (p1,p1), (p2,p2), (p3,p3)
        rule_works = Rule(
            name="works_same_org",
            body=(("?x", "works_at", "?c"), ("?y", "works_at", "?c")),
            head=("?x", "colleague_of", "?y"),
        )
        res_works = colleague_precision_dual(rule_works, g, truth)
        assert res_works["strict_precision"] == pytest.approx(1.0)
        assert res_works["relaxed_precision"] == pytest.approx(1.0)
        assert res_works["predicted_pairs_strict"] == 2.0
        assert res_works["predicted_pairs_all"] == 5.0
        assert res_works["true_pairs"] == 2.0

        # Rule using member_of (same dept across companies): strict < relaxed
        rule_dept = Rule(
            name="same_dept",
            body=(("?x", "member_of", "?d"), ("?y", "member_of", "?d")),
            head=("?x", "colleague_of", "?y"),
            distinct=(("?x", "?y"),),
        )
        res_dept = colleague_precision_dual(rule_dept, g, truth)
        assert res_dept["strict_precision"] == pytest.approx(2 / 6)
        assert res_dept["relaxed_precision"] == pytest.approx(1.0)
        assert res_dept["predicted_pairs_strict"] == 6.0

        # WorldGraph truth + fallback to graph for works_at when absent from truth dict
        wg_truth = WorldGraph.from_triples(
            [("p1", "colleague_of", "p2"), ("p2", "colleague_of", "p1")]
        )
        res_wg = colleague_precision_dual(rule_works, g, wg_truth)
        assert res_wg["strict_precision"] == pytest.approx(1.0)
        assert res_wg["relaxed_precision"] == pytest.approx(1.0)


class TestEndToEndMockTier4Pipeline:
    def test_e2e_unseen_export_and_mock_tier4_metrics(self, tmp_path: Path) -> None:
        """Offline E2E check: unseen_entity split -> export Arm B -> mock Tier 4 eval."""
        bench = generate_unseen_benchmark(
            "unseen_entity",
            UnseenBenchmarkConfig(seed=5, holdout_entity_fraction=0.2),
        )
        engine = RuleEngine(bench.ontology, list(bench.given_rules))
        engine.materialise(bench.train_graph)

        manifest = export_tier4_dataset(
            bench.train_graph,
            engine,
            bench.truth,
            bench.train_entities,
            bench.eval_entities,
            tmp_path,
            arm="B",
            seed=5,
        )
        assert manifest.record_count > 0
        assert manifest.leakage_checked is True

        # Mock Tier 4 model answering the unseen workload using eval_graph oracle
        eval_engine = RuleEngine(bench.ontology, list(bench.given_rules))
        eval_engine.materialise(bench.eval_graph)

        predictions = []
        for h, r in bench.workload[:10]:
            sym = eval_engine.query(h, r)
            if sym.answered:
                predictions.append(
                    {
                        "head": h,
                        "relation": r,
                        "tier": 4,
                        "answers": sym.answers,
                        "proof": sym.proof,
                        "grounded": True,
                    }
                )
            else:
                predictions.append(
                    {
                        "head": h,
                        "relation": r,
                        "tier": 4,
                        "answers": bench.truth[(h, r)],
                        "proof": [],
                        "grounded": False,
                    }
                )
        assert len(predictions) == 10
        assert 0.0 <= entailment_rate(predictions, eval_engine) <= 1.0
        assert 0.0 <= proof_coverage_tier4(predictions) <= 1.0
        assert 0.0 <= grounded_fraction(predictions) <= 1.0

    def test_e2e_mock_lora_tier_with_entailment_gate_in_cascade(self) -> None:
        """Offline E2E check: TACET(tiers=[SymbolicTier, EntailmentGate(MockLoRATier), LLMTier])."""
        from tacet import (
            TACET,
            EntailmentGate,
            LLMTier,
            SymbolicTier,
            TierResult,
        )
        from tacet.core.ontology import NodeType, Ontology, RelationType
        from tacet.core.symbolic import Rule
        from tacet.llm.teacher import CallableTeacher

        onto = Ontology(
            node_types={"Person": NodeType("Person")},
            relation_types={
                "works_at": RelationType("works_at", "Person", "Person"),
                "colleague_of": RelationType("colleague_of", "Person", "Person", symmetric=True),
            },
        )
        g = WorldGraph.from_triples([("alice", "works_at", "acme"), ("bob", "works_at", "acme")])
        for p in ("alice", "bob"):
            g.add_node(p, "Person")
        g.add_node("acme", "Person")

        col_rule = Rule(
            name="same_org",
            body=(("?x", "works_at", "?o"), ("?y", "works_at", "?o")),
            head=("?x", "colleague_of", "?y"),
            distinct=(("?x", "?y"),),
        )
        entail_engine = RuleEngine(onto, [col_rule])
        entail_engine.materialise(g)

        class _MockLoRATier:
            tier_id = 4
            name = "mock-lora"

            def can_handle(self, head: str, relation: str) -> bool:
                return relation == "colleague_of"

            def answer(self, head: str, relation: str) -> TierResult | None:
                if head == "alice" and relation == "colleague_of":
                    return TierResult(tier=4, answers=["bob"], confidence=0.92)
                if head == "bob" and relation == "colleague_of":
                    return TierResult(tier=4, answers=["hallucinated"], confidence=0.80)
                return None

        teacher = CallableTeacher(lambda h, r: ["alice"])
        empty_sym = RuleEngine(onto, [])
        empty_sym.materialise(g)

        cascade = TACET(
            g,
            onto,
            teacher,
            tiers=[
                SymbolicTier(empty_sym),
                EntailmentGate(_MockLoRATier(), entail_engine, strict=True),
                LLMTier(teacher, g),
            ],
        )

        # Entailed Tier 4 prediction passes gate with symbolic proof & grounded=True
        ans_t4 = cascade.ask("alice", "colleague_of")
        assert ans_t4.tier == 4
        assert ans_t4.answers == ["bob"]
        assert ans_t4.grounded is True
        assert ans_t4.proof

        # Unentailed Tier 4 prediction is rejected by strict gate and falls through to Tier 3 LLM
        ans_t3 = cascade.ask("bob", "colleague_of")
        assert ans_t3.tier == 3
        assert ans_t3.answers == ["alice"]

        records = [ans_t4, ans_t3]
        assert entailment_rate(records, entail_engine) == pytest.approx(1.0)
        assert proof_coverage_tier4(records) == pytest.approx(1.0)
        assert grounded_fraction(records) == pytest.approx(0.5)
        rep = cascade.report()
        assert rep["tier_counts"][4] == 1
        assert rep["tier_counts"][3] == 1
