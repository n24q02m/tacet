"""Tests for the pluggable Tier protocol, built-in tiers, EntailmentGate, and custom cascades."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from tacet import (
    TACET,
    CascadeConfig,
    EntailmentCheck,
    EntailmentGate,
    KGEConfig,
    KGETier,
    LLMTier,
    Narrator,
    Ontology,
    Rule,
    RuleEngine,
    SymbolicTier,
    Tier,
    TierResult,
    WorldGraph,
)
from tacet.cascade.config import TIER_COST, TIER_LATENCY_MS
from tacet.core.ontology import NodeType, RelationType
from tacet.core.symbolic import Derivation
from tacet.kge.kge import ComplEx, Prediction
from tacet.llm.teacher import OracleTeacher


class _MockLoRATier:
    """Minimal custom tier simulating a distilled LoRA model (Tier 4)."""

    tier_id = 4
    name = "lora"

    def __init__(self, answers_map: dict[tuple[str, str], list[str]], confidence: float = 0.9):
        self._map = answers_map
        self._confidence = confidence

    def can_handle(self, head: str, relation: str) -> bool:
        return (head, relation) in self._map

    def answer(self, head: str, relation: str) -> TierResult | None:
        ans = self._map.get((head, relation))
        if ans is None:
            return None
        return TierResult(
            tier=self.tier_id,
            answers=list(ans),
            confidence=self._confidence,
            proof=[],
            grounded=False,
            note="LoRA prediction",
        )


def _build_fixture():
    g = WorldGraph()
    g.add_node("Alice", "Person")
    g.add_node("Bob", "Person")
    g.add_node("Carol", "Person")
    g.add_node("ACME", "Company")
    g.add_edge("Alice", "parent_of", "Bob")
    g.add_edge("Bob", "parent_of", "Carol")
    g.add_edge("Alice", "works_at", "ACME")

    onto = Ontology()
    onto.add_node_type(NodeType("Person"))
    onto.add_node_type(NodeType("Company"))
    onto.add_relation_type(RelationType("parent_of", frozenset({"Person"}), frozenset({"Person"})))
    onto.add_relation_type(
        RelationType("grandparent_of", frozenset({"Person"}), frozenset({"Person"}))
    )
    onto.add_relation_type(RelationType("works_at", frozenset({"Person"}), frozenset({"Company"})))

    gp_rule = Rule(
        "gp_rule",
        (("?x", "parent_of", "?z"), ("?z", "parent_of", "?y")),
        ("?x", "grandparent_of", "?y"),
    )
    engine = RuleEngine(onto, [gp_rule])
    engine.materialise(g)

    def oracle(h: str, r: str) -> list[str]:
        if h == "Bob" and r == "works_at":
            return ["Globex"]
        return []

    teacher = OracleTeacher(oracle)
    return g, onto, engine, teacher


def test_public_exports_and_tier_result_frozen():
    assert EntailmentCheck is not None
    res = TierResult(tier=4, answers=["Carol"], confidence=0.9)
    assert res.proof == []
    assert res.grounded is False
    assert res.note == ""
    with pytest.raises(FrozenInstanceError):
        res.tier = 1  # type: ignore[misc]


def test_symbolic_tier():
    _, _, engine, _ = _build_fixture()
    tier = SymbolicTier(engine)
    assert isinstance(tier, Tier)
    assert tier.tier_id == 1
    assert tier.name == "symbolic"
    assert tier.can_handle("Alice", "grandparent_of") is True

    hit = tier.answer("Alice", "grandparent_of")
    assert hit is not None
    assert hit.tier == 1
    assert hit.answers == ["Carol"]
    assert hit.confidence == 1.0
    assert hit.grounded is True
    assert hit.proof

    miss = tier.answer("Bob", "works_at")
    assert miss is None


def test_kge_tier():
    g, onto, _, _ = _build_fixture()
    kge = ComplEx(KGEConfig(epochs=5, seed=0))
    tier_unfit = KGETier(kge, g, onto, threshold=0.6)
    assert isinstance(tier_unfit, Tier)
    assert tier_unfit.tier_id == 2
    assert tier_unfit.name == "kge"
    assert tier_unfit.can_handle("Alice", "works_at") is False
    assert tier_unfit.answer("Alice", "works_at") is None

    # Mock KGE predictor to deterministically test threshold & ontology gate branches.
    mock_kge = MagicMock()
    mock_kge.ent = {"Alice": 0, "Bob": 1, "ACME": 2}
    mock_kge.rel = {"works_at": 0, "unknown_rel": 1}

    tier = KGETier(mock_kge, g, onto, threshold=0.6)
    assert tier.can_handle("Alice", "works_at") is True
    assert tier.can_handle("Nobody", "works_at") is False

    # Below threshold -> None
    mock_kge.predict_tail.return_value = Prediction(tail="ACME", confidence=0.4)
    assert tier.answer("Alice", "works_at") is None

    # Ontology rejects (Bob is Person, not Company) -> None
    mock_kge.predict_tail.return_value = Prediction(tail="Bob", confidence=0.95)
    assert tier.answer("Alice", "works_at") is None

    # Accepted -> TierResult
    mock_kge.predict_tail.return_value = Prediction(tail="ACME", confidence=0.85)
    accepted = tier.answer("Alice", "works_at")
    assert accepted is not None
    assert accepted.tier == 2
    assert accepted.answers == ["ACME"]
    assert accepted.confidence == 0.85
    assert accepted.grounded is False
    assert "conf=0.85" in accepted.note

    # Unknown relation falls back to graph.entities() for candidate lookup
    mock_kge.predict_tail.return_value = None
    assert tier.answer("Alice", "unknown_rel") is None


def test_llm_tier():
    g, _, _, teacher = _build_fixture()
    tier = LLMTier(teacher, g)
    assert isinstance(tier, Tier)
    assert tier.tier_id == 3
    assert tier.name == "llm"
    assert tier.can_handle("Bob", "works_at") is True

    res = tier.answer("Bob", "works_at")
    assert res is not None
    assert res.tier == 3
    assert res.answers == ["Globex"]
    assert res.confidence == 1.0
    assert res.grounded is False


def test_entailment_gate_accept_filter_and_dedup_proof():
    _, _, engine, _ = _build_fixture()
    # Both Carol (entailed) and Phantom (unentailed), plus duplicate check
    mock = _MockLoRATier({("Alice", "grandparent_of"): ["Carol", "Phantom", "Carol"]})
    gate = EntailmentGate(mock, engine, strict=True)
    assert isinstance(gate, Tier)
    assert gate.tier_id == 4
    assert gate.name == "gate(lora)"
    assert gate.can_handle("Alice", "grandparent_of") is True
    assert gate.can_handle("Bob", "works_at") is False

    # Inner returns None -> Gate returns None
    assert gate.answer("Bob", "works_at") is None

    # Inner returns mix of entailed and unentailed -> keeps entailed with unique proof lines
    res = gate.answer("Alice", "grandparent_of")
    assert res is not None
    assert res.tier == 4
    assert res.answers == ["Carol"]
    assert res.grounded is True
    assert len(res.proof) == len(set(res.proof))
    assert any("DERIVED" in line for line in res.proof)


def test_entailment_gate_ungrounded_derivation():
    _, _, engine, _ = _build_fixture()
    bad_triple = ("Alice", "grandparent_of", "Ghost")
    engine._closure.add(bad_triple)
    engine._derivations[bad_triple] = Derivation(bad_triple, "dangling_rule", ())

    mock = _MockLoRATier({("Alice", "grandparent_of"): ["Ghost"]})
    gate = EntailmentGate(mock, engine, strict=True)
    res = gate.answer("Alice", "grandparent_of")
    assert res is not None
    assert res.answers == ["Ghost"]
    assert res.grounded is False
    assert res.proof


def test_entailment_gate_strict_vs_non_strict_reject():
    _, _, engine, _ = _build_fixture()
    mock = _MockLoRATier({("Bob", "works_at"): ["WrongCo"]})

    strict_gate = EntailmentGate(mock, engine, strict=True)
    assert strict_gate.answer("Bob", "works_at") is None

    lenient_gate = EntailmentGate(mock, engine, strict=False)
    res = lenient_gate.answer("Bob", "works_at")
    assert res is not None
    assert res.tier == 4
    assert res.answers == ["WrongCo"]
    assert res.grounded is False
    assert res.note == "unentailed"


def test_config_and_narrator_tier4():
    assert TIER_COST[4] == pytest.approx(0.0005)
    assert TIER_LATENCY_MS[4] == pytest.approx(40.0)

    narrator = Narrator()
    assert narrator.tier_label[4] == "distilled LoRA"

    text4 = narrator.render("Alice", "grandparent_of", ["Carol"], tier=4, proof=["DERIVED ..."])
    assert "[via distilled LoRA]" in text4
    assert "proof:" in text4
    assert "DERIVED ..." in text4

    # Non-tier-1 with proof also renders proof
    text2 = narrator.render("Alice", "works_at", ["ACME"], tier=2, proof=["step A"])
    assert "[via structural inference]" in text2
    assert "proof:" in text2

    # Unknown tier falls back gracefully
    text99 = narrator.render("Alice", "works_at", ["ACME"], tier=99)
    assert "[via tier 99]" in text99


def test_tacet_custom_tiers_gate_accept_and_fallthrough():
    g, onto, engine, teacher = _build_fixture()
    # Custom cascade with EntailmentGate(LoRA) before SymbolicTier to observe Tier 4 acceptance,
    # and fallthrough to LLMTier when LoRA predicts an unentailed answer.
    lora = _MockLoRATier(
        {
            ("Alice", "grandparent_of"): ["Carol"],  # entailed -> gate accepts (Tier 4)
            ("Bob", "works_at"): ["HallucinatedCo"],  # unentailed -> strict gate rejects -> Tier 3
        }
    )
    tiers = [
        EntailmentGate(lora, engine, strict=True),
        SymbolicTier(engine),
        LLMTier(teacher, g),
    ]
    ak = TACET(g, onto, teacher, config=CascadeConfig(kge=KGEConfig(epochs=1)), tiers=tiers)

    ans_t4 = ak.ask("Alice", "grandparent_of")
    assert ans_t4.tier == 4
    assert ans_t4.answers == ["Carol"]
    assert ans_t4.grounded is True
    assert ans_t4.proof
    assert ans_t4.cost == pytest.approx(0.0005)
    assert ans_t4.latency_ms == pytest.approx(40.0)
    assert "[via distilled LoRA]" in ans_t4.text
    assert "proof:" in ans_t4.text

    ans_t1 = ak.ask("Alice", "works_at")
    assert ans_t1.tier == 1
    assert ans_t1.answers == ["ACME"]
    assert ans_t1.grounded is True

    ans_t3 = ak.ask("Bob", "works_at")
    assert ans_t3.tier == 3
    assert ans_t3.answers == ["Globex"]
    assert ans_t3.grounded is False

    rep = ak.report()
    assert rep["queries"] == 3
    assert rep["tier1"] == 1
    assert rep["tier2"] == 0
    assert rep["tier3"] == 1
    assert rep["tier4"] == 1
    assert rep["tier_counts"] == {1: 1, 2: 0, 3: 1, 4: 1}
    assert rep["tier1_share"] == pytest.approx(1 / 3)
    assert rep["tier2_share"] == pytest.approx(0.0)
    assert rep["tier3_share"] == pytest.approx(1 / 3)
    assert rep["tier4_share"] == pytest.approx(1 / 3)
    assert rep["tier_shares"][4] == pytest.approx(1 / 3)
    assert rep["pct_no_llm"] == pytest.approx(2 / 3 * 100)


def test_tacet_custom_tier_unknown_id_and_all_none_fallback():
    g, onto, _, teacher = _build_fixture()

    class _UnknownTier:
        tier_id = 77
        name = "custom77"

        def can_handle(self, head: str, relation: str) -> bool:
            return head == "Alice"

        def answer(self, head: str, relation: str) -> TierResult | None:
            if head == "Alice":
                return TierResult(tier=77, answers=["ACME"], confidence=0.7, grounded=False)
            return None

    ak = TACET(g, onto, teacher, tiers=[_UnknownTier()])
    ans_hit = ak.ask("Alice", "works_at")
    assert ans_hit.tier == 77
    assert ans_hit.cost == 0.0
    assert ans_hit.latency_ms == 0.0
    assert "[via tier 77]" in ans_hit.text

    ans_miss = ak.ask("Bob", "works_at")
    assert ans_miss.answers == []
    assert ans_miss.grounded is False
    assert ans_miss.note == "no tier answered"
