"""Tests for the E12 shadow-validation harness (label-free rule promotion).

The E12 mechanism runs the cascade's rule miner at a candidate confidence
(gamma_candidate = 0.50) but keeps every installed rule in SHADOW mode: it
predicts and never routes. On each teacher call for an UNSEEN head the rule
covers, its prediction is compared against the teacher's answer under the same
gold-subset criterion the benchmark uses for accuracy. A rule PROMOTES (starts
routing) after k distinct agreeing unseen heads with zero disagreements, and
DEMOTES permanently on the first disagreement. A junk self-referential rule only
reproduces already-written-back facts, so on unseen heads it predicts nothing,
is never checked, and is rejected without any gold label.

All fixtures are TINY and SYNTHETIC: MetaQA is never loaded or run here. Streams
are injected directly so the head order (and thus the mining trigger and the
unseen checks) is fully deterministic.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_shadow_validation import shadow_report  # noqa: E402

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark  # noqa: E402


# ------------------------------------------------------------- shared helpers
def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


def _bench(kg: WorldGraph) -> MetaQABenchmark:
    """A minimal benchmark carrying only the graph the arms need (no questions).

    The shadow harness is driven with an injected stream, so ``questions`` is
    unused; only ``kg`` (held-out graph + ontology induction) is read.
    """
    return MetaQABenchmark(
        name=kg.name,
        hop=1,
        split="test",
        kg=kg,
        questions=[],
        entities=set(kg.entities()),
        relations=kg.relations(),
    )


def _stream(n: int):
    """A stream of n distinct directed_by heads, each queried once, in order."""
    return [(f"M{i}", "directed_by", frozenset({f"D{i}"})) for i in range(n)]


def _true_rule_bench(n: int) -> MetaQABenchmark:
    """A KG whose latent rule ``directed_by <= made_by`` is exactly correct.

    Every movie's ``made_by`` points at its true director, so the mined rule
    predicts precisely the teacher's answer on unseen heads and should promote.
    """
    kg = WorldGraph(name="true-rule")
    for i in range(n):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)  # held out per query by the arms
        kg.add_edge(m, "made_by", d)  # base body edge, survives hold-out
        kg.add_edge(m, "has_genre", "drama")  # background, never queried
    return _bench(kg)


def _junk_rule_bench(n: int) -> MetaQABenchmark:
    """A KG where the only minable rule is self-referential (junk).

    ``same_as`` is an identity self-loop, so the miner induces
    ``directed_by <= same_as . directed_by`` — a rule whose body uses the target
    relation. It reconstructs written-back facts on complete heads but predicts
    nothing on unseen heads (their directed_by is not yet written back), so it
    must never promote.
    """
    kg = WorldGraph(name="junk-rule")
    for i in range(n):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "same_as", m)  # identity self-loop
        kg.add_edge(m, "has_genre", "drama")
    return _bench(kg)


def _wrong_rule_bench(n: int, n_train: int) -> MetaQABenchmark:
    """A KG whose latent rule looks right on training heads but wrong on unseen.

    The first ``n_train`` heads have ``made_by`` == their true director, so
    ``directed_by <= made_by`` is mined with confidence 1.0. Every later (unseen)
    head has ``made_by`` == a fixed decoy ``W`` that is not its director, so the
    rule disagrees with the teacher on the first unseen check and is demoted.
    """
    kg = WorldGraph(name="wrong-rule")
    for i in range(n):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "made_by", d if i < n_train else "W")
        kg.add_edge(m, "has_genre", "drama")
    return _bench(kg)


def _run(bench: MetaQABenchmark, stream, **kw) -> dict:
    return shadow_report(
        slug="oracle-test",
        seed=0,
        bench=bench,
        settings=_oracle_settings(),
        stream=stream,
        budget_usd=1e9,
        verbose=False,
        **kw,
    )


# ------------------------------------------------ (a) a true latent rule promotes
def test_true_rule_promotes_after_k_and_saves_calls() -> None:
    bench = _true_rule_bench(16)
    rep = _run(bench, _stream(16), k=3)

    assert rep["promoted_rules"] == ["syn:directed_by<=made_by"]
    assert rep["demoted_rules"] == []
    # exactly k distinct agreeing unseen heads were needed to promote.
    assert rep["shadow_checks_used"] == 3
    # promotion routes later covered heads, so the shadow arm calls the teacher
    # strictly less than the pure cache arm over the full stream.
    assert rep["net_calls_saved_pct"] > 0.0
    assert rep["shadow_teacher_calls"] < rep["cache_teacher_calls"]
    # accuracy is not sacrificed: the rule reproduces the teacher exactly.
    assert rep["shadow_accuracy"] == rep["cache_accuracy"] == 1.0
    assert rep["junk_promoted"] is False


# ------------------------------------------------ (b) a junk self-rule never promotes
def test_junk_self_rule_is_rejected_without_gold() -> None:
    bench = _junk_rule_bench(16)
    rep = _run(bench, _stream(16), k=3)

    # a self-referential rule WAS mined (so the test is not vacuous) ...
    assert rep["shadow_rules_mined"], "expected the miner to install a self-rule"
    # ... but it predicts nothing on unseen heads, so it is never checked and
    # never promoted -- rejected structurally, without any gold label.
    assert rep["promoted_rules"] == []
    assert rep["junk_promoted"] is False
    assert rep["shadow_checks_used"] == 0
    # with nothing promoted the shadow arm degrades to pure cache behaviour.
    assert rep["net_calls_saved_pct"] == 0.0
    assert rep["shadow_teacher_calls"] == rep["cache_teacher_calls"]


# ------------------------------------------------ (c) a disagreeing rule is demoted
def test_disagreeing_rule_is_demoted_and_degrades_to_cache() -> None:
    bench = _wrong_rule_bench(16, n_train=10)
    rep = _run(bench, _stream(16), k=3)

    assert rep["promoted_rules"] == []
    assert rep["demoted_rules"] == ["syn:directed_by<=made_by"]
    # the first unseen head disagreed, so exactly one shadow check was spent.
    assert rep["shadow_checks_used"] == 1
    # a demoted rule never routes: the shadow arm equals the cache arm, and the
    # net saving is exactly zero -- never negative.
    assert rep["net_calls_saved_pct"] == 0.0
    assert rep["shadow_teacher_calls"] == rep["cache_teacher_calls"]
    assert rep["junk_promoted"] is False


# ------------------------------------------------ (d) determinism
def test_report_is_deterministic() -> None:
    bench = _true_rule_bench(16)
    first = _run(bench, _stream(16), k=3)
    second = _run(_true_rule_bench(16), _stream(16), k=3)
    assert first == second


# ------------------------------------------------ (e) k sensitivity (2 / 5)
def test_k_sensitivity_changes_promotion_latency() -> None:
    # k=2 promotes one head earlier than k=3, so it saves at least as many calls.
    k2 = _run(_true_rule_bench(16), _stream(16), k=2)
    k5 = _run(_true_rule_bench(16), _stream(16), k=5)
    assert k2["shadow_checks_used"] == 2
    assert k5["shadow_checks_used"] == 5
    assert k2["net_calls_saved_pct"] >= k5["net_calls_saved_pct"]
    assert k2["promoted_rules"] == k5["promoted_rules"] == ["syn:directed_by<=made_by"]
