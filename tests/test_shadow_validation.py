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

The 1-hop cases exercise a length-1 latent rule; the 2-hop cases exercise the
composed relation the E11 recorded ladder actually runs on (its true rule is a
two-atom join and its junk pathology is a self-rule on the composed relation).

All fixtures are TINY and SYNTHETIC: MetaQA is never loaded or run here, and no
private path is read -- the replay record used to check the provenance guard is
constructed in-test with the same provenance shape as a real record. Streams are
injected directly so the head order (and thus the mining trigger and the unseen
checks) is fully deterministic.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_real_kg_controlled import (  # noqa: E402
    ANSWERS_RECORD_SCHEMA,
    ProvenanceMismatchError,
)
from run_shadow_validation import shadow_report  # noqa: E402

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark  # noqa: E402

#: The composed relation the E11 hop-2 recorded ladder queries (q2 pattern).
Q2 = "q2_directors_of_movies_acted_in_by"


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
    # A repeat of M0's pair with a different declared gold (the oracle's answer
    # is keyed on the LAST occurrence of a (head, relation) pair in the stream)
    # makes the run's true accuracy the repeating decimal 16/17, so a rounding
    # mismatch between the two arms' accuracy fields would actually manifest.
    stream = _stream(16) + [("M0", "directed_by", frozenset({"D5"}))]
    rep = _run(bench, stream, k=3)

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
    # the two arms replay the identical stream through the identical shared
    # teacher and neither routes a rule, so their accuracy must match exactly.
    assert rep["cache_accuracy"] == rep["shadow_accuracy"]


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


# ============================================================= 2-hop composed
# The E11 recorded ladder runs at hop 2 on the q2 composed relation, so the
# harness must promote a two-atom composed rule and reject a self-rule on that
# same composed relation.
def _hop2_true_bench(n: int) -> MetaQABenchmark:
    """A KG whose composed rule ``q2 <= ~starred_actors . directed_by`` is exact.

    Each actor ``Ai`` starred in one movie ``Mi`` directed by one director ``Di``,
    so the composed answer for ``Ai`` is exactly ``{Di}`` and the two-atom join
    predicts precisely the teacher's answer on unseen actor heads.
    """
    kg = WorldGraph(name="hop2-true")
    for i in range(n):
        a, m, d = f"A{i}", f"M{i}", f"D{i}"
        kg.add_edge(m, "starred_actors", a)  # ~starred_actors walks actor -> movie
        kg.add_edge(m, "directed_by", d)  # movie -> director
        kg.add_edge(m, "has_genre", "drama")  # background, never queried
    return _bench(kg)


def _hop2_junk_bench(n: int) -> MetaQABenchmark:
    """A KG where the only minable rule on q2 is self-referential (junk).

    There is no ``starred_actors`` / ``directed_by`` path to compose, so the true
    rule cannot form; an ``same_as`` identity self-loop lets the miner induce
    ``q2 <= same_as . q2``, whose body uses the composed target and therefore
    predicts nothing on unseen heads.
    """
    kg = WorldGraph(name="hop2-junk")
    for i in range(n):
        a = f"A{i}"
        kg.add_edge(a, "same_as", a)  # identity self-loop
        kg.add_edge(a, "has_genre", "drama")
    return _bench(kg)


def _hop2_stream(n: int):
    """A stream of n distinct actor heads on the composed relation, each once."""
    return [(f"A{i}", Q2, frozenset({f"D{i}"})) for i in range(n)]


def test_hop2_true_composed_rule_promotes_and_saves() -> None:
    bench = _hop2_true_bench(16)
    rep = _run(bench, _hop2_stream(16), k=3, hop=2, composed_relation=Q2)

    assert rep["hop"] == 2
    # the dataset label must reflect the actual workload hop, not the KB load hop.
    assert rep["dataset"] == "MetaQA-2hop-test"
    assert rep["promoted_rules"] == [f"syn:{Q2}<=~starred_actors.directed_by"]
    assert rep["demoted_rules"] == []
    assert rep["shadow_checks_used"] == 3
    assert rep["net_calls_saved_pct"] > 0.0
    assert rep["shadow_teacher_calls"] < rep["cache_teacher_calls"]
    assert rep["shadow_accuracy"] == rep["cache_accuracy"] == 1.0
    assert rep["junk_promoted"] is False


def test_hop2_junk_self_rule_on_composed_relation_never_promotes() -> None:
    bench = _hop2_junk_bench(16)
    rep = _run(bench, _hop2_stream(16), k=3, hop=2, composed_relation=Q2)

    assert rep["shadow_rules_mined"], "expected a self-rule on the composed relation"
    assert rep["promoted_rules"] == []
    assert rep["junk_promoted"] is False
    assert rep["shadow_checks_used"] == 0
    assert rep["net_calls_saved_pct"] == 0.0


# ------------------------------------------------ replay accepts a real record shape
def _hop2_record(stream, provenance: dict) -> dict:
    """A record with the same shape run_real_kg_controlled writes, over ``stream``."""
    return {
        "schema": ANSWERS_RECORD_SCHEMA,
        "spend_semantics": "measured cost of every answer served",
        "provenance": provenance,
        "answers": [
            {"head": h, "relation": r, "answers": sorted(g), "cost_usd": 0.001, "usage": None}
            for h, r, g in stream
        ],
    }


def _real_hop2_provenance(seed: int) -> dict:
    """Provenance mirroring a real hop-2 ladder record (q2 relation, openrouter)."""
    return {
        "model": "x-ai/grok-4.3",
        "price_key": "grok-4.3",
        "hop": 2,
        "split": "test",
        "limit": 300,
        "zipf_a": 1.5,
        "seed": seed,
        "composed_relation": Q2,
        "teacher_kind": "openrouter",
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }


def test_replay_accepts_real_hop2_provenance_shape(tmp_path) -> None:
    stream = _hop2_stream(4)
    record = _hop2_record(stream, _real_hop2_provenance(seed=0))
    path = tmp_path / "grok-4.3_seed0.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    rep = shadow_report(
        "grok-4.3",
        0,
        hop=2,
        split="test",
        limit=300,
        zipf_a=1.5,
        composed_relation=Q2,
        answers_path=str(path),
        bench=_hop2_true_bench(4),
        settings=_oracle_settings(),
        stream=stream,
        budget_usd=1e9,
        verbose=False,
    )
    # the guard accepts the real hop-2 provenance and the run completes.
    assert rep["hop"] == 2
    assert rep["composed_relation"] == Q2
    assert rep["cache_teacher_calls"] == rep["shadow_teacher_calls"] == 4


def test_replay_rejects_mismatched_composed_relation(tmp_path) -> None:
    stream = _hop2_stream(4)
    record = _hop2_record(stream, _real_hop2_provenance(seed=0))
    path = tmp_path / "grok-4.3_seed0.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    # a run declaring a different composed relation must be refused, not replayed.
    with pytest.raises(ProvenanceMismatchError):
        shadow_report(
            "grok-4.3",
            0,
            hop=2,
            split="test",
            limit=300,
            zipf_a=1.5,
            composed_relation="q2_WRONG_relation",
            answers_path=str(path),
            bench=_hop2_true_bench(4),
            settings=_oracle_settings(),
            stream=stream,
            budget_usd=1e9,
            verbose=False,
        )
