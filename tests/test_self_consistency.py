"""Tests for the E13 self-consistency harness (k-sample majority rule promotion).

E13 keeps the E12 shadow mechanism (mine at gamma_candidate = 0.50, hold every
rule in SHADOW, promote after k_prime distinct agreeing unseen heads, demote on
the first disagreement) but validates each prediction against the k-sample
MAJORITY of the teacher instead of a single sample. The majority answer set is
the entities present in at least ceil(k / 2) of the k samples.

The tests pin, on TINY SYNTHETIC KGs (MetaQA is never loaded and no private path
is read):

* (a) ``majority()`` computes the per-entity >= ceil(k/2) vote, including ties,
  within-sample de-duplication, and empties;
* (b) a true rule whose k-sample MAJORITY agrees on k_prime unseen heads promotes
  and yields net teacher-call savings > 0;
* (c) a rule whose MAJORITY disagrees on the FIRST unseen head is demoted and the
  net saving stays exactly 0 (never negative) -- even though the primary sample
  (which E12 would have compared against) agrees, so the two experiments diverge
  precisely on the majority signal;
* (d) a planted junk self-rule predicts nothing on unseen heads and never promotes;
* (e) the report is deterministic;
* (f) a k=1 record reproduces the E12 ``shadow_report`` result exactly (parity: the
  majority of a single sample is that sample);
* the k-sample RECORDER plumbing draws k teacher samples per pair, stores all k,
  and threads the sampling temperature through to the teacher.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import run_real_kg_controlled as rkc  # noqa: E402
from run_real_kg_amortization import BudgetGuard  # noqa: E402
from run_real_kg_controlled import (  # noqa: E402
    ANSWERS_RECORD_SCHEMA,
    SharedKSampleAnswerCache,
    _build_k_sample_answers_record,
    majority,
    run_controlled,
)
from run_self_consistency import self_consistency_report  # noqa: E402
from run_shadow_validation import shadow_report  # noqa: E402

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark, MetaQAQuestion  # noqa: E402
from tacet.llm.teacher import TeacherResponse  # noqa: E402

#: The rule name the miner installs for the 1-hop latent rule under test.
RULE = "syn:directed_by<=made_by"
#: A fixed decoy entity used to corrupt a fraction of the k teacher samples.
DECOY = "W"


# ================================================================ (a) majority()
def test_majority_strict_vote_over_k_samples() -> None:
    # k=3, threshold ceil(3/2)=2: an entity needs >= 2 of the 3 samples.
    assert majority([["A"], ["A"], ["B"]]) == frozenset({"A"})
    assert majority([["A", "B"], ["A", "B"], ["A", "C"]]) == frozenset({"A", "B"})


def test_majority_ties_and_de_duplication() -> None:
    # k=2, threshold ceil(2/2)=1: every entity seen at least once wins.
    assert majority([["A"], ["B"]]) == frozenset({"A", "B"})
    # k=4, threshold 2: a 2-2 tie keeps BOTH (>= threshold, not strict-more-than).
    assert majority([["A"], ["A"], ["B"], ["B"]]) == frozenset({"A", "B"})
    # a repeat WITHIN one sample counts once, so "A" is present in 2 of 3 samples.
    assert majority([["A", "A"], ["A"], ["B"]]) == frozenset({"A"})


def test_majority_edges_k1_and_empty() -> None:
    # k=1 is its own majority, so a single-sample record reduces E13 to E12.
    assert majority([["A", "B"]]) == frozenset({"A", "B"})
    # no samples, or all-empty samples, vote for nothing.
    assert majority([]) == frozenset()
    assert majority([[], [], []]) == frozenset()


# --------------------------------------------------------- synthetic benches / streams
def _bench(kg: WorldGraph) -> MetaQABenchmark:
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
    """n distinct directed_by heads, each queried once, in order."""
    return [(f"M{i}", "directed_by", frozenset({f"D{i}"})) for i in range(n)]


def _true_rule_bench(n: int) -> MetaQABenchmark:
    """A KG whose latent rule ``directed_by <= made_by`` is exactly correct.

    Every movie's ``made_by`` points at its true director, so the mined rule
    predicts precisely the true director on unseen heads.
    """
    kg = WorldGraph(name="true-rule")
    for i in range(n):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)  # held out per query by the arms
        kg.add_edge(m, "made_by", d)  # base body edge, survives hold-out
        kg.add_edge(m, "has_genre", "drama")  # background, never queried
    return _bench(kg)


def _junk_rule_bench(n: int) -> MetaQABenchmark:
    """A KG whose only minable rule is self-referential (a ``same_as`` self-loop)."""
    kg = WorldGraph(name="junk-rule")
    for i in range(n):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "same_as", m)  # identity self-loop
        kg.add_edge(m, "has_genre", "drama")
    return _bench(kg)


def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


# --------------------------------------------------------- k-sample record construction
def _hop1_provenance(seed: int = 0, k: int | None = None) -> dict:
    prov = {
        "model": "x-ai/grok-4.3",
        "price_key": "grok-4.3",
        "hop": 1,
        "split": "test",
        "limit": 300,
        "zipf_a": 1.5,
        "seed": seed,
        "composed_relation": None,
        "teacher_kind": "openrouter",
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }
    if k is not None:
        prov["k"] = k
    return prov


def _k_record(stream, samples_of, provenance: dict) -> dict:
    """A k-sample record with the shape ``_build_k_sample_answers_record`` writes.

    ``samples_of(head, relation, gold)`` returns the list of k answer-sets for a
    pair; ``answers`` is its primary sample (served for routing, exactly as a plain
    single-sample replay of ``samples[0]``).
    """
    return {
        "schema": ANSWERS_RECORD_SCHEMA,
        "spend_semantics": "measured cost of every answer served",
        "provenance": provenance,
        "answers": [
            {
                "head": h,
                "relation": r,
                "answers": samples_of(h, r, g)[0],
                "samples": samples_of(h, r, g),
                "cost_usd": 0.001,
                "usage": None,
            }
            for h, r, g in stream
        ],
    }


def _write_record(tmp_path, record: dict, slug: str = "grok-4.3", seed: int = 0) -> Path:
    path = tmp_path / f"{slug}_seed{seed}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _run_e13(bench, stream, path, **kw) -> dict:
    return self_consistency_report(
        "grok-4.3",
        0,
        hop=1,
        split="test",
        limit=300,
        zipf_a=1.5,
        composed_relation=None,
        answers_path=str(path),
        bench=bench,
        settings=_oracle_settings(),
        stream=stream,
        budget_usd=1e9,
        verbose=False,
        **kw,
    )


# ==================================================== (b) majority agrees -> promote
def test_true_rule_promotes_on_majority_and_saves_calls(tmp_path) -> None:
    stream = _stream(16)
    # Every pair: primary + one more clean sample (director D_i) and one decoy, so the
    # MAJORITY is {D_i} while the served/write-back sample stays clean (rule mines).
    record = _k_record(
        stream,
        lambda h, r, g: [sorted(g), sorted(g), [DECOY]],
        _hop1_provenance(k=3),
    )
    path = _write_record(tmp_path, record)
    rep = _run_e13(_true_rule_bench(16), stream, path, k_prime=3)

    assert rep["promoted_rules"] == [RULE]
    assert rep["demoted_rules"] == []
    # exactly k_prime distinct agreeing unseen heads promoted the rule.
    assert rep["shadow_checks_used"] == 3
    # the extra teacher cost is k samples per check.
    assert rep["k"] == 3
    assert rep["self_consistency_calls"] == 3 * 3
    # promotion routes later covered heads, so net teacher-call savings are positive.
    assert rep["net_calls_saved_pct"] > 0.0
    assert rep["shadow_teacher_calls"] < rep["cache_teacher_calls"]
    assert rep["shadow_accuracy"] == rep["cache_accuracy"] == 1.0
    assert rep["junk_promoted"] is False


# ==================================================== (c) majority disagrees -> demote
def test_majority_disagreement_demotes_even_when_primary_agrees(tmp_path) -> None:
    stream = _stream(16)
    # Primary sample is the true director (so the rule mines AND E12's single-sample
    # signal would AGREE), but two of three samples are the decoy, so the MAJORITY is
    # {W} and disagrees with the rule's {D_i} on the first unseen check.
    record = _k_record(
        stream,
        lambda h, r, g: [sorted(g), [DECOY], [DECOY]],
        _hop1_provenance(k=3),
    )
    path = _write_record(tmp_path, record)
    rep = _run_e13(_true_rule_bench(16), stream, path, k_prime=3)

    assert rep["promoted_rules"] == []
    assert rep["demoted_rules"] == [RULE]
    # the first unseen head disagreed, so exactly one shadow check was spent.
    assert rep["shadow_checks_used"] == 1
    # a demoted rule never routes: the shadow arm equals the cache arm, net is 0 (never < 0).
    assert rep["net_calls_saved_pct"] == 0.0
    assert rep["net_calls_saved_pct"] >= 0.0
    assert rep["shadow_teacher_calls"] == rep["cache_teacher_calls"]
    assert rep["junk_promoted"] is False

    # CONTRAST: the SAME record run through E12 (single-sample = the primary D_i) PROMOTES,
    # so the demotion above is caused by the majority signal, not the served answer.
    e12 = shadow_report(
        "grok-4.3",
        0,
        hop=1,
        split="test",
        limit=300,
        zipf_a=1.5,
        composed_relation=None,
        answers_path=str(path),
        bench=_true_rule_bench(16),
        settings=_oracle_settings(),
        stream=stream,
        budget_usd=1e9,
        verbose=False,
    )
    assert e12["promoted_rules"] == [RULE]
    assert e12["demoted_rules"] == []


# ==================================================== (d) junk self-rule never promotes
def test_junk_self_rule_is_rejected_without_gold() -> None:
    # Oracle mode (no record): the junk rule predicts nothing on unseen heads, so it is
    # never checked -- rejected structurally, without any gold label, exactly as in E12.
    rep = self_consistency_report(
        "oracle",
        0,
        bench=_junk_rule_bench(16),
        settings=_oracle_settings(),
        stream=_stream(16),
        budget_usd=1e9,
        k=3,
        k_prime=3,
        verbose=False,
    )
    assert rep["shadow_rules_mined"], "expected the miner to install a self-rule"
    assert rep["promoted_rules"] == []
    assert rep["junk_promoted"] is False
    assert rep["shadow_checks_used"] == 0
    assert rep["self_consistency_calls"] == 0
    assert rep["net_calls_saved_pct"] == 0.0


# ==================================================== (e) determinism
def test_report_is_deterministic(tmp_path) -> None:
    stream = _stream(16)
    record = _k_record(
        stream, lambda h, r, g: [sorted(g), sorted(g), [DECOY]], _hop1_provenance(k=3)
    )
    path = _write_record(tmp_path, record)
    first = _run_e13(_true_rule_bench(16), stream, path, k_prime=3)
    second = _run_e13(_true_rule_bench(16), stream, path, k_prime=3)
    assert first == second


# ==================================================== (f) k=1 record reproduces E12
def test_k1_record_reproduces_e12_shadow_report(tmp_path) -> None:
    stream = _stream(16)
    # A genuine k=1 record: one sample per pair, the true director. Its majority is that
    # sole sample, so E13's majority signal is identical to E12's single-sample signal.
    record = _k_record(stream, lambda h, r, g: [sorted(g)], _hop1_provenance(k=1))
    path = _write_record(tmp_path, record)

    e13 = _run_e13(_true_rule_bench(16), stream, path, k_prime=3)
    e12 = shadow_report(
        "grok-4.3",
        0,
        hop=1,
        split="test",
        limit=300,
        zipf_a=1.5,
        composed_relation=None,
        answers_path=str(path),
        bench=_true_rule_bench(16),
        settings=_oracle_settings(),
        stream=stream,
        budget_usd=1e9,
        verbose=False,
    )
    # Every signal-bearing field matches exactly (E13 adds self_consistency_calls / k_prime;
    # E12 lacks them, so only the shared keys are compared).
    for key in (
        "net_calls_saved_pct",
        "promoted_rules",
        "demoted_rules",
        "shadow_rules_mined",
        "shadow_checks_used",
        "cache_teacher_calls",
        "shadow_teacher_calls",
        "cache_accuracy",
        "shadow_accuracy",
        "junk_promoted",
    ):
        assert e13[key] == e12[key], f"parity broken on {key!r}: {e13[key]!r} != {e12[key]!r}"
    # the k=1 record is priced as a single sample per check (no extra self-consistency cost).
    assert e13["k"] == 1
    assert e13["self_consistency_calls"] == e13["shadow_checks_used"]


# ==================================================== recorder plumbing (deliverable 4)
class _CannedMeteredTeacher:
    """A metered-teacher stub: canned per-call answers, counts every call (no network)."""

    def __init__(self, temperature=None) -> None:  # noqa: ANN001
        self.temperature = temperature  # captured to prove the value threaded through
        self.calls: list[tuple[str, str]] = []
        self._per_pair: dict[tuple[str, str], int] = defaultdict(int)
        self.last_cost_usd = 0.0
        self.last_usage: dict | None = None

    def answer(self, graph, head: str, relation: str) -> TeacherResponse:  # noqa: ANN001
        idx = self._per_pair[(head, relation)]
        self._per_pair[(head, relation)] += 1
        self.calls.append((head, relation))
        self.last_cost_usd = 0.01
        self.last_usage = {"prompt_tokens": 4, "completion_tokens": 2}
        # A distinct answer per sample so k independent draws are visible as k samples.
        return TeacherResponse(answers=[f"{head}_s{idx}"])


def test_k_sample_cache_draws_k_samples_and_stores_them() -> None:
    fake = _CannedMeteredTeacher()
    kg = WorldGraph(name="empty")
    guard = BudgetGuard(1e9)
    cache = SharedKSampleAnswerCache(fake, guard, kg, n_samples=3)

    served, cost = cache.answer("M0", "directed_by")
    # the cold pair drew exactly k=3 teacher answers ...
    assert len(fake.calls) == 3
    assert cache.samples[("M0", "directed_by")] == [["M0_s0"], ["M0_s1"], ["M0_s2"]]
    # ... the served answer is the PRIMARY sample and its cost is the k-sample sum.
    assert served == ["M0_s0"]
    assert cost == pytest.approx(0.03)

    # a second request re-serves the cache without re-sampling.
    cache.answer("M0", "directed_by")
    assert len(fake.calls) == 3

    # the record carries all k samples per pair, self-describing with k in provenance.
    record = _build_k_sample_answers_record(cache, provenance={"k": 3, "temperature": 0.7})
    row = record["answers"][0]
    assert row["samples"] == [["M0_s0"], ["M0_s1"], ["M0_s2"]]
    assert row["answers"] == ["M0_s0"]
    assert len(row["sample_costs"]) == 3

    # writer -> reader contract: a record the recorder wrote replays through the k-sample
    # cache, serving the primary sample for routing and voting the majority (M0_s* all
    # differ, so with threshold 2 the majority is empty).
    replay = rkc.KSampleReplayAnswerCache.from_record(record, BudgetGuard(1e9), kg)
    assert replay.answer("M0", "directed_by")[0] == ["M0_s0"]
    assert replay.majority("M0", "directed_by") == frozenset()


def test_openrouter_teacher_threads_temperature_to_client() -> None:
    pytest.importorskip("openai")
    from tacet.llm.teachers.llm import OpenRouterTeacher

    captured: list[dict] = []

    def _make(**kwargs):  # noqa: ANN003, ANN202
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='["Belgium"]'))],
            usage=None,
        )

    teacher = OpenRouterTeacher("or-test-key", "x-ai/grok-4.3", temperature=0.7)
    teacher._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_make))
    )
    teacher.answer(None, "France", "borders")
    assert captured[0]["temperature"] == 0.7


def test_openrouter_teacher_no_temperature_key_when_unset() -> None:
    pytest.importorskip("openai")
    from tacet.llm.teachers.llm import OpenRouterTeacher

    captured: list[dict] = []

    def _make(**kwargs):  # noqa: ANN003, ANN202
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='["Belgium"]'))],
            usage=None,
        )

    teacher = OpenRouterTeacher("or-test-key", "x-ai/grok-4.3")  # temperature unset
    teacher._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_make))
    )
    teacher.answer(None, "France", "borders")
    # the default path stays byte-identical to today: no temperature field is sent.
    assert "temperature" not in captured[0]


def _tiny_bench() -> MetaQABenchmark:
    kg = WorldGraph(name="tiny-movies")
    questions: list[MetaQAQuestion] = []
    for i in range(8):
        m, d = f"M{i}", f"D{i % 2}"
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "has_genre", "drama")
        questions.append(
            MetaQAQuestion(question=f"who directed [{m}]?", head=m, answers=[d], hop=1)
        )
    return MetaQABenchmark(
        name="tiny-movies",
        hop=1,
        split="test",
        kg=kg,
        questions=questions,
        entities=set(kg.entities()),
        relations=kg.relations(),
    )


def test_run_controlled_records_k_sample_record(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def factory(  # noqa: ANN001, ANN202
        settings,
        model,
        nl_template,
        oracle_gold=None,
        error_rate=0.0,
        seed=0,
        response_format=None,
        temperature=None,
    ):
        captured["temperature"] = temperature  # proves --temperature threaded through
        return _CannedMeteredTeacher(temperature=temperature)

    monkeypatch.setattr(rkc, "_new_metered", factory)
    path = tmp_path / "ksample.json"
    run_controlled(
        hop=1,
        split="test",
        limit=16,
        zipf_a=1.5,
        seed=0,
        oracle_error_rate=0.0,
        gamma=0.95,
        answers_path=str(path),
        samples=3,
        temperature=0.7,
        bench=_tiny_bench(),
        settings=_oracle_settings(),
        verbose=False,
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    # the record is a self-describing k-sample artifact ...
    assert record["provenance"]["k"] == 3
    assert record["provenance"]["temperature"] == 0.7
    assert captured["temperature"] == 0.7
    # ... every pair stored all k samples, and every distinct pair cost 3 teacher calls.
    assert record["answers"], "expected at least one recorded pair"
    for row in record["answers"]:
        assert len(row["samples"]) == 3
        assert len(row["sample_costs"]) == 3


def test_samples_and_temperature_refused_on_replay(tmp_path, monkeypatch) -> None:
    def factory(  # noqa: ANN001, ANN202
        settings,
        model,
        nl_template,
        oracle_gold=None,
        error_rate=0.0,
        seed=0,
        response_format=None,
        temperature=None,
    ):
        return _CannedMeteredTeacher(temperature=temperature)

    monkeypatch.setattr(rkc, "_new_metered", factory)
    path = tmp_path / "ksample.json"
    common = dict(
        hop=1,
        split="test",
        limit=16,
        zipf_a=1.5,
        seed=0,
        oracle_error_rate=0.0,
        gamma=0.95,
        bench=_tiny_bench(),
        settings=_oracle_settings(),
        verbose=False,
    )
    # record a k-sample file first so the next run REPLAYS it ...
    run_controlled(answers_path=str(path), samples=3, temperature=0.7, **common)
    # ... and re-sampling a fixed record is refused (its k is a property of the record).
    with pytest.raises(SystemExit) as excinfo:
        run_controlled(answers_path=str(path), samples=3, temperature=0.7, **common)
    msg = str(excinfo.value).lower()
    assert "replay" in msg or "record" in msg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
