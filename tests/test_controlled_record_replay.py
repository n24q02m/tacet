"""Record/replay tests for the controlled runner's teacher-answer cache (E11).

The E11 gamma sweep must pay the real teacher ONCE and then replay byte-identical
answers for every gamma, so gamma is the only variable and the metered bill is
reproduced without re-calling. These tests pin that contract with a FAKE teacher
(no network, no real LLM): a call-counting stand-in is monkeypatched in for
``_new_metered`` so we can assert exactly how many times the teacher is invoked.

All fixtures are TINY and SYNTHETIC: MetaQA is never loaded or run here.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import run_real_kg_controlled as rkc  # noqa: E402
from run_real_kg_amortization import BudgetGuard  # noqa: E402
from run_real_kg_controlled import (  # noqa: E402
    MissingRecordedAnswerError,
    ProvenanceMismatchError,
    ReplayAnswerCache,
    run_controlled,
)

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark, MetaQAQuestion  # noqa: E402
from tacet.llm.teacher import TeacherResponse  # noqa: E402


# ------------------------------------------------------------- fake teacher
class _FakeMeteredTeacher:
    """Call-counting stand-in for the metered teacher run_controlled builds.

    Duck-types the ``MeteredTeacher`` surface :class:`SharedAnswerCache` reads
    (``answer`` + ``last_cost_usd`` + ``last_usage``). It returns the workload's
    own gold (a PERFECT teacher, so accuracy is deterministic and matched) at a
    fixed non-zero canned cost, and bumps a shared counter on every call so a test
    can assert it is called N times on record and exactly 0 times on replay.
    """

    def __init__(self, oracle_gold, counter, cost_per_call: float = 0.01) -> None:
        self._gold = oracle_gold or {}
        self._counter = counter
        self._cost = cost_per_call
        self.last_cost_usd = 0.0
        self.last_usage: dict | None = None

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        self._counter["calls"] += 1
        self.last_cost_usd = self._cost
        self.last_usage = {"prompt_tokens": 4, "completion_tokens": 2}
        return TeacherResponse(answers=sorted(self._gold.get(f"{head}\t{relation}", ())))


def _install_fake_teacher(monkeypatch, counter: dict) -> None:
    """Replace ``run_real_kg_controlled._new_metered`` with the counting fake.

    Signature mirrors the real ``_new_metered`` so run_controlled's call site is
    unchanged; on a replay run run_controlled never calls it, so the fake teacher
    is never constructed and the counter cannot move.
    """

    def factory(settings, model, nl_template, oracle_gold=None, error_rate=0.0, seed=0):  # noqa: ANN001
        return _FakeMeteredTeacher(oracle_gold, counter)

    monkeypatch.setattr(rkc, "_new_metered", factory)


# --------------------------------------------------------------- synthetic benches
def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


def _tiny_bench() -> MetaQABenchmark:
    """A few movies with directors + actors (flat 1-hop, no latent rule)."""
    kg = WorldGraph(name="tiny-movies")
    questions: list[MetaQAQuestion] = []
    directors = ["D1", "D2"]
    actors = ["A1", "A2", "A3"]
    for i in range(10):
        m = f"M{i}"
        d = directors[i % len(directors)]
        a = actors[i % len(actors)]
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "starred_actors", a)
        kg.add_edge(m, "has_genre", "drama")  # background: never queried
        questions.append(
            MetaQAQuestion(question=f"who directed [{m}]?", head=m, answers=[d], hop=1)
        )
        questions.append(
            MetaQAQuestion(question=f"who acted in [{m}]?", head=m, answers=[a], hop=1)
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


def _latent_composition_bench() -> MetaQABenchmark:
    """A KG with a latent length-1 rule whose confidence is EXACTLY 0.5.

    ``directed_by`` is the only queried relation (gold ``Mi -> Di``). A background
    ``made_by`` relation carries the true director plus a shared decoy ``W``, so the
    rule ``directed_by <= made_by`` has confidence 0.5: a LOW gamma installs it, a
    HIGH gamma does not. That makes the install contrast (and thus the arm outputs)
    depend on gamma while the teacher answers do not.
    """
    kg = WorldGraph(name="latent-comp")
    questions: list[MetaQAQuestion] = []
    for i in range(24):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "made_by", d)
        kg.add_edge(m, "made_by", "W")
        kg.add_edge(m, "has_genre", "drama")
        questions.append(
            MetaQAQuestion(question=f"who directed [{m}]?", head=m, answers=[d], hop=1)
        )
    return MetaQABenchmark(
        name="latent-comp",
        hop=1,
        split="test",
        kg=kg,
        questions=questions,
        entities=set(kg.entities()),
        relations=kg.relations(),
    )


def _common(bench, settings) -> dict:
    return dict(
        hop=1,
        split="test",
        limit=24,
        zipf_a=1.5,
        seed=0,
        oracle_error_rate=0.0,
        gamma=0.95,
        bench=bench,
        settings=settings,
        verbose=False,
    )


# ---------------------------------------------------------- 1. record then replay is free
def test_record_then_replay_is_free_and_identical(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _oracle_settings()
    path = tmp_path / "answers.json"

    rec = run_controlled(answers_path=str(path), **_common(bench, settings))
    # N = number of distinct (head, relation) pairs; the teacher answered each once.
    n_distinct = rec["distinct_queries"]
    assert counter["calls"] == n_distinct == rec["real_teacher_calls"]
    assert path.exists()

    calls_before_replay = counter["calls"]
    rep = run_controlled(answers_path=str(path), **_common(bench, settings))

    # THE fake teacher is not called at all on replay.
    assert counter["calls"] == calls_before_replay

    # The report is reproduced exactly: spend, per-arm calls/accuracy/rules, and the
    # teacher's own answer accuracy all match the recording run.
    assert rep["total_measured_spend_usd"] == rec["total_measured_spend_usd"]
    assert rep["total_measured_spend_usd"] > 0.0  # a real (non-zero) bill was reproduced
    assert rep["teacher_answer_accuracy"] == rec["teacher_answer_accuracy"]
    rec_arms = {a["arm"]: a for a in rec["arms"]}
    rep_arms = {a["arm"]: a for a in rep["arms"]}
    assert rep_arms.keys() == rec_arms.keys()
    for arm in rec_arms:
        assert rep_arms[arm]["teacher_calls"] == rec_arms[arm]["teacher_calls"]
        assert rep_arms[arm]["accuracy"] == rec_arms[arm]["accuracy"]
        assert rep_arms[arm].get("synthesised_rules") == rec_arms[arm].get("synthesised_rules")


# --------------------------------------------------------------- 2. gamma is the only variable
def test_gamma_is_the_only_variable(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _latent_composition_bench(), _oracle_settings()
    path = tmp_path / "answers.json"
    common = dict(
        hop=1,
        split="test",
        limit=150,
        zipf_a=1.2,
        seed=0,
        oracle_error_rate=0.0,
        bench=bench,
        settings=settings,
        verbose=False,
    )

    rec = run_controlled(gamma=0.95, answers_path=str(path), **common)
    calls_before_replays = counter["calls"]

    low = run_controlled(gamma=0.5, answers_path=str(path), **common)
    high = run_controlled(gamma=0.99, answers_path=str(path), **common)

    # zero teacher calls across BOTH replays
    assert counter["calls"] == calls_before_replays

    # the teacher answers (the recorded cache contents) are identical regardless of
    # gamma: the cache-derived metrics match across both replays and the recording.
    assert (
        low["teacher_answer_accuracy"]
        == high["teacher_answer_accuracy"]
        == (rec["teacher_answer_accuracy"])
    )
    assert low["total_measured_spend_usd"] == high["total_measured_spend_usd"]

    # assert directly on the recorded cache contents (not the arms' outputs)
    record = json.loads(path.read_text(encoding="utf-8"))
    replayed = ReplayAnswerCache.from_record(record, BudgetGuard(budget_usd=1e9), bench.kg)
    assert len(replayed.answers) == rec["distinct_queries"]

    # ...while the arms DO differ by gamma: the low gamma installs the latent rule,
    # the high gamma does not.
    assert low["verdict"]["rule_installed"] is True
    assert high["verdict"]["rule_installed"] is False
    assert low["verdict"]["synthesised_rules"] != high["verdict"]["synthesised_rules"]


# ----------------------------------------------------- 3. a missing pair is a loud error
def test_missing_pair_is_loud_never_an_api_call(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _oracle_settings()
    path = tmp_path / "answers.json"
    run_controlled(answers_path=str(path), **_common(bench, settings))

    record = json.loads(path.read_text(encoding="utf-8"))
    dropped = record["answers"].pop(0)  # remove one recorded pair the stream requests
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")

    calls_before_replay = counter["calls"]
    with pytest.raises(MissingRecordedAnswerError) as excinfo:
        run_controlled(answers_path=str(tampered), **_common(bench, settings))

    msg = str(excinfo.value)
    assert dropped["head"] in msg
    assert dropped["relation"] in msg
    # no silent fall-back to a real teacher call
    assert counter["calls"] == calls_before_replay


# ------------------------------------------------ 4. a parameter mismatch refuses to replay
def test_parameter_mismatch_refuses_to_replay(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _oracle_settings()
    path = tmp_path / "answers.json"
    base = dict(
        hop=1,
        split="test",
        limit=24,
        zipf_a=1.5,
        oracle_error_rate=0.0,
        gamma=0.95,
        bench=bench,
        settings=settings,
        verbose=False,
    )
    run_controlled(seed=0, answers_path=str(path), **base)

    calls_before_replay = counter["calls"]
    with pytest.raises(ProvenanceMismatchError) as excinfo:
        run_controlled(seed=1, answers_path=str(path), **base)

    assert "seed" in str(excinfo.value)
    # refused before touching the teacher
    assert counter["calls"] == calls_before_replay


# ------------------------------------------------------- 5. answers_path=None is unchanged
def test_answers_path_none_is_unchanged(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _oracle_settings()

    base = run_controlled(answers_path=None, **_common(bench, settings))
    n_calls_plain = counter["calls"]

    path = tmp_path / "answers.json"
    rec = run_controlled(answers_path=str(path), **_common(bench, settings))

    # a recording run behaves EXACTLY as answers_path=None, only additionally writing
    # the file (the returned report is identical).
    assert base == rec
    assert path.exists()
    # both runs made the same number of real teacher calls (recording adds no calls)
    assert counter["calls"] - n_calls_plain == n_calls_plain

    # the existing report keys are all still present (schema unchanged)
    assert {
        "dataset",
        "hop",
        "arms",
        "verdict",
        "total_measured_spend_usd",
        "real_teacher_calls",
        "teacher_answer_accuracy",
        "gamma",
        "seed",
    } <= set(base)


if __name__ == "__main__":
    import unittest

    unittest.main()
