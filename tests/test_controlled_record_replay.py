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

    def factory(  # noqa: ANN001
        settings, model, nl_template, oracle_gold=None, error_rate=0.0, seed=0, response_format=None
    ):
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


# =====================================================================================
# Crash-safety / resume suite (kept in this file so it reuses the tiny synthetic bench,
# oracle settings and ``_common`` above; the record layer's durability is a property of
# the SAME runner, so co-locating the tests keeps one contract in one place). Everything
# above is unchanged. These pin that a paid teacher call, once obtained, survives the
# process being killed mid-record and is never re-bought on resume.
# =====================================================================================


class _RecordingKilledError(RuntimeError):
    """Stand-in for the container dying mid-recording (a non-budget crash)."""


class _CountingTeacher:
    """A perfect fake teacher that counts calls and can 'die' after ``raise_after`` of them.

    Duck-types the metered-teacher surface :class:`SharedAnswerCache` reads. Returns the
    workload's own gold at a fixed non-zero cost, records every ``(head, relation)`` it is
    asked (so a test can assert it is NEVER asked for a warm pair), and — when
    ``raise_after`` is set — raises :class:`_RecordingKilledError` on the call that WOULD be its
    ``(raise_after + 1)``-th, so exactly ``raise_after`` answers are produced before the
    simulated kill.
    """

    def __init__(self, oracle_gold, counter, *, raise_after=None, cost_per_call: float = 0.01):
        self._gold = oracle_gold or {}
        self._counter = counter
        self._raise_after = raise_after
        self._cost = cost_per_call
        self.last_cost_usd = 0.0
        self.last_usage: dict | None = None

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        if self._raise_after is not None and self._counter["calls"] >= self._raise_after:
            raise _RecordingKilledError(f"container died after {self._raise_after} paid calls")
        self._counter["calls"] += 1
        self._counter["asked"].append((head, relation))
        self.last_cost_usd = self._cost
        self.last_usage = {"prompt_tokens": 4, "completion_tokens": 2}
        return TeacherResponse(answers=sorted(self._gold.get(f"{head}\t{relation}", ())))


def _install_counting_teacher(monkeypatch, counter: dict, *, raise_after=None) -> None:
    def factory(  # noqa: ANN001
        settings, model, nl_template, oracle_gold=None, error_rate=0.0, seed=0, response_format=None
    ):
        return _CountingTeacher(oracle_gold, counter, raise_after=raise_after)

    monkeypatch.setattr(rkc, "_new_metered", factory)


def _fresh_counter() -> dict:
    return {"calls": 0, "asked": []}


def _record_no_timing(report: dict) -> dict:
    """A deep copy of a report with per-arm wall-clock times dropped (non-semantic)."""
    stripped = json.loads(json.dumps(report))
    for arm in stripped["arms"]:
        arm.pop("wallclock_s", None)
    return stripped


# ------------------------------------------------------- 1. a killed run loses no paid call
def test_killed_run_persists_every_paid_call(tmp_path, monkeypatch) -> None:
    bench, settings = _tiny_bench(), _oracle_settings()

    # Learn the distinct-pair count from a clean reference run, then kill halfway.
    ref_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, ref_counter)
    ref = run_controlled(answers_path=str(tmp_path / "ref.json"), **_common(bench, settings))
    n_distinct = ref["distinct_queries"]
    k = n_distinct // 2
    assert 0 < k < n_distinct

    path = tmp_path / "answers.json"
    kill_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, kill_counter, raise_after=k)
    with pytest.raises(_RecordingKilledError):
        run_controlled(answers_path=str(path), **_common(bench, settings))

    # The run never finished, so the canonical record does NOT exist ...
    assert not path.exists()
    # ... but the durable sidecar holds exactly the k paid answers, all readable.
    partial = rkc._AnswerLog(rkc._partial_log_path(str(path)))
    assert partial.exists()
    header, rows = partial.read()
    assert header["schema"] == rkc.ANSWERS_RECORD_SCHEMA
    assert len(rows) == k == kill_counter["calls"]
    for row in rows:
        assert isinstance(row["answers"], list)
        assert row["cost_usd"] == 0.01


# --------------------------------------------------------------- 2. resume re-bills nothing
def test_resume_rebills_nothing_and_completes_the_record(tmp_path, monkeypatch) -> None:
    bench, settings = _tiny_bench(), _oracle_settings()

    ref_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, ref_counter)
    ref = run_controlled(answers_path=str(tmp_path / "ref.json"), **_common(bench, settings))
    n_distinct = ref["distinct_queries"]
    ref_record = json.loads((tmp_path / "ref.json").read_text(encoding="utf-8"))
    k = n_distinct // 2
    assert 0 < k < n_distinct

    # Kill after k, then read the pairs the sidecar already holds (the warm set).
    path = tmp_path / "answers.json"
    kill_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, kill_counter, raise_after=k)
    with pytest.raises(_RecordingKilledError):
        run_controlled(answers_path=str(path), **_common(bench, settings))
    _, warm_rows = rkc._AnswerLog(rkc._partial_log_path(str(path))).read()
    warm_pairs = {(r["head"], r["relation"]) for r in warm_rows}
    assert len(warm_pairs) == k

    # Resume with a FRESH counting teacher.
    resume_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, resume_counter)
    res = run_controlled(answers_path=str(path), **_common(bench, settings))

    # THE money assertion: the teacher is called exactly for the still-missing pairs, and
    # never for a pair already recorded.
    assert resume_counter["calls"] == n_distinct - k
    assert set(resume_counter["asked"]).isdisjoint(warm_pairs)
    assert len(set(resume_counter["asked"])) == n_distinct - k

    # The completed record equals the uninterrupted one (same pairs, answers, costs) ...
    assert path.exists()
    resumed_record = json.loads(path.read_text(encoding="utf-8"))
    assert resumed_record == ref_record
    # ... the reported call total counts the whole record, and the sidecar is gone.
    assert res["real_teacher_calls"] == n_distinct
    assert not rkc._partial_log_path(str(path)).exists()


# --------------------------------------------- 3. uninterrupted == interrupted-then-resumed
def test_uninterrupted_equals_interrupted_then_resumed(tmp_path, monkeypatch) -> None:
    bench, settings = _tiny_bench(), _oracle_settings()

    # A) one clean uninterrupted recording.
    clean_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, clean_counter)
    clean = run_controlled(answers_path=str(tmp_path / "clean.json"), **_common(bench, settings))
    clean_record = json.loads((tmp_path / "clean.json").read_text(encoding="utf-8"))
    n_distinct = clean["distinct_queries"]
    k = n_distinct // 2

    # B) the same recording, killed after k then resumed.
    path = tmp_path / "resumed.json"
    kill_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, kill_counter, raise_after=k)
    with pytest.raises(_RecordingKilledError):
        run_controlled(answers_path=str(path), **_common(bench, settings))
    resume_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, resume_counter)
    resumed = run_controlled(answers_path=str(path), **_common(bench, settings))
    resumed_record = json.loads(path.read_text(encoding="utf-8"))

    # Identical FINAL RECORDS (timing-free by construction).
    assert resumed_record == clean_record
    # Identical REPORTS modulo per-arm wall-clock (a non-semantic nuisance field).
    assert _record_no_timing(resumed) == _record_no_timing(clean)

    # Spend semantics asserted EXPLICITLY (item 1b): the resumed run's measured spend is
    # the total cost of ALL answers, equal to the uninterrupted run's, NOT only the
    # n_distinct - k pairs it personally billed.
    assert resumed["total_measured_spend_usd"] == clean["total_measured_spend_usd"]
    assert resumed["real_teacher_calls"] == clean["real_teacher_calls"] == n_distinct
    # The record self-documents which meaning the number carries.
    assert resumed_record["spend_semantics"] == rkc.SPEND_SEMANTICS


# ------------------------------------------------------ 4. a crash mid-write is survivable
def test_crash_mid_write_drops_the_damaged_tail_only(tmp_path, monkeypatch) -> None:
    bench, settings = _tiny_bench(), _oracle_settings()

    clean_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, clean_counter)
    clean = run_controlled(answers_path=str(tmp_path / "clean.json"), **_common(bench, settings))
    clean_record = json.loads((tmp_path / "clean.json").read_text(encoding="utf-8"))
    n_distinct = clean["distinct_queries"]
    k = n_distinct // 2

    # Kill after k -> a clean k-line sidecar.
    path = tmp_path / "answers.json"
    kill_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, kill_counter, raise_after=k)
    with pytest.raises(_RecordingKilledError):
        run_controlled(answers_path=str(path), **_common(bench, settings))
    partial_path = rkc._partial_log_path(str(path))

    # Simulate a crash MID-WRITE of the (k+1)-th answer: a truncated, newline-less partial
    # JSON line appended after the k durable ones.
    with partial_path.open("a", encoding="utf-8") as fh:
        fh.write('{"head": "Mx", "relation": "directed_by", "answers": ["D')

    # The tolerant loader keeps exactly the k intact answers and drops the damaged tail;
    # the half-written pair is NOT accepted as an answer.
    header, rows = rkc._AnswerLog(partial_path).read()
    assert header["schema"] == rkc.ANSWERS_RECORD_SCHEMA
    assert len(rows) == k
    assert ("Mx", "directed_by") not in {(r["head"], r["relation"]) for r in rows}

    # Resume: re-buys ONLY the pairs the intact prefix lacks, and the completed record is
    # byte-identical to the uninterrupted one (no corruption leaked in).
    resume_counter = _fresh_counter()
    _install_counting_teacher(monkeypatch, resume_counter)
    run_controlled(answers_path=str(path), **_common(bench, settings))
    assert resume_counter["calls"] == n_distinct - k
    assert json.loads(path.read_text(encoding="utf-8")) == clean_record


# --------------------------------- 4b. corruption in the MIDDLE is loud, never silently skipped
def test_corrupt_middle_line_is_loud_not_silently_skipped(tmp_path) -> None:
    log = rkc._AnswerLog(rkc._partial_log_path(str(tmp_path / "answers.json")))
    log.start({"schema": rkc.ANSWERS_RECORD_SCHEMA, "provenance": {"seed": 0}})
    log.append("M0", "directed_by", ["D0"], 0.01, None)
    # A corrupt but newline-terminated line BEFORE a valid one is mid-file corruption,
    # which must fail loudly rather than be dropped like a truncated tail.
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write("NOT-JSON\n")
    log.append("M1", "directed_by", ["D1"], 0.01, None)
    with pytest.raises(rkc.AnswerRecordError):
        log.read()


if __name__ == "__main__":
    import unittest

    unittest.main()
