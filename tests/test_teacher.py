import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tacet.core.graph import WorldGraph
from tacet.llm.teacher import CallableTeacher, Narrator, OracleTeacher


def test_oracle_teacher_perfect():
    def oracle(h, r):
        if h == "Alice" and r == "lives_in":
            return ["London"]
        return []

    teacher = OracleTeacher(oracle=oracle)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["London"]
    assert resp.cost == 1.0
    assert resp.correct is True


def test_oracle_teacher_noisy():
    def oracle(h, r):
        return ["London"]

    # error_rate=1.0 means it should always pick from the pool
    teacher = OracleTeacher(oracle=oracle, error_rate=1.0, entity_pool=["Paris"], seed=42)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["Paris"]
    assert resp.correct is False


def test_oracle_teacher_noisy_correct_by_chance():
    def oracle(h, r):
        return ["London"]

    # If the randomly chosen entity happens to be in the truth, correct should be True
    teacher = OracleTeacher(oracle=oracle, error_rate=1.0, entity_pool=["London"], seed=42)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["London"]
    assert resp.correct is True


def test_oracle_teacher_reproducibility():
    def oracle(h, r):
        return ["London"]

    pool = ["Paris", "Berlin", "Tokyo", "New York"]
    # With a fixed seed and error_rate, it should be deterministic
    t1 = OracleTeacher(oracle=oracle, error_rate=0.5, entity_pool=pool, seed=123)
    t2 = OracleTeacher(oracle=oracle, error_rate=0.5, entity_pool=pool, seed=123)

    graph = WorldGraph()
    results1 = [t1.answer(graph, "A", "R").answers for _ in range(10)]
    results2 = [t2.answer(graph, "A", "R").answers for _ in range(10)]

    assert results1 == results2


def test_per_key_answer_is_call_order_invariant():
    # This is the regression test that would have caught the bug. Two arms share
    # an identical teacher (same seed / error_rate / pool) but call it a different
    # number of times and in a different order -- exactly the real-KG amortization
    # setup, where LLM-only asks every key, the cache arm asks only on misses, and
    # the full arm asks even fewer. Under noise_mode="per_key" a key's answer is a
    # pure function of (seed, head, relation), so arm B (a subset asker) must see
    # the same answer arm A saw for every shared key. Under "sequential" it does
    # not, because the shared RNG has been advanced a different number of times --
    # that is the bug, pinned here so it cannot silently return as the default.
    def oracle(h, r):
        return [f"truth::{h}"]

    pool = ["W0", "W1", "W2", "W3", "W4"]
    keys = [(f"h{i}", "rel") for i in range(12)]
    graph = WorldGraph()

    def run_arm(subset, mode):
        t = OracleTeacher(oracle=oracle, error_rate=0.5, entity_pool=pool, seed=7, noise_mode=mode)
        return {k: t.answer(graph, k[0], k[1]).answers for k in keys if k in subset}

    all_keys = set(keys)
    every_other = set(keys[::2])  # arm B asks only every other key

    # per_key: arm B's answer equals arm A's answer for every shared key.
    a_pk = run_arm(all_keys, "per_key")
    b_pk = run_arm(every_other, "per_key")
    for k in every_other:
        assert b_pk[k] == a_pk[k], f"per_key diverged for {k}"

    # sequential (the bug): at least one shared key diverges between the arms.
    a_seq = run_arm(all_keys, "sequential")
    b_seq = run_arm(every_other, "sequential")
    assert any(b_seq[k] != a_seq[k] for k in every_other), (
        "sequential mode was expected to diverge between arms that call the "
        "teacher a different number of times, but it did not -- the fixture no "
        "longer exercises the bug"
    )


def test_default_noise_mode_is_sequential_and_values_unchanged():
    # The published synthetic-grid numbers ride on the sequential default, so its
    # per-call outputs must not move. Values pinned from the pre-change code.
    def oracle(h, r):
        return ["London"]

    pool = ["Paris", "Berlin", "Tokyo", "New York"]

    default_mode = inspect.signature(OracleTeacher.__init__).parameters["noise_mode"].default
    assert default_mode == "sequential"

    t = OracleTeacher(oracle=oracle, error_rate=0.5, entity_pool=pool, seed=123)
    graph = WorldGraph()
    got = [t.answer(graph, "A", "R").answers for _ in range(10)]
    assert got == [
        ["Paris"],
        ["London"],
        ["Paris"],
        ["Tokyo"],
        ["Paris"],
        ["Tokyo"],
        ["London"],
        ["London"],
        ["New York"],
        ["London"],
    ]


def test_invalid_noise_mode_raises():
    with pytest.raises(ValueError):
        OracleTeacher(oracle=lambda h, r: [], noise_mode="bogus")


def test_error_rate_extremes_in_both_modes():
    def oracle(h, r):
        return ["London"]

    pool = ["Paris", "Berlin"]
    graph = WorldGraph()
    for mode in ("sequential", "per_key"):
        # error_rate=0.0 returns the exact truth, no matter the mode.
        perfect = OracleTeacher(
            oracle=oracle, error_rate=0.0, entity_pool=pool, seed=1, noise_mode=mode
        )
        for h in ("A", "B", "C", "D", "E"):
            resp = perfect.answer(graph, h, "R")
            assert resp.answers == ["London"]
            assert resp.correct is True

        # error_rate=1.0 always corrupts, no matter the mode.
        always = OracleTeacher(
            oracle=oracle, error_rate=1.0, entity_pool=pool, seed=1, noise_mode=mode
        )
        for h in ("A", "B", "C", "D", "E"):
            resp = always.answer(graph, h, "R")
            assert resp.answers != ["London"]
            assert resp.answers[0] in pool


def test_per_key_is_process_stable():
    # per_key must NOT depend on PYTHONHASHSEED. It is derived from a blake2b
    # digest of the joined key, never from builtin hash() of a str/tuple (which
    # is randomised per process). Prove it two ways: hard-coded expected values,
    # and two subprocesses run under different PYTHONHASHSEED that must agree.
    def oracle(h, r):
        return ["truth"]

    pool = ["W0", "W1", "W2", "W3"]
    key_heads = ["alpha", "beta", "gamma", "delta", "epsilon"]
    graph = WorldGraph()

    t = OracleTeacher(
        oracle=oracle, error_rate=0.5, entity_pool=pool, seed=99, noise_mode="per_key"
    )
    got = [t.answer(graph, h, "rel").answers[0] for h in key_heads]
    expected = ["truth", "truth", "W3", "W2", "truth"]
    assert got == expected

    snippet = (
        "from tacet.llm.teacher import OracleTeacher\n"
        "from tacet.core.graph import WorldGraph\n"
        "t = OracleTeacher(oracle=lambda h, r: ['truth'], error_rate=0.5, "
        "entity_pool=['W0','W1','W2','W3'], seed=99, noise_mode='per_key')\n"
        "g = WorldGraph()\n"
        "heads = ['alpha','beta','gamma','delta','epsilon']\n"
        "print(','.join(t.answer(g, h, 'rel').answers[0] for h in heads))\n"
    )
    src_dir = Path(__file__).resolve().parent.parent / "src"

    def run(hashseed):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(src_dir)}
        proc = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    r1 = run("1")
    r2 = run("2")
    assert r1 == r2 == ",".join(got)


def test_callable_teacher():
    def my_llm(h, r):
        return ["Response from LLM"]

    teacher = CallableTeacher(fn=my_llm, cost=5.0)
    graph = WorldGraph()
    resp = teacher.answer(graph, "Alice", "lives_in")

    assert resp.answers == ["Response from LLM"]
    assert resp.cost == 5.0
    assert resp.correct is True


def test_narrator_render():
    narrator = Narrator()

    # Test multiple answers
    res1 = narrator.render("Alice", "lives_in", ["London", "Paris"], tier=3)
    assert "Alice — lives in → London, Paris" in res1
    assert "[via expert reasoning]" in res1

    # Test no answers
    res2 = narrator.render("Alice", "lives_in", [], tier=2)
    assert "No answer found for (Alice, lives in)." in res2

    # Test tier 1 with proof
    proof = ["Step 1: ...", "Step 2: ..."]
    res3 = narrator.render("Alice", "lives_in", ["London"], tier=1, proof=proof)
    assert "Alice — lives in → London" in res3
    assert "[via verified rules]" in res3
    assert "proof:" in res3
    assert "Step 1:" in res3
    assert "Step 2:" in res3
