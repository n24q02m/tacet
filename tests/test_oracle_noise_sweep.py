"""Tests for the E11 oracle-noise sweep and the oracle mode of the controlled runner.

All fixtures are TINY and SYNTHETIC (tens of triples): MetaQA is never loaded or
run here. The heavy MetaQA sweep runs later on Modal by importing
``run_oracle_noise_sweep.sweep`` — these tests pin the mechanism, the sharing
contract under noise, the pre-registered verdict, and bootstrap reproducibility.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_oracle_noise_sweep import (  # noqa: E402
    aggregate_error_rate,
    bootstrap_ci,
    classify_verdict,
    cliff_teacher_accuracy,
    sweep,
    sweep_2d,
    wilson_ci,
)
from run_real_kg_amortization import BudgetGuard, _new_metered  # noqa: E402
from run_real_kg_controlled import (  # noqa: E402
    ReplayTeacher,
    SharedAnswerCache,
    _cell_valid,
    run_controlled,
)

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark, MetaQAQuestion  # noqa: E402


# --------------------------------------------------------------- synthetic bench
def _tiny_bench() -> MetaQABenchmark:
    """A few movies with directors + actors, plus never-queried background edges.

    ``directed_by`` / ``starred_actors`` are the only queried relations (their NL
    questions resolve through ``run_metaqa._relation_for_question``); the
    ``has_genre`` edges are never queried, so the held-out graph is never empty
    and the KGE warm-up always has triples to fit.
    """
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
    entities = set(kg.entities())
    relations = kg.relations()
    return MetaQABenchmark(
        name="tiny-movies",
        hop=1,
        split="test",
        kg=kg,
        questions=questions,
        entities=entities,
        relations=relations,
    )


def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


def _latent_composition_bench() -> MetaQABenchmark:
    """A KG with a latent length-1 rule whose confidence is EXACTLY 0.5.

    ``directed_by`` is the only queried relation (gold: ``Mi -> Di``). A background
    ``made_by`` relation carries, for every movie, TWO tails: the true director
    ``Di`` (matching ``directed_by``) and a shared decoy ``W`` (never a director).
    So over any set of teacher-answered heads the rule ``directed_by <= made_by``
    has confidence ``k / 2k = 0.5`` exactly, regardless of which/how many heads the
    Zipfian stream covers. That makes the install contrast deterministic: a LOW
    gamma (0.5) installs it (``0.5 >= 0.5``), a HIGH gamma (0.99) does not — and no
    other body reaches even 0.5 (checked in the miner search space). ``has_genre``
    keeps the held-out graph and KGE warm-up non-empty; 24 movies clear the miner's
    synthesis trigger (10 complete heads) under a broad Zipfian stream.
    """
    kg = WorldGraph(name="latent-comp")
    questions: list[MetaQAQuestion] = []
    for i in range(24):
        m, d = f"M{i}", f"D{i}"
        kg.add_edge(m, "directed_by", d)  # queried target; gold Mi -> Di
        kg.add_edge(m, "made_by", d)  # correct body edge (matches directed_by)
        kg.add_edge(m, "made_by", "W")  # decoy body edge -> confidence 0.5
        kg.add_edge(m, "has_genre", "drama")  # background: never queried
        questions.append(
            MetaQAQuestion(question=f"who directed [{m}]?", head=m, answers=[d], hop=1)
        )
    entities = set(kg.entities())
    relations = kg.relations()
    return MetaQABenchmark(
        name="latent-comp",
        hop=1,
        split="test",
        kg=kg,
        questions=questions,
        entities=entities,
        relations=relations,
    )


# --------------------------------------------------- controlled runner: oracle mode
def test_controlled_oracle_perfect_is_matched_and_free():
    """error_rate=0.0 on a tiny synthetic KG: all arms see the identical (perfect)
    answers, accuracy is matched, measured USD is 0, and the rule arm makes no more
    teacher calls than the cache arm.
    """
    report = run_controlled(
        hop=1,
        split="test",
        limit=24,
        zipf_a=1.5,
        seed=0,
        oracle_error_rate=0.0,
        settings=_oracle_settings(),
        bench=_tiny_bench(),
        verbose=False,
    )

    # the run is unambiguously an oracle run, recorded so it cannot be misread
    assert report["teacher_kind"] == "oracle"
    assert report["oracle_error_rate"] == 0.0
    assert report["noise_mode"] == "per_key"
    assert report["real_llm"] is False
    assert report["tier2_disabled"] is True
    assert report["shared_teacher_answers"] is True

    # free: no provider usage -> $0 across every arm
    assert report["total_measured_spend_usd"] == 0.0
    assert all(a["total_cost_usd"] == 0.0 for a in report["arms"])

    v = report["verdict"]
    assert v["accuracy_matched"] is True
    assert v["full_teacher_calls"] <= v["cache_teacher_calls"]

    # a perfect oracle: every arm is fully correct and the teacher's own accuracy is 1
    accs = {a["arm"]: a["accuracy"] for a in report["arms"]}
    assert accs["llm_only"] == accs["cache_cascade"] == accs["full_distillation"] == 1.0
    assert report["teacher_answer_accuracy"] == 1.0


def test_controlled_report_keeps_all_original_keys_plus_new_ones():
    """The refactor into ``run_controlled`` must keep the report schema stable
    (same keys the CLI wrote before) and only ADD the oracle-provenance keys.
    """
    report = run_controlled(
        hop=1,
        split="test",
        limit=16,
        seed=1,
        oracle_error_rate=0.0,
        settings=_oracle_settings(),
        bench=_tiny_bench(),
        verbose=False,
    )
    original_keys = {
        "dataset",
        "hop",
        "design",
        "tier2_disabled",
        "shared_teacher_answers",
        "kg_stats",
        "real_llm",
        "teacher_model_called",
        "priced_as_model",
        "composed_relation",
        "workload_cap",
        "zipf_a",
        "seed",
        "stream_len",
        "distinct_queries",
        "real_teacher_calls",
        "truncated_by_budget",
        "total_measured_spend_usd",
        "arms",
        "verdict",
    }
    new_keys = {
        "teacher_kind",
        "oracle_error_rate",
        "noise_mode",
        "teacher_answer_accuracy",
        "teacher_answers_correct",
        "teacher_answers_total",
    }
    assert original_keys <= set(report)
    assert new_keys <= set(report)


# ------------------------------------------------ sharing preserved under noise
def test_sharing_preserved_under_noise():
    """SharedAnswerCache contract at error_rate=0.5: every arm gets the SAME answer
    for the same (head, relation), one real oracle call per distinct pair, AND the
    noise actually fires (else the fixture would not exercise the corruption path).
    """
    settings = SimpleNamespace(teacher="oracle")
    gold = {f"h{i}\trel": frozenset({f"t{i}a", f"t{i}b"}) for i in range(24)}
    metered = _new_metered(settings, "grok-4.3", None, gold, error_rate=0.5, seed=3)
    guard = BudgetGuard(budget_usd=1e9)
    kg = WorldGraph()
    shared = SharedAnswerCache(metered, guard, kg)

    arm_a, arm_b, arm_c = (ReplayTeacher(shared) for _ in range(3))
    corrupted = 0
    for i in range(24):
        h, r = f"h{i}", "rel"
        a = arm_a.answer(kg, h, r).answers
        b = arm_b.answer(kg, h, r).answers
        c = arm_c.answer(kg, h, r).answers
        assert a == b == c, f"arms diverged for {h} — SharedAnswerCache contract broken"
        if set(a) != set(gold[f"{h}\trel"]):
            corrupted += 1

    assert metered.n_calls == 24  # exactly one real oracle call per distinct pair
    assert shared.real_calls == 24
    assert corrupted > 0, "error_rate=0.5 must corrupt at least one answer"


# ---------------------------------------------------- pre-registered verdict rule
def test_verdict_boundaries():
    # POSITIVE: mean >= 20 AND CI excludes 0
    assert classify_verdict(30.0, 10.0, 50.0) == "POSITIVE"
    # mean 19.9 with a clean (0-excluding) CI is NOT positive (sub-threshold)
    assert classify_verdict(19.9, 15.0, 24.0) != "POSITIVE"
    assert classify_verdict(19.9, 15.0, 24.0) == "INCONCLUSIVE"
    # mean 20.1 but the CI contains 0 -> NEUTRAL (CI-contains-0 wins over the mean)
    assert classify_verdict(20.1, -5.0, 45.0) == "NEUTRAL"
    # NEGATIVE: mean < 0 AND CI excludes 0
    assert classify_verdict(-15.0, -30.0, -2.0) == "NEGATIVE"
    # NEUTRAL: CI straddles 0
    assert classify_verdict(4.0, -3.0, 11.0) == "NEUTRAL"


def test_aggregate_error_rate_classifies_arrays():
    rng = np.random.default_rng(0)
    # tight, well-above-threshold savings -> POSITIVE
    pos = aggregate_error_rate([25.0, 30.0, 35.0], rng)
    assert pos["verdict"] == "POSITIVE"
    assert pos["ci_excludes_zero"] is True

    # savings straddling 0 -> CI contains 0 -> NEUTRAL
    neu = aggregate_error_rate([-20.0, 5.0, 25.0], np.random.default_rng(0))
    assert neu["verdict"] == "NEUTRAL"
    assert neu["ci_excludes_zero"] is False


# --------------------------------------------------------- bootstrap reproducible
def test_bootstrap_ci_reproducible():
    vals = [25.0, 30.0, 12.0, 40.0, -5.0]
    ci1 = bootstrap_ci(vals, np.random.default_rng(12345))
    ci2 = bootstrap_ci(vals, np.random.default_rng(12345))
    assert ci1 == ci2  # same inputs + same rng seed -> identical CI

    # the aggregate wrapper is reproducible the same way
    a1 = aggregate_error_rate(vals, np.random.default_rng(777))
    a2 = aggregate_error_rate(vals, np.random.default_rng(777))
    assert a1 == a2


# ----------------------------------------------------- Wilson score interval (E11)
def test_wilson_ci_known_values():
    """Wilson interval at the endpoints and the symmetric case (n=20, z=1.96)."""
    lo0, hi0 = wilson_ci(0, 20)
    assert lo0 == 0.0  # no successes -> lower bound pinned at 0
    assert 0.0 < hi0 < 0.2  # upper bound strictly inside (0, 0.2)

    lo1, hi1 = wilson_ci(20, 20)
    assert hi1 == 1.0  # all successes -> upper bound pinned at 1
    assert 0.0 < lo1 < 1.0

    loh, hih = wilson_ci(10, 20)
    assert loh < 0.5 < hih  # symmetric case brackets 0.5


def test_wilson_ci_bounds_never_escape_unit_interval():
    """Bounds stay within [0, 1] across the full grid, incl. the degenerate n=0."""
    # n == 0 contract: no evidence -> the point interval (0.0, 0.0), never raises.
    assert wilson_ci(0, 0) == (0.0, 0.0)
    for n in range(0, 25):
        for s in range(0, n + 1):
            lo, hi = wilson_ci(s, n)
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0
            assert lo <= hi


# ----------------------------------------------- one-sided validity vs. matched
def test_one_sided_validity_distinct_from_accuracy_matched():
    """A cell where the rule arm is MORE accurate is VALID (one-sided) even though
    ``accuracy_matched`` (the symmetric diagnostic) is False; a cell where the rule
    arm is LESS accurate is INVALID.
    """
    # full > cache: cell_valid True, but accuracy_matched (|full-cache|<1e-9) False.
    assert _cell_valid(0.9933, 0.8933) is True
    assert (abs(0.9933 - 0.8933) < 1e-9) is False
    # exactly equal: valid (and matched).
    assert _cell_valid(0.90, 0.90) is True
    # full < cache: the rule made things worse -> invalid.
    assert _cell_valid(0.80, 0.90) is False


# ------------------------------------------------ gamma threading (Item 1 + Item 5)
def test_gamma_threading_high_installs_nothing_low_installs_rule():
    """On a synthetic KG with a latent 0.5-confidence rule, a HIGH gamma installs
    nothing while a LOW gamma installs the rule — everything else fixed — and the
    report's ``gamma`` field round-trips.
    """
    bench = _latent_composition_bench()
    settings = _oracle_settings()
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
    high = run_controlled(gamma=0.99, **common)
    low = run_controlled(gamma=0.5, **common)

    # the gamma field round-trips into the report
    assert high["gamma"] == 0.99
    assert low["gamma"] == 0.5

    # high gamma installs NOTHING; low gamma installs a rule
    assert high["verdict"]["rule_installed"] is False
    assert high["verdict"]["synthesised_rules"] == []
    assert low["verdict"]["rule_installed"] is True
    assert len(low["verdict"]["synthesised_rules"]) >= 1

    # the installed rule's world precision is scored (0.5: half the body is the decoy)
    assert low["verdict"]["rule_world_precision"] is not None
    assert high["verdict"]["rule_world_precision"] is None


# ------------------------------------------------- 2-D sweep == 1-D sweep (Item 4)
def test_sweep_2d_single_gamma_matches_1d_sweep():
    """``sweep_2d`` with gammas=[0.95] produces the SAME cell values as ``sweep``."""
    bench = _tiny_bench()
    settings = _oracle_settings()
    kw = dict(
        hop=1,
        seeds=[0, 1],
        error_rates=[0.0, 0.5],
        limit=12,
        zipf_a=1.5,
        split="test",
        bench=bench,
        settings=settings,
    )
    one_d = sweep(**kw)
    two_d = sweep_2d(gammas=[0.95], **kw)
    assert one_d["cells"] == two_d["cells"]


# ---------------------------------------------------- cliff interpolation (Item 5)
def test_cliff_teacher_accuracy_interpolation():
    """p_rule_installed = [1.0, 1.0, 0.4] against known accuracies: the crossing of
    0.5 is linearly interpolated between the last two points.
    """
    accs = [1.0, 0.9, 0.8]  # teacher accuracy, ordered by error_rate ascending
    ps = [1.0, 1.0, 0.4]
    expected = 0.9 + (0.5 - 1.0) / (0.4 - 1.0) * (0.8 - 0.9)
    got = cliff_teacher_accuracy(accs, ps)
    assert got == expected


def test_cliff_teacher_accuracy_none_when_no_crossing():
    """When p_rule_installed never dips through 0.5, there is no cliff."""
    accs = [1.0, 0.9, 0.8]
    ps = [1.0, 1.0, 0.9]  # stays above 0.5 the whole way
    assert cliff_teacher_accuracy(accs, ps) is None
    # too few usable points -> None
    assert cliff_teacher_accuracy([0.9], [0.6]) is None


if __name__ == "__main__":
    import unittest

    unittest.main()
