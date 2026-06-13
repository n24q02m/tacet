"""Unit tests for the controlled-runner arms (ArmScore, compile_once, nl_strategy).

No network, no real teacher: a FakeShared cache + deterministic fake embedder +
scripted fake weak teacher drive every branch. These lock the fairness
invariants the experiment relies on (suffix scoring, frozen ruleset, weak-token
accounting) so a regression cannot silently turn a baseline into a strawman.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_privaci_controlled import (  # noqa: E402
    ArmScore,
    _run_compile_once,
    _run_nl_strategy,
)

from tacet.data.privaci import PrivaCICase  # noqa: E402
from tacet.data.privaci_graph import build_compliance_benchmark  # noqa: E402
from tacet.data.privaci_vocab import normalize_case  # noqa: E402
from tacet.llm.teacher import TeacherResponse  # noqa: E402

VOCAB = {
    "information_type": {
        "categories": ["credentials", "health_data", "other"],
        "aliases": {"passwords": "credentials", "health data": "health_data"},
    },
    "purpose": {
        "categories": ["storage", "care", "other"],
        "aliases": {"data storage": "storage", "medical care": "care"},
    },
    "sender_role": {"categories": ["controller", "other"], "aliases": {"controller": "controller"}},
    "recipient_role": {"categories": ["other"], "aliases": {}},
    "subject_role": {"categories": ["other"], "aliases": {}},
}


def _case(i, info, consent, verdict, arts):
    return PrivaCICase(
        case_id=f"C-{i:03d}",
        norm_type=verdict,
        sender=("acme",),
        sender_role=("controller",),
        recipient=("sys",),
        recipient_role=("x",),
        subject=("users",),
        subject_role=("y",),
        information_type=(info,),
        consent_form=consent,
        purpose=("data storage",),
        followed_articles=(),
        violated_articles=tuple(arts),
        case_content=f"case {i}: {info} consent={consent}",
    )


def _bench(cases):
    return build_compliance_benchmark(cases, VOCAB)


def _atoms(cases):
    from run_privaci_controlled import _case_atoms

    return {c.case_id: _case_atoms(normalize_case(c, VOCAB)) for c in cases}


class FakeShared:
    """Stands in for SharedComplianceCache: one (verdict, articles) per case_id."""

    def __init__(self, bench):
        self.bench = bench
        self.real_calls = 0
        self.calls: list[str] = []

    def answer(self, case_id):
        self.real_calls += 1
        self.calls.append(case_id)
        return self.bench.oracle[case_id], 0.01  # fixed per-case teacher cost


# --------------------------------------------------------------- ArmScore
def test_armscore_suffix_scoring_skips_prefix():
    oracle = {f"C-{i:03d}": ("prohibit", ("art32",)) for i in range(5)}
    s = ArmScore(oracle, score_from=2)
    for idx in range(5):
        s.record(f"C-{idx:03d}", ("prohibit", ("art32",)), usd=1.0, teacher_call=True, idx=idx)
    rep = s.report("x", 0.0)
    assert rep["n"] == 3  # only idx 2,3,4 scored
    assert rep["total_cost_usd"] == 3.0
    assert rep["teacher_calls"] == 3


# --------------------------------------------------------------- compile_once
def test_compile_once_freezes_and_scores_suffix():
    # 12 cases: credentials+none -> art32 (planted, recurs in both halves)
    cases = []
    for i in range(6):
        cases.append(_case(i, "passwords", "none", "prohibit", ["art32"]))
    for i in range(6, 12):
        cases.append(_case(i, "passwords", "none", "prohibit", ["art32"]))
    bench = _bench(cases)
    atoms = _atoms(cases)
    shared = FakeShared(bench)
    rep = _run_compile_once(bench, shared, atoms, prefix_k=6, min_support=3, min_confidence=0.9)
    # exactly the 6 prefix cases were escalated to the teacher during training
    assert all(cid in shared.calls for cid in bench.workload[:6])
    assert rep["prefix_k"] == 6
    assert rep["n_test"] == 6
    assert rep["n"] == 6  # scored on suffix only
    # a planted rule was mined and froze, so suffix cases hit the engine for free
    assert rep["engine_hits"] >= 1
    assert rep["total_cost_usd"] < 6 * 0.01  # not every suffix case paid the teacher
    # the engine-served verdicts are not just cheap, they are CORRECT: the planted
    # rule matches the oracle, so every scored suffix verdict is right.
    assert rep["verdict_acc"] == 1.0


def test_compile_once_with_no_rules_falls_back_to_teacher():
    # all-distinct, no repeated pattern reaching support -> no rule mined
    cases = [
        _case(i, "passwords" if i % 2 else "health data", "none", "prohibit", [f"art{i}"])
        for i in range(8)
    ]
    bench = _bench(cases)
    atoms = _atoms(cases)
    shared = FakeShared(bench)
    rep = _run_compile_once(bench, shared, atoms, prefix_k=4, min_support=3, min_confidence=0.9)
    assert rep["engine_hits"] == 0
    # every suffix case escalated to the teacher (frozen empty ruleset)
    assert rep["teacher_calls"] == 4


# --------------------------------------------------------------- nl_strategy
class FakeWeak:
    """Weak teacher that always returns a parseable verdict (one cost per call)."""

    def __init__(self):
        self.model = "fake-weak"
        self.last_cost_usd = 0.002
        self.n = 0

    def answer(self, graph, head, relation):
        self.n += 1
        return TeacherResponse(answers=["prohibit", "art32"], cost=self.last_cost_usd)


class FakeGuard:
    def __init__(self):
        self.spent = 0.0

    def add(self, usd):
        self.spent += usd


def _embed_groups(groups):
    """Embedder where same-group texts are identical vectors (cosine 1), else 0.

    ``groups`` maps a scenario substring -> a one-hot dimension; identical group
    -> cosine 1.0 (covered), different group -> orthogonal -> cosine 0.
    """

    def embed(texts):
        out = np.zeros((len(texts), len(groups) + 1), dtype=np.float32)
        for r, t in enumerate(texts):
            dim = next((d for sub, d in groups.items() if sub in str(t)), len(groups))
            out[r, dim] = 1.0
        return out

    return embed


def test_nl_strategy_cold_start_defers_then_covers():
    # two cases share scenario text (group A); the 2nd should be COVERED by the
    # 1st's appended strategy (cosine 1.0 >= tau) -> answered by the weak model.
    cases = [_case(0, "passwords", "none", "prohibit", ["art32"]) for _ in range(1)]
    cases += [_case(1, "passwords", "none", "prohibit", ["art32"])]
    # make both scenarios share the substring 'passwords' -> same embedding group
    bench = _bench(cases)
    shared = FakeShared(bench)
    weak = FakeWeak()
    embed = _embed_groups({"passwords": 0})
    rep = _run_nl_strategy(
        bench, shared, weak_metered=weak, embed=embed, guard=FakeGuard(), tau=0.5
    )
    # case 0: empty repo -> defer to teacher + append strategy; case 1: covered -> weak
    assert rep["defers"] == 1
    assert rep["accepts"] == 1
    assert shared.real_calls == 1
    assert rep["n_strategies"] == 1
    assert weak.n == 1  # weak called only on the covered case
    # cost = 1 teacher (defer) + 1 weak (accept)
    assert abs(rep["total_cost_usd"] - (0.01 + 0.002)) < 1e-9


def test_nl_strategy_all_distinct_always_defers():
    # every case is in its own embedding group -> never covered -> always defer
    cases = [_case(i, "passwords", "none", "prohibit", ["art32"]) for i in range(3)]
    bench = _bench(cases)
    shared = FakeShared(bench)
    weak = FakeWeak()
    # each case_content contains 'case {i}:' -> unique group per case
    embed = _embed_groups({"case 0:": 0, "case 1:": 1, "case 2:": 2})
    rep = _run_nl_strategy(
        bench, shared, weak_metered=weak, embed=embed, guard=FakeGuard(), tau=0.5
    )
    assert rep["defers"] == 3
    assert rep["accepts"] == 0
    assert weak.n == 0  # uncovered cases skip the weak call entirely
    assert shared.real_calls == 3
    assert abs(rep["total_cost_usd"] - 3 * 0.01) < 1e-9  # teacher only


# --------------------------------------------------------------- _engine_answer
def _engine_with(*facts):
    from tacet.core.graph import WorldGraph
    from tacet.core.symbolic import RuleEngine

    bench = _bench([_case(0, "passwords", "none", "prohibit", ["art32"])])
    g = WorldGraph()
    for h, r, t in facts:
        g.add_edge(h, r, t)
    eng = RuleEngine(bench.ontology)
    eng.materialise(g)  # closure includes the planted edges
    return eng


def test_engine_answer_serves_prohibit_from_violates():
    from run_privaci_controlled import _engine_answer

    eng = _engine_with(("X", "violates", "article:art32"))
    assert _engine_answer(eng, "X") == ("prohibit", ("art32",))


def test_engine_answer_serves_permit_from_verdict():
    from run_privaci_controlled import _engine_answer

    eng = _engine_with(("Y", "verdict", "verdict:permit"))
    assert _engine_answer(eng, "Y") == ("permit", ())


def test_engine_answer_abstains_on_violates_permit_conflict():
    # a case the closure marks BOTH violates and permit is a rule conflict;
    # _engine_answer must abstain (return None) and let the teacher arbitrate.
    from run_privaci_controlled import _engine_answer

    eng = _engine_with(("Z", "violates", "article:art32"), ("Z", "verdict", "verdict:permit"))
    assert _engine_answer(eng, "Z") is None


def test_engine_answer_abstains_when_silent():
    from run_privaci_controlled import _engine_answer

    eng = _engine_with()
    assert _engine_answer(eng, "nobody") is None
