"""Tests for the E11 two-axis teacher-quality ladder analysis.

All fixtures are TINY and HAND-BUILT: the real on-disk artifacts
(``oracle_gamma_sweep_hop2.json`` / ``real_ladder_hop2.json``) are NEVER read
here. These tests pin the two-axis aggregation, the H2 cliff on either ruler, and
the localisation delta (real teacher vs. the oracle i.i.d.-noise curve at matched
accuracy) — the sign of that delta is the study's headline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from analyze_teacher_ladder import (  # noqa: E402
    aggregate_oracle,
    cliff,
    locate_on_curve,
)


# ------------------------------------------------------------- aggregate_oracle
def _cell(gamma, error_rate, cache_acc, sub_gold, installed, saved, valid, rwp):
    """One synthetic oracle cell in the on-disk schema (only the read fields)."""
    return {
        "gamma": gamma,
        "error_rate": error_rate,
        "cache_accuracy": cache_acc,
        "teacher_answer_accuracy": sub_gold,
        "synthesised_rules": ["syn:r"] if installed else [],
        "calls_saved_pct": saved,
        "cell_valid": valid,
        "rule_world_precision": rwp,
    }


def test_aggregate_oracle_both_axes_p_install_and_invalid():
    """Both accuracy means, p_install, its Wilson bracket, the invalid count and the
    None-skipping rule-world-precision mean are all computed per (gamma, error_rate).
    """
    cells = [
        # group A: (0.5, 0.0) — both installed, both valid
        _cell(0.5, 0.0, 1.0, 1.0, True, 80.0, True, 1.0),
        _cell(0.5, 0.0, 0.8, 0.9, True, 60.0, True, 0.5),
        # group B: (0.5, 0.1) — one install, one invalid cell with None precision
        _cell(0.5, 0.1, 0.6, 0.5, False, 0.0, False, None),
        _cell(0.5, 0.1, 0.4, 0.3, True, 50.0, True, 0.9),
    ]
    out = aggregate_oracle(cells)
    by_key = {(a["gamma"], a["error_rate"]): a for a in out}

    a = by_key[(0.5, 0.0)]
    assert a["mean_arm_accuracy"] == pytest.approx(0.9)
    assert a["mean_sub_gold_accuracy"] == pytest.approx(0.95)
    assert a["p_install"] == pytest.approx(1.0)
    assert a["mean_calls_saved_pct"] == pytest.approx(70.0)
    assert a["mean_rule_world_precision"] == pytest.approx(0.75)
    assert a["n_seeds"] == 2
    assert a["n_invalid"] == 0
    lo, hi = a["p_install_wilson95"]
    assert lo <= a["p_install"] <= hi  # Wilson interval brackets the point estimate
    assert 0.0 <= lo <= hi <= 1.0

    b = by_key[(0.5, 0.1)]
    assert b["mean_arm_accuracy"] == pytest.approx(0.5)
    assert b["mean_sub_gold_accuracy"] == pytest.approx(0.4)
    assert b["p_install"] == pytest.approx(0.5)
    assert b["n_invalid"] == 1  # the cell_valid is False cell is counted
    # rule_world_precision mean skips the None cell -> only the 0.9 cell contributes
    assert b["mean_rule_world_precision"] == pytest.approx(0.9)
    lo_b, hi_b = b["p_install_wilson95"]
    assert lo_b <= 0.5 <= hi_b


# --------------------------------------------------------------------- cliff
def _agg(gamma, error_rate, arm, sub, p):
    """One synthetic aggregate row in the aggregate_oracle output schema."""
    return {
        "gamma": gamma,
        "error_rate": error_rate,
        "mean_arm_accuracy": arm,
        "mean_sub_gold_accuracy": sub,
        "p_install": p,
    }


def test_cliff_interpolates_on_each_axis_and_none_when_no_crossing():
    """p_install crosses 0.5 between two known points at gamma 0.7 (interpolated on
    BOTH rulers); gamma 0.99 never crosses -> None. Rows are shuffled to prove the
    per-gamma sort by error_rate.
    """
    aggregates = [
        # gamma 0.7 — crosses 0.5 between error_rate 0.1 (p=1.0) and 0.2 (p=0.0)
        _agg(0.7, 0.2, 0.7, 0.75, 0.0),
        _agg(0.7, 0.0, 0.9, 0.95, 1.0),
        _agg(0.7, 0.1, 0.8, 0.85, 1.0),
        # gamma 0.99 — p stays above 0.5 the whole way -> no cliff
        _agg(0.99, 0.0, 0.9, 0.95, 1.0),
        _agg(0.99, 0.1, 0.8, 0.85, 1.0),
        _agg(0.99, 0.2, 0.7, 0.75, 0.9),
    ]

    arm = {c["gamma"]: c for c in cliff(aggregates, "mean_arm_accuracy")}
    sub = {c["gamma"]: c for c in cliff(aggregates, "mean_sub_gold_accuracy")}

    # crossing at p=0.5 between (acc=0.8, p=1.0) and (acc=0.7, p=0.0): frac = 0.5
    assert arm[0.7]["cliff_accuracy"] == pytest.approx(0.8 + 0.5 * (0.7 - 0.8))  # 0.75
    assert sub[0.7]["cliff_accuracy"] == pytest.approx(0.85 + 0.5 * (0.75 - 0.85))  # 0.80
    assert arm[0.7]["axis"] == "mean_arm_accuracy"
    assert sub[0.7]["axis"] == "mean_sub_gold_accuracy"

    # never crosses -> None on both rulers
    assert arm[0.99]["cliff_accuracy"] is None
    assert sub[0.99]["cliff_accuracy"] is None


# ------------------------------------------------------------- locate_on_curve
def _real(slug, gamma, cache_acc, sub_gold, installed):
    """One synthetic real-ladder row (only the read fields)."""
    return {
        "slug": slug,
        "gamma": gamma,
        "cache_accuracy": cache_acc,
        "teacher_answer_accuracy": sub_gold,
        "synthesised_rules": ["syn:r"] if installed else [],
    }


def test_locate_on_curve_delta_sign_off_and_on_curve():
    """Off-curve: a real teacher whose accuracy MATCHES an oracle point but installs
    LESS -> delta < 0 with the exact magnitude. On-curve: a real teacher sitting on
    an interpolated oracle point -> delta ~ 0.
    """
    oracle = [
        _agg(0.5, 0.0, 1.0, 1.0, 1.0),
        _agg(0.5, 0.2, 0.8, 0.8, 0.8),
        _agg(0.5, 0.4, 0.6, 0.6, 0.4),
    ]
    real_rows = [
        # off-curve: arm acc == 0.8 (oracle p_install there = 0.8) but only 1/2 install
        _real("fake/below", 0.5, 0.8, 0.8, True),
        _real("fake/below", 0.5, 0.8, 0.8, False),
        # on-curve: arm acc == 0.7 -> oracle p_install interpolates to 0.6; 3/5 install
        _real("fake/oncurve", 0.5, 0.7, 0.7, True),
        _real("fake/oncurve", 0.5, 0.7, 0.7, True),
        _real("fake/oncurve", 0.5, 0.7, 0.7, True),
        _real("fake/oncurve", 0.5, 0.7, 0.7, False),
        _real("fake/oncurve", 0.5, 0.7, 0.7, False),
    ]
    out = {r["slug"]: r for r in locate_on_curve(oracle, real_rows, "mean_arm_accuracy")}

    below = out["fake/below"]
    assert below["real_accuracy"] == pytest.approx(0.8)
    assert below["real_p_install"] == pytest.approx(0.5)
    assert below["oracle_p_install_at_same_accuracy"] == pytest.approx(0.8)
    assert below["delta"] == pytest.approx(-0.3)  # installs less than i.i.d. noise
    assert below["delta"] < 0

    onc = out["fake/oncurve"]
    # oracle p_install at acc 0.7 interpolates between (0.6,0.4) and (0.8,0.8): 0.6
    assert onc["oracle_p_install_at_same_accuracy"] == pytest.approx(0.6)
    assert onc["real_p_install"] == pytest.approx(0.6)
    assert onc["delta"] == pytest.approx(0.0)


# --------------------------------------------------------------- determinism
def test_deterministic_same_inputs_same_output():
    """No wall-clock, no RNG: identical inputs -> byte-identical outputs."""
    cells = [
        _cell(0.5, 0.0, 1.0, 1.0, True, 80.0, True, 1.0),
        _cell(0.5, 0.0, 0.8, 0.9, True, 60.0, True, 0.5),
        _cell(0.9, 0.2, 0.6, 0.4, False, 0.0, True, None),
    ]
    assert aggregate_oracle(cells) == aggregate_oracle(cells)

    aggs = aggregate_oracle(cells)
    assert cliff(aggs, "mean_arm_accuracy") == cliff(aggs, "mean_arm_accuracy")

    oracle = [_agg(0.5, 0.0, 1.0, 1.0, 1.0), _agg(0.5, 0.4, 0.6, 0.6, 0.4)]
    real_rows = [_real("t", 0.5, 0.8, 0.8, True)]
    once = locate_on_curve(oracle, real_rows, "mean_arm_accuracy")
    twice = locate_on_curve(oracle, real_rows, "mean_arm_accuracy")
    assert once == twice


if __name__ == "__main__":
    import unittest

    unittest.main()
