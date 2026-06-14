"""Tests for the experiment harness aggregation (``tacet.eval.experiment``).

These pin the statistics that feed every reported error bar (the ``_summary``
helper) and smoke the job-building / aggregation pipeline, which were previously
untested.
"""

from __future__ import annotations

import math
import sys
import unittest
from unittest import mock

import pytest

# The harness's error bars are exact Student-t intervals, which require SciPy.
# Skip (rather than hard-fail) the numeric assertions when SciPy is absent.
pytest.importorskip("scipy")

from tacet.eval.experiment import _summary, _t_critical, aggregate, build_jobs  # noqa: E402


class TestSummary(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(_summary([]), {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0})

    def test_singleton_has_no_spread(self) -> None:
        s = _summary([3.0])
        self.assertEqual(s, {"mean": 3.0, "std": 0.0, "ci95": 0.0, "n": 1})

    def test_sample_std_and_t_interval(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = _summary(vals)
        self.assertAlmostEqual(s["mean"], 3.0)
        # sample standard deviation (ddof=1) of 1..5 is sqrt(2.5)
        self.assertAlmostEqual(s["std"], math.sqrt(2.5))
        sem = math.sqrt(2.5) / math.sqrt(5)
        # the interval is the Student-t half-width, not the 1.96 normal one
        self.assertAlmostEqual(s["ci95"], _t_critical(4) * sem)
        self.assertEqual(s["n"], 5)

    def test_t_critical_is_student_t_not_normal(self) -> None:
        self.assertEqual(_t_critical(0), 0.0)
        # df=7 (the 8-seed grid): t_{0.975,7} = 2.365, wider than 1.96
        self.assertAlmostEqual(_t_critical(7), 2.365, places=2)
        self.assertGreater(_t_critical(7), 1.96)

    def test_t_critical_raises_without_scipy_instead_of_silent_z(self) -> None:
        # A missing SciPy must be loud: silently returning the narrower normal
        # z=1.96 would misreport the paper's Student-t error bars.
        with (
            mock.patch.dict(sys.modules, {"scipy": None, "scipy.stats": None}),
            self.assertRaises(RuntimeError),
        ):
            _t_critical(7)


class TestBuildJobsAggregate(unittest.TestCase):
    def test_build_jobs_one_seed_smoke(self) -> None:
        jobs = build_jobs(seeds=1, fast=True)
        self.assertTrue(jobs)
        self.assertEqual({j.experiment for j in jobs}, {"E1", "E3", "E4", "E5", "E6"})

    def test_aggregate_groups_by_experiment(self) -> None:
        rows = [
            {
                "experiment": "E1",
                "system": "tacet",
                "tag": "",
                "seed": 0,
                "total_cost": 1.0,
                "accuracy": 0.9,
                "avg_latency_ms": 5.0,
                "tier_counts": {1: 2, 3: 1},
                "cost_trajectory": [0.1, 0.2],
                "cost_by_class": {},
                "accuracy_by_class": {},
                "n_queries": 3,
                "synthesised_rules": [],
            }
        ]
        out = aggregate(rows)
        self.assertIn("tacet", out["E1"])
        self.assertAlmostEqual(out["E1"]["tacet"]["cost"]["mean"], 1.0)
        self.assertEqual(out["E1"]["tacet"]["cost"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
