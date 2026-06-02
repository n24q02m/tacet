"""Tests for the worked causal demonstration (Task 4).

These exercise the small confounded SCM built by ``build_demo_scm`` in
``experiments/run_causal_demo.py``: a classic confounder ``exposure`` over a
treatment/outcome pair where the naive observational association is biased and
``do``-intervention recovers the true effect. The identification side asserts
that the backdoor criterion returns the expected adjustment set.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from tacet.core.causal import backdoor_set

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from run_causal_demo import build_demo_scm  # noqa: E402


class TestCausalDemo(unittest.TestCase):
    def test_intervention_differs_from_observation_on_confounded_edge(self) -> None:
        """Naive P(Y=1 | X=1) (confounded) must differ from P(Y=1 | do(X=1))."""
        model = build_demo_scm()

        observational = model.probability("downstream", 1, evidence={"treatment": 1})
        intervened = model.intervene(do={"treatment": 1})
        interventional = intervened.probability("downstream", 1)

        # The shared confounder inflates the observational association, so the
        # two estimates must be materially different (not just MC noise).
        self.assertGreater(abs(observational - interventional), 0.05)

    def test_backdoor_set_is_the_confounder(self) -> None:
        """The adjustment set that de-confounds treatment->downstream is {confounder}."""
        model = build_demo_scm()
        adjustment = backdoor_set(model, "treatment", "downstream")
        self.assertEqual(adjustment, {"confounder"})


if __name__ == "__main__":
    unittest.main()
