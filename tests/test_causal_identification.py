"""Tests cho front-door + IV identification (G1.2)."""

from __future__ import annotations

import unittest

from tacet.core.causal import (
    CausalModel,
    front_door_set,
    instrumental_variables,
)


def _smoking_tar_cancer() -> CausalModel:
    """Pearl's canonical front-door example.

    U (unobserved) ↔ smoking, cancer
    smoking → tar → cancer
    """
    m = CausalModel()
    m.add_variable("smoking", domain=(0, 1))
    m.add_variable("tar", domain=(0, 1), parents=("smoking",))
    m.add_variable("cancer", domain=(0, 1), parents=("tar",))
    # The unobserved genetic confounder between smoking and cancer.
    m.add_bidirected_edge("smoking", "cancer")
    return m


def _iv_example() -> CausalModel:
    """Standard instrumental-variable graph.

    instrument → treatment → outcome
    unobserved ↔ treatment, outcome
    """
    m = CausalModel()
    m.add_variable("instrument", domain=(0, 1))
    m.add_variable("treatment", domain=(0, 1), parents=("instrument",))
    m.add_variable("outcome", domain=(0, 1), parents=("treatment",))
    m.add_bidirected_edge("treatment", "outcome")
    return m


class TestFrontDoorSet(unittest.TestCase):
    def test_finds_mediator_in_smoking_example(self) -> None:
        m = _smoking_tar_cancer()
        z = front_door_set(m, "smoking", "cancer")
        self.assertIsNotNone(z)
        self.assertEqual(z, {"tar"})

    def test_returns_none_when_no_directed_path(self) -> None:
        m = CausalModel()
        m.add_variable("x", domain=(0, 1))
        m.add_variable("y", domain=(0, 1))  # no edge x→y
        self.assertIsNone(front_door_set(m, "x", "y"))

    def test_returns_empty_set_when_treatment_equals_outcome(self) -> None:
        m = _smoking_tar_cancer()
        z = front_door_set(m, "smoking", "smoking")
        self.assertEqual(z, set())


class TestInstrumentalVariables(unittest.TestCase):
    def test_finds_instrument(self) -> None:
        m = _iv_example()
        ivs = instrumental_variables(m, "treatment", "outcome")
        self.assertIn("instrument", ivs)

    def test_returns_empty_when_bidirected_violates(self) -> None:
        # Even though Z causes X causes Y, a Z↔Y bidirected edge breaks IV.
        m = CausalModel()
        m.add_variable("z", domain=(0, 1))
        m.add_variable("x", domain=(0, 1), parents=("z",))
        m.add_variable("y", domain=(0, 1), parents=("x",))
        m.add_bidirected_edge("z", "y")
        ivs = instrumental_variables(m, "x", "y")
        self.assertNotIn("z", ivs)

    def test_ignores_descendants_of_treatment(self) -> None:
        # Direct descendant of treatment shouldn't qualify as IV
        # (not a cause of treatment).
        m = _iv_example()
        ivs = instrumental_variables(m, "treatment", "outcome")
        self.assertNotIn("outcome", ivs)
        self.assertNotIn("treatment", ivs)


if __name__ == "__main__":
    unittest.main()
