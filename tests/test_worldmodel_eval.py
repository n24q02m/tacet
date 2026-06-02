import numpy as np

from tacet.experimental.dynamics.worldmodel_eval import expected_calibration_error


def test_ece_perfect_is_zero():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert expected_calibration_error(probs, labels, n_bins=5) == 0.0


def test_ece_detects_miscalibration():
    probs = np.array([0.9, 0.9, 0.9, 0.9])  # confident
    labels = np.array([1, 0, 0, 0])  # but mostly wrong (acc 0.25)
    assert expected_calibration_error(probs, labels, n_bins=5) > 0.5


def test_ece_empty_is_zero():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0
