"""Tests for the PyTorch KGE backend (TorchComplEx)."""

from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

if HAS_TORCH:
    from tacet.eval.benchmark import BenchmarkConfig, generate
    from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchComplEx(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = generate(BenchmarkConfig(seed=1))
        # Increase epochs to ensure it beats the random baseline (25%)
        cls.model = TorchComplEx(TorchKGEConfig(epochs=80, seed=1, device="cpu")).fit(
            cls.bench.graph.triples()
        )

    def test_training_reduces_loss(self) -> None:
        h = self.model.loss_history
        self.assertGreater(len(h), 0)
        self.assertLess(h[-1], h[0])

    def test_predicts_held_out_language(self) -> None:
        langs = self.bench.graph.nodes_of_type("Language")
        stated = {e.source for e in self.bench.graph.edges if e.relation == "primary_language"}
        test = [
            (p, self.bench.truth[(p, "primary_language")][0])
            for (p, r) in self.bench.truth
            if r == "primary_language" and p not in stated
        ]
        if not test:
            self.skipTest("benchmark seed produced no withheld primary_language")

        hit = sum(
            self.model.predict_tail(p, "primary_language", langs).tail == truth for p, truth in test
        )
        # 4 languages -> 25% baseline.
        self.assertGreater(hit / len(test), 0.25)

    def test_confidence_in_unit_interval(self) -> None:
        langs = self.bench.graph.nodes_of_type("Language")
        p = next(iter(self.bench.entity_pool))
        pred = self.model.predict_tail(p, "primary_language", langs)
        self.assertTrue(0.0 <= pred.confidence <= 1.0)

    def test_unknown_symbol_returns_none(self) -> None:
        self.assertIsNone(self.model.predict_tail("ghost", "primary_language", ["x"]))

    def test_partial_fit_warm_start(self) -> None:
        before = dict(self.model.ent)
        self.model.partial_fit(self.bench.graph.triples()[:50], epochs=5)
        for e in before:
            self.assertIn(e, self.model.ent)

    def test_calibration_sets_temperature(self) -> None:
        model = TorchComplEx(TorchKGEConfig(epochs=40, seed=2, device="cpu")).fit(
            self.bench.graph.triples()
        )
        model.calibrate(self.bench.calibration)
        self.assertGreater(model.temperature, 0.0)

    def test_evaluate_returns_metrics(self) -> None:
        test = self.bench.graph.triples()[:40]
        metrics = self.model.evaluate(test, set(self.bench.graph.triples()))
        self.assertGreaterEqual(metrics["MRR"], 0.0)
        self.assertLessEqual(metrics["Hits@10"], 1.0)


if __name__ == "__main__":
    unittest.main()
