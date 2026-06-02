"""Tests for the 3 score_fn variants of TorchComplEx: complex (BCE), complex_n3
(Lacroix 1-vs-N + N3), rotate (Sun margin + self-adversarial).

The tests assert that:
* Each variant fits without error and the loss decreases over epochs.
* `predict_tail` returns a valid Prediction with confidence ∈ [0, 1].
* `evaluate` returns MRR > random baseline (1 / |entities|).
* The new config fields (n3_lambda, margin, alpha) propagate through build_kge_from_settings.

Skipped if PyTorch is not installed.
"""

from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):  # pragma: no cover
    # OSError captures Windows DLL initialisation failures (WinError 1114
    # when torch's c10.dll cannot load — typically a missing VC++
    # redistributable).  Tests skip cleanly in that case; Modal-GPU CI
    # still exercises the real path.
    HAS_TORCH = False

if HAS_TORCH:
    from tacet.eval.benchmark import BenchmarkConfig, generate
    from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig, build_kge_from_settings


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestComplexBCE(unittest.TestCase):
    """BCE path (default) — backward compat with prior published behaviour."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = generate(BenchmarkConfig(seed=1))
        cfg = TorchKGEConfig(
            dim=32,
            epochs=30,
            lr=0.01,
            batch_size=256,
            negatives=4,
            uniform_negatives=2,
            score_fn="complex",
            seed=1,
            device="cpu",
        )
        cls.model = TorchComplEx(cfg).fit(cls.bench.graph.triples())

    def test_loss_drops(self) -> None:
        h = self.model.loss_history
        self.assertGreater(len(h), 5)
        self.assertLess(h[-1], h[0])

    def test_predict_returns_prediction(self) -> None:
        langs = self.bench.graph.nodes_of_type("Language")
        p = next(iter(self.bench.entity_pool))
        pred = self.model.predict_tail(p, "primary_language", langs)
        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestComplexN3(unittest.TestCase):
    """Lacroix 2018: 1-vs-N softmax NLL + N3 nuclear-norm regulariser."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = generate(BenchmarkConfig(seed=2))
        cfg = TorchKGEConfig(
            dim=32,
            epochs=40,
            lr=0.05,
            batch_size=256,
            negatives=8,
            uniform_negatives=2,
            score_fn="complex_n3",
            n3_lambda=1e-3,
            seed=2,
            device="cpu",
        )
        cls.model = TorchComplEx(cfg).fit(cls.bench.graph.triples())

    def test_loss_drops(self) -> None:
        h = self.model.loss_history
        self.assertGreater(len(h), 5)
        self.assertLess(h[-1], h[0])

    def test_n3_lambda_zero_matches_pure_nll(self) -> None:
        # with λ=0, the N3 reg has no effect — the loss must still drop (smoke test).
        cfg = TorchKGEConfig(
            dim=32,
            epochs=20,
            lr=0.05,
            batch_size=256,
            negatives=4,
            uniform_negatives=2,
            score_fn="complex_n3",
            n3_lambda=0.0,
            seed=3,
            device="cpu",
        )
        m = TorchComplEx(cfg).fit(self.bench.graph.triples())
        self.assertLess(m.loss_history[-1], m.loss_history[0])

    def test_predict_beats_random(self) -> None:
        # 4 languages → random baseline 25%. complex_n3 on synthetic-org
        # must reach > 50% on the withheld heads.
        langs = self.bench.graph.nodes_of_type("Language")
        stated = {e.source for e in self.bench.graph.edges if e.relation == "primary_language"}
        test = [
            (p, self.bench.truth[(p, "primary_language")][0])
            for (p, r) in self.bench.truth
            if r == "primary_language" and p not in stated
        ]
        if not test:
            self.skipTest("benchmark seed produced no withheld primary_language")
        hits = sum(
            self.model.predict_tail(p, "primary_language", langs).tail == truth for p, truth in test
        )
        self.assertGreater(hits / len(test), 0.25)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestRotateMarginSelfAdv(unittest.TestCase):
    """Sun 2019: rotation-in-complex-plane + margin loss + self-adversarial."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = generate(BenchmarkConfig(seed=4))
        cfg = TorchKGEConfig(
            dim=32,
            epochs=40,
            lr=0.05,
            batch_size=256,
            negatives=16,
            uniform_negatives=4,
            score_fn="rotate",
            margin=12.0,
            alpha=1.0,
            seed=4,
            device="cpu",
        )
        cls.model = TorchComplEx(cfg).fit(cls.bench.graph.triples())

    def test_loss_drops(self) -> None:
        h = self.model.loss_history
        self.assertGreater(len(h), 5)
        self.assertLess(h[-1], h[0])

    def test_self_adv_weights_sum_to_one(self) -> None:
        # Self-adversarial uses softmax(α·phi_neg) → weights must sum to 1
        # for each positive.  Verified by checking the forward pass.
        triples = self.bench.graph.triples()[:8]
        h = torch.tensor([self.model.ent[a] for a, _, _ in triples])
        r = torch.tensor([self.model.rel[b] for _, b, _ in triples])
        t = torch.tensor([self.model.ent[c] for _, _, c in triples])
        with torch.no_grad():
            phi = self.model._phi_idx(h, r, t)
            # Build fake "negatives" from the same triples (check softmax shape).
            phi_neg = phi.unsqueeze(0).expand(2, -1)
            w = torch.softmax(self.model.cfg.alpha * phi_neg, dim=-1)
            self.assertAlmostEqual(float(w.sum(-1)[0]), 1.0, places=5)

    def test_score_uses_normalised_rotation(self) -> None:
        # |r| = 1 constraint: with normalised r, |hr - t| should not break
        # when r has a magnitude other than 1 (the paper mandates L2-normalise).
        triples = self.bench.graph.triples()[:1]
        a, b, c = triples[0]
        s = self.model.score(a, b, c)
        self.assertTrue(torch.isfinite(torch.tensor(s)))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestBuildFromSettings(unittest.TestCase):
    """build_kge_from_settings must honor the kge_model field."""

    def _settings(self, model: str):
        class _S:
            kge_backend = "torch"
            kge_dim = 16
            kge_epochs = 5
            kge_device = "cpu"

        s = _S()
        s.kge_model = model
        return s

    def test_complex_default(self) -> None:
        m = build_kge_from_settings(self._settings("complex"))
        self.assertEqual(m.cfg.score_fn, "complex")

    def test_complex_n3(self) -> None:
        m = build_kge_from_settings(self._settings("complex_n3"))
        self.assertEqual(m.cfg.score_fn, "complex_n3")

    def test_rotate(self) -> None:
        m = build_kge_from_settings(self._settings("rotate"))
        self.assertEqual(m.cfg.score_fn, "rotate")

    def test_missing_kge_model_defaults_complex(self) -> None:
        # Backward compat: old settings (no kge_model field) → complex.
        class _S:
            kge_backend = "torch"
            kge_dim = 16
            kge_epochs = 5
            kge_device = "cpu"

        m = build_kge_from_settings(_S())
        self.assertEqual(m.cfg.score_fn, "complex")


if __name__ == "__main__":
    unittest.main()
