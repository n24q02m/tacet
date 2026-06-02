"""Tests cho text-attributed KGE (G1.1: multi-modal Tier-2)."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):  # pragma: no cover
    HAS_TORCH = False

from tacet.kge.kge_textual import hash_encoder, seed_kge_from_descriptions

if HAS_TORCH:
    from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig


class TestHashEncoder(unittest.TestCase):
    def test_shape_and_normalised(self) -> None:
        enc = hash_encoder(dim=32)
        out = enc(["hello world", "this is tacet"])
        self.assertEqual(out.shape, (2, 32))
        norms = np.linalg.norm(out, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_deterministic_for_same_input(self) -> None:
        enc = hash_encoder(dim=16, seed=42)
        a = enc(["foo bar baz"])
        b = enc(["foo bar baz"])
        self.assertTrue(np.allclose(a, b))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestSeedKGE(unittest.TestCase):
    def test_seed_writes_text_init(self) -> None:
        kge = TorchComplEx(TorchKGEConfig(dim=16, epochs=2, device="cpu"))
        desc = {"Alice": "person who works at acme", "Bob": "person who works at zoot"}
        seed_kge_from_descriptions(kge, desc, encoder=hash_encoder(dim=8))
        self.assertIsNotNone(kge._text_init)
        self.assertIn("Alice", kge._text_init)
        self.assertEqual(kge._text_init["Alice"].shape, (8,))

    def test_seed_then_fit_uses_text_init(self) -> None:
        # If text init runs, _E_re row for "Alice" must equal text vec
        # projected through the deterministic random projection (i.e.
        # not equal the random init that would otherwise be there).
        kge = TorchComplEx(TorchKGEConfig(dim=16, epochs=1, device="cpu", seed=0))
        desc = {"Alice": "alpha", "Bob": "beta"}
        seed_kge_from_descriptions(kge, desc, encoder=hash_encoder(dim=8))
        # Fit on a trivial triple set; _E_re for Alice is filled by
        # _seed_from_text before training and then updated by 1 epoch
        # of SGD.  Compare against a parallel KGE without text init —
        # the two should diverge on the seeded rows.
        triples = [
            ("Alice", "knows", "Bob"),
            ("Bob", "knows", "Alice"),
            ("Alice", "owns", "Cat"),
            ("Bob", "owns", "Dog"),
        ]
        kge.fit(triples)
        # also fit a no-text parallel and compare
        kge_plain = TorchComplEx(TorchKGEConfig(dim=16, epochs=1, device="cpu", seed=0))
        kge_plain.fit(triples)
        i_alice = kge.ent["Alice"]
        diff = float((kge._E_re[i_alice] - kge_plain._E_re[i_alice]).abs().sum().item())
        # The seeded row diverges noticeably from the plain init after
        # one epoch; we just assert the *signal* is non-zero (the exact
        # magnitude depends on the random projection and SGD step).
        self.assertGreater(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
