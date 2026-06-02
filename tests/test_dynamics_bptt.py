import unittest

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class TestUnimix(unittest.TestCase):
    def test_unimix_floors_probability_mass(self):
        import torch

        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        m = RSSM(RSSMConfig(z_categories=4, z_per_state=2, event_in_dim=3, state_in_dim=4))
        logits = torch.full((1, 8), -50.0)
        logits[0, 0] = 50.0  # near one-hot
        z = m._categorical(logits)  # noqa: SLF001
        # with 1% unimix, every class keeps positive mass
        self.assertGreater(z.min().item(), 0.0)

    def test_no_unimix_can_be_zero(self):
        import torch

        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        m = RSSM(RSSMConfig(z_categories=4, z_per_state=2, event_in_dim=3, state_in_dim=4))
        logits = torch.full((1, 8), -50.0)
        logits[0, 0] = 50.0
        z = m._categorical(logits, unimix=0.0)  # noqa: SLF001
        self.assertAlmostEqual(z.min().item(), 0.0, places=6)


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class TestKLBalanced(unittest.TestCase):
    def test_free_bits_floor(self):
        import torch

        from tacet.experimental.dynamics.train import _kl_balanced

        # identical distributions -> KL 0, but free_bits floors it
        p = torch.full((1, 8), 0.125)
        val = _kl_balanced(p, p, z_per=2, z_cat=4, free_bits=1.0)
        self.assertGreaterEqual(val.item(), 1.0 - 1e-6)
