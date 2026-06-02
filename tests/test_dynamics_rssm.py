import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestRSSM(unittest.TestCase):
    def test_prior_forward_shapes(self):
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        cfg = RSSMConfig(z_categories=32, z_per_state=8, h_dim=64, state_in_dim=32, event_in_dim=10)
        rssm = RSSM(cfg)
        z_dim = cfg.z_categories * cfg.z_per_state
        h, z = rssm.initial(1)
        e_t = torch.zeros(1, cfg.event_in_dim)
        z_next, h_next = rssm.prior(h, z, e_t)
        self.assertEqual(z_next.shape, (1, z_dim))
        self.assertEqual(h_next.shape, (1, cfg.h_dim))

    def test_posterior_forward_shapes(self):
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        cfg = RSSMConfig(z_categories=32, z_per_state=8, h_dim=64, state_in_dim=32, event_in_dim=10)
        rssm = RSSM(cfg)
        z_dim = cfg.z_categories * cfg.z_per_state
        h, z = rssm.initial(1)
        e_t = torch.zeros(1, cfg.event_in_dim)
        s_t = torch.zeros(1, cfg.state_in_dim)
        z_post, h_next = rssm.posterior(h, z, e_t, s_t)
        self.assertEqual(z_post.shape, (1, z_dim))
        self.assertEqual(h_next.shape, (1, cfg.h_dim))

    def test_initial_shapes(self):
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        cfg = RSSMConfig(z_categories=8, z_per_state=4, h_dim=16, state_in_dim=8, event_in_dim=4)
        rssm = RSSM(cfg)
        h, z = rssm.initial(3)
        self.assertEqual(h.shape, (3, cfg.h_dim))
        self.assertEqual(z.shape, (3, cfg.z_categories * cfg.z_per_state))
