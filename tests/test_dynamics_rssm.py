import unittest

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestRSSM(unittest.TestCase):
    def test_config_defaults(self):
        from tacet.experimental.dynamics.rssm import RSSMConfig

        cfg = RSSMConfig()
        self.assertEqual(cfg.z_categories, 32)
        self.assertEqual(cfg.z_per_state, 32)
        self.assertEqual(cfg.h_dim, 256)
        self.assertEqual(cfg.state_in_dim, 400)
        self.assertEqual(cfg.event_in_dim, 280)
        self.assertEqual(cfg.hidden, 256)

    def test_categorical_logic(self):
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        cfg = RSSMConfig(z_categories=4, z_per_state=2)
        rssm = RSSM(cfg)

        # Test unimix=0
        logits = torch.zeros(1, cfg.z_per_state * cfg.z_categories)
        p = rssm._categorical(logits, unimix=0.0)
        # 1/4 = 0.25
        expected = torch.full((1, cfg.z_per_state * cfg.z_categories), 0.25)
        torch.testing.assert_close(p, expected)

        # Test unimix > 0
        unimix = 0.1
        rssm._categorical(logits, unimix=unimix)
        # (1-0.1)*0.25 + 0.1/4 = 0.9*0.25 + 0.025 = 0.225 + 0.025 = 0.25
        # Since uniform stays uniform, let's try non-uniform
        logits[0, 0] = 10.0  # High value for first category of first state
        p_skewed = rssm._categorical(logits, unimix=unimix)
        p_view = p_skewed.view(cfg.z_per_state, cfg.z_categories)
        # Sum of probabilities for each state should be 1
        torch.testing.assert_close(p_view.sum(dim=-1), torch.ones(cfg.z_per_state))
        # Each category should be at least unimix / z_categories
        self.assertTrue((p_view >= unimix / cfg.z_categories).all())

    def test_initial_device(self):
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        cfg = RSSMConfig(z_categories=8, z_per_state=4, h_dim=16)
        rssm = RSSM(cfg)
        device = torch.device("cpu")
        h, z = rssm.initial(3, device=device)
        self.assertEqual(h.device.type, "cpu")
        self.assertEqual(z.device.type, "cpu")
        self.assertEqual(h.shape, (3, cfg.h_dim))
        self.assertEqual(z.shape, (3, cfg.z_categories * cfg.z_per_state))

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

    def test_gradient_flow(self):
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

        cfg = RSSMConfig(z_categories=8, z_per_state=4, h_dim=16, state_in_dim=8, event_in_dim=4)
        rssm = RSSM(cfg)

        h0, z0 = rssm.initial(2)
        e = torch.randn(2, cfg.event_in_dim)
        s = torch.randn(2, cfg.state_in_dim)

        # Test prior gradient
        z_prior, h_prior = rssm.prior(h0, z0, e)
        loss_prior = z_prior.sum() + h_prior.sum()
        loss_prior.backward()

        for name, param in rssm.named_parameters():
            if "posterior_head" not in name:
                self.assertIsNotNone(param.grad, f"No grad for {name} in prior")
            else:
                self.assertIsNone(param.grad, f"Unexpected grad for {name} in prior")

        rssm.zero_grad()

        # Test posterior gradient
        z_post, h_post = rssm.posterior(h0, z0, e, s)
        loss_post = z_post.sum() + h_post.sum()
        loss_post.backward()

        for name, param in rssm.named_parameters():
            if "prior_head" not in name:
                self.assertIsNotNone(param.grad, f"No grad for {name} in posterior")
            else:
                self.assertIsNone(param.grad, f"Unexpected grad for {name} in posterior")


if __name__ == "__main__":
    unittest.main()
