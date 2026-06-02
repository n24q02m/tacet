import unittest

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestDecoder(unittest.TestCase):
    def test_edge_head_logit_shape(self):
        from tacet.experimental.dynamics.decoder import EdgeProbabilityHead

        head = EdgeProbabilityHead(z_dim=64, kge_dim=16, hidden=32)
        batch = 4
        z = torch.zeros(batch, 64)
        s = torch.zeros(batch, 16)
        r = torch.zeros(batch, 16)
        o = torch.zeros(batch, 16)
        logits = head(z, s, r, o)
        self.assertEqual(logits.shape, (batch,))

    def test_edge_head_single_item(self):
        from tacet.experimental.dynamics.decoder import EdgeProbabilityHead

        head = EdgeProbabilityHead(z_dim=8, kge_dim=4, hidden=16)
        z = torch.randn(1, 8)
        s = torch.randn(1, 4)
        r = torch.randn(1, 4)
        o = torch.randn(1, 4)
        logits = head(z, s, r, o)
        self.assertEqual(logits.shape, (1,))
        # logit is a scalar float, not NaN
        self.assertFalse(torch.isnan(logits).any())

    def test_edge_head_grad_flows(self):
        from tacet.experimental.dynamics.decoder import EdgeProbabilityHead

        head = EdgeProbabilityHead(z_dim=8, kge_dim=4, hidden=16)
        z = torch.randn(2, 8, requires_grad=True)
        s = torch.randn(2, 4)
        r = torch.randn(2, 4)
        o = torch.randn(2, 4)
        logits = head(z, s, r, o)
        logits.sum().backward()
        self.assertIsNotNone(z.grad)
