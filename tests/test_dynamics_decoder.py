import unittest

import torch

try:
    from tacet.experimental.dynamics.decoder import EdgeProbabilityHead

    HAS_DECODER = True
except ImportError:
    HAS_DECODER = False

HAS_TORCH = True  # We know it is installed now as per my sync command


@unittest.skipUnless(HAS_TORCH and HAS_DECODER, "torch or decoder missing")
class TestDecoder(unittest.TestCase):
    def test_edge_head_logit_shape(self):
        head = EdgeProbabilityHead(z_dim=64, kge_dim=16, hidden=32)
        batch = 4
        z = torch.zeros(batch, 64)
        s = torch.zeros(batch, 16)
        r = torch.zeros(batch, 16)
        o = torch.zeros(batch, 16)
        logits = head(z, s, r, o)
        self.assertEqual(logits.shape, (batch,))

    def test_edge_head_single_item(self):
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
        head = EdgeProbabilityHead(z_dim=8, kge_dim=4, hidden=16)
        z = torch.randn(2, 8, requires_grad=True)
        s = torch.randn(2, 4)
        r = torch.randn(2, 4)
        o = torch.randn(2, 4)
        logits = head(z, s, r, o)
        logits.sum().backward()
        self.assertIsNotNone(z.grad)

    def test_edge_head_different_hidden(self):
        # Verify it works with various hidden dimensions
        for h in [1, 128, 512]:
            head = EdgeProbabilityHead(z_dim=10, kge_dim=5, hidden=h)
            z = torch.randn(3, 10)
            s = torch.randn(3, 5)
            r = torch.randn(3, 5)
            o = torch.randn(3, 5)
            logits = head(z, s, r, o)
            self.assertEqual(logits.shape, (3,))

    def test_edge_head_mismatched_batch_size(self):
        head = EdgeProbabilityHead(z_dim=8, kge_dim=4, hidden=16)
        z = torch.randn(2, 8)
        s = torch.randn(3, 4)  # mismatched
        r = torch.randn(2, 4)
        o = torch.randn(2, 4)
        with self.assertRaises(RuntimeError):
            head(z, s, r, o)

    def test_edge_head_zero_batch(self):
        head = EdgeProbabilityHead(z_dim=8, kge_dim=4, hidden=16)
        z = torch.randn(0, 8)
        s = torch.randn(0, 4)
        r = torch.randn(0, 4)
        o = torch.randn(0, 4)
        logits = head(z, s, r, o)
        self.assertEqual(logits.shape, (0,))


if __name__ == "__main__":
    unittest.main()
