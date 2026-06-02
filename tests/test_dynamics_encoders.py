import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not importable")
class TestGraphEncoder(unittest.TestCase):
    def test_mean_pool_encoder_returns_fixed_dim(self):
        from tacet.core.graph import WorldGraph
        from tacet.experimental.dynamics.encoders import ComplexN3PooledEncoder
        from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig

        g = WorldGraph(name="t")
        g.add_edge("A", "rel", "B")
        g.add_edge("B", "rel", "C")
        kge = TorchComplEx(TorchKGEConfig(dim=16, epochs=2, score_fn="complex_n3"))
        kge.fit(g.triples())
        enc = ComplexN3PooledEncoder(kge)
        z = enc.encode(g)
        self.assertEqual(z.shape, (32,))  # 16 re + 16 im


class TestEventEncoder(unittest.TestCase):
    def test_bow_encoder_counts_event_types(self):
        from tacet.experimental.dynamics.encoders import BagOfTypesEventEncoder
        from tacet.experimental.dynamics.events import Event, EventBatch

        enc = BagOfTypesEventEncoder(event_types=["visits", "trade"])
        batch = EventBatch(
            timestamp=0.0,
            events=[
                Event(timestamp=0.0, type="visits", actor="A", target="B"),
                Event(timestamp=0.0, type="visits", actor="C", target="D"),
                Event(timestamp=0.0, type="trade", actor="A", target="C"),
            ],
        )
        v = enc.encode(batch)
        self.assertEqual(v.tolist(), [2.0, 1.0])
