import unittest

import numpy as np

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

    def test_mean_pool_encoder_empty_graph(self):
        from tacet.core.graph import WorldGraph
        from tacet.experimental.dynamics.encoders import ComplexN3PooledEncoder
        from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig

        g = WorldGraph(name="t")
        kge = TorchComplEx(TorchKGEConfig(dim=16, epochs=2, score_fn="complex_n3"))
        # We don't need to fit it, just need cfg.dim
        enc = ComplexN3PooledEncoder(kge)
        z = enc.encode(g)
        self.assertEqual(z.shape, (32,))
        self.assertTrue(np.all(z == 0))


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

    def test_bow_encoder_unknown_types(self):
        from tacet.experimental.dynamics.encoders import BagOfTypesEventEncoder
        from tacet.experimental.dynamics.events import Event, EventBatch

        enc = BagOfTypesEventEncoder(event_types=["visits"])
        batch = EventBatch(
            timestamp=0.0,
            events=[
                Event(timestamp=0.0, type="trade", actor="A", target="C"),
            ],
        )
        v = enc.encode(batch)
        self.assertEqual(v.tolist(), [0.0])

    def test_action_as_node_encoder(self):
        from types import SimpleNamespace

        from tacet.experimental.dynamics.encoders import ActionAsNodeEventEncoder
        from tacet.experimental.dynamics.events import Event, EventBatch

        class FakeKGE:
            def __init__(self):
                self.ent = {"A": 0, "B": 1}
                self.cfg = SimpleNamespace(dim=2)
                self._E_re = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        enc = ActionAsNodeEventEncoder(["t1"], FakeKGE())
        self.assertEqual(enc.dim, 5)  # n_types(1) + 2*kge_dim(2)

        batch = EventBatch(timestamp=0.0, events=[Event(0.0, "t1", "A", "B")])
        v = enc.encode(batch)
        self.assertEqual(v.shape, (5,))
        self.assertEqual(v[0], 1.0)  # type histogram
        # entity part = mean of (actor_re || target_re) = ([1,0] || [0,1])
        self.assertTrue(np.allclose(v[1:], [1.0, 0.0, 0.0, 1.0]))

    def test_action_as_node_unknown_entity_zero(self):
        from types import SimpleNamespace

        from tacet.experimental.dynamics.encoders import ActionAsNodeEventEncoder
        from tacet.experimental.dynamics.events import Event, EventBatch

        class FakeKGE:
            def __init__(self):
                self.ent = {"A": 0}
                self.cfg = SimpleNamespace(dim=2)
                self._E_re = np.array([[1.0, 0.0]], dtype=np.float32)

        enc = ActionAsNodeEventEncoder(["t1"], FakeKGE())
        batch = EventBatch(timestamp=0.0, events=[Event(0.0, "t1", "Z", None)])
        v = enc.encode(batch)
        # unknown actor + no target -> entity part all zeros
        self.assertTrue(np.allclose(v[1:], [0.0, 0.0, 0.0, 0.0]))

    def test_action_as_node_empty_batch(self):
        from types import SimpleNamespace

        from tacet.experimental.dynamics.encoders import ActionAsNodeEventEncoder
        from tacet.experimental.dynamics.events import EventBatch

        class FakeKGE:
            def __init__(self):
                self.ent = {}
                self.cfg = SimpleNamespace(dim=2)
                self._E_re = np.array([], dtype=np.float32)

        enc = ActionAsNodeEventEncoder(["t1"], FakeKGE())
        batch = EventBatch(timestamp=0.0, events=[])
        v = enc.encode(batch)
        self.assertEqual(v.shape, (5,))
        self.assertTrue(np.all(v == 0))
