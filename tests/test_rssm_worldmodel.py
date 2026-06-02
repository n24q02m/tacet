import types
import unittest

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch unavailable")
class TestRSSMWorldModel(unittest.TestCase):
    def _build(self):
        import torch

        from tacet.experimental.dynamics.decoder import EdgeProbabilityHead
        from tacet.experimental.dynamics.encoders import (
            BagOfTypesEventEncoder,
            ComplexN3PooledEncoder,
        )
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig
        from tacet.experimental.dynamics.rssm_worldmodel import RSSMWorldModel

        dim = 4
        kge = types.SimpleNamespace(
            device=torch.device("cpu"),
            cfg=types.SimpleNamespace(dim=dim),
            ent={"a": 0, "b": 1, "c": 2},
            rel={"r": 0},
            _E_re=torch.randn(3, dim),
            _E_im=torch.randn(3, dim),
            _R_re=torch.randn(1, dim),
        )
        event_types = ["t"]
        state_enc = ComplexN3PooledEncoder(kge)
        event_enc = BagOfTypesEventEncoder(event_types)
        z_cat, z_per = 4, 2
        z_dim = z_cat * z_per
        rssm = RSSM(
            RSSMConfig(
                z_categories=z_cat,
                z_per_state=z_per,
                state_in_dim=2 * dim,
                event_in_dim=len(event_types),
                h_dim=8,
                hidden=8,
            )
        )
        decoder = EdgeProbabilityHead(z_dim=z_dim, kge_dim=dim, hidden=8)
        return RSSMWorldModel(rssm, decoder, state_enc, event_enc, kge)

    def test_rollout_shapes(self):
        from tacet.core.graph import WorldGraph
        from tacet.experimental.dynamics.events import Event, EventBatch

        wm = self._build()
        g = WorldGraph(name="g")
        g.add_edge("a", "r", "b")
        s0 = wm.observe(g)
        self.assertEqual(len(s0), 2)  # (h, z)
        plan = [
            EventBatch(timestamp=float(i), events=[Event(float(i), "t", "a", "b")])
            for i in range(3)
        ]
        traj = wm.rollout(s0, plan)
        self.assertEqual(len(traj.states), 4)  # start + 3 steps
        self.assertEqual(len(traj.actions), 3)
        self.assertEqual(len(traj.rewards), 3)

    def test_predict_advances_state(self):
        import torch

        from tacet.core.graph import WorldGraph
        from tacet.experimental.dynamics.events import Event, EventBatch

        wm = self._build()
        g = WorldGraph(name="g")
        g.add_edge("a", "r", "b")
        h0, z0 = wm.observe(g)
        h1, z1 = wm.predict((h0, z0), EventBatch(0.0, [Event(0.0, "t", "a", "b")]))
        # the recurrent hidden state should change after a step
        self.assertFalse(torch.equal(h0, h1))
