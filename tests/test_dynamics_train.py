"""Smoke test for Layer 4 teacher-forced training loop.

Skipped on this Windows box because torch DLL does not load (WinError 1114).
On a torch-capable machine (Modal/Linux GPU) this asserts:
  - loss_history is non-empty
  - last epoch loss <= first epoch loss (loss decreases or is stable)
"""

import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


def _make_tiny_trajectory(no_change=False):
    """Build a 4-step Trajectory with tiny WorldGraphs for smoke testing."""
    from tacet.core.graph import WorldGraph
    from tacet.experimental.dynamics.events import Event, EventBatch
    from tacet.experimental.dynamics.trajectory import Trajectory

    # Shared entities and relation
    entities = ["A", "B", "C", "D"]
    rel = "rel"

    # Build 5 snapshots (4 transitions)
    snapshots = []
    for i in range(5):
        g = WorldGraph(name=f"g{i}")
        for ent in entities:
            g.add_node(ent)

        if no_change:
            # All snapshots have the same edges
            g.add_edge("A", rel, "B")
        else:
            # Cycle edges through each snapshot
            src = entities[i % len(entities)]
            tgt = entities[(i + 1) % len(entities)]
            g.add_edge(src, rel, tgt)
        snapshots.append(g)

    # Build 4 event batches (one per transition)
    event_types = ["move", "stay"]
    event_batches = []
    for i in range(4):
        evt = Event(timestamp=float(i), type=event_types[i % 2], actor=entities[i % len(entities)])
        event_batches.append(EventBatch(timestamp=float(i), events=[evt]))

    return Trajectory(snapshots=snapshots, event_batches=event_batches), event_types


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestTrainLayer4(unittest.TestCase):
    def _build_kge(self, triples):
        """Fit a tiny TorchComplEx on the given triples."""
        from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig

        cfg = TorchKGEConfig(
            dim=8, epochs=5, device="cpu", batch_size=32, negatives=2, uniform_negatives=2, seed=0
        )
        kge = TorchComplEx(cfg)
        kge.fit(triples)
        return kge

    def test_loss_history_non_empty_and_decreasing(self):
        from tacet.experimental.dynamics.decoder import EdgeProbabilityHead
        from tacet.experimental.dynamics.encoders import (
            BagOfTypesEventEncoder,
            ComplexN3PooledEncoder,
        )
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig
        from tacet.experimental.dynamics.train import TrainConfig, train_layer4

        traj, event_types = _make_tiny_trajectory()

        # Collect all triples across all snapshots for KGE fitting
        all_triples = []
        for snap in traj.snapshots:
            all_triples.extend(snap.triples())
        # Deduplicate
        all_triples = list(set(all_triples))

        kge = self._build_kge(all_triples)

        kge_dim = kge.cfg.dim  # 8
        state_in_dim = 2 * kge_dim  # ComplexN3PooledEncoder: re+im -> 16
        event_in_dim = len(event_types)  # BagOfTypesEventEncoder: 2

        state_enc = ComplexN3PooledEncoder(kge)
        event_enc = BagOfTypesEventEncoder(event_types)

        z_categories = 4
        z_per_state = 4
        z_dim = z_categories * z_per_state  # 16

        rssm_cfg = RSSMConfig(
            z_categories=z_categories,
            z_per_state=z_per_state,
            h_dim=16,
            state_in_dim=state_in_dim,
            event_in_dim=event_in_dim,
            hidden=16,
        )
        rssm = RSSM(rssm_cfg)
        decoder = EdgeProbabilityHead(z_dim=z_dim, kge_dim=kge_dim, hidden=16)

        train_cfg = TrainConfig(
            epochs=10,
            lr=1e-3,
            weight_decay=0.0,
            kl_weight=0.1,
            neg_per_pos=2,
            seed=42,
        )

        loss_history = train_layer4(
            traj=traj,
            kge=kge,
            state_enc=state_enc,
            event_enc=event_enc,
            rssm=rssm,
            decoder=decoder,
            cfg=train_cfg,
        )

        self.assertGreater(len(loss_history), 0, "loss_history must not be empty")
        # Training on a tiny graph should not blow up — loss must stay finite
        for loss_val in loss_history:
            self.assertFalse(
                loss_val != loss_val,  # NaN check
                f"NaN in loss_history: {loss_history}",
            )
        # Loss should not increase overall (last <= first), allowing 10% slack
        first, last = loss_history[0], loss_history[-1]
        self.assertLessEqual(
            last,
            first * 1.1 + 1e-6,
            f"Loss did not decrease: first={first:.4f} last={last:.4f}",
        )

    def test_kl_categorical(self):
        """Test the _kl_categorical helper directly."""
        import torch

        from tacet.experimental.dynamics.train import _kl_categorical

        z_per_state = 2
        z_categories = 3
        b = 4

        # Create two sets of logits and softmax them
        logits1 = torch.randn(b, z_per_state * z_categories)
        logits2 = torch.randn(b, z_per_state * z_categories)

        # Correctly apply softmax per category slot
        p = torch.softmax(logits1.view(b, z_per_state, z_categories), dim=-1).view(b, -1)
        q = torch.softmax(logits2.view(b, z_per_state, z_categories), dim=-1).view(b, -1)

        kl = _kl_categorical(p, q, z_per_state, z_categories)

        self.assertIsInstance(kl, torch.Tensor)
        self.assertEqual(kl.shape, ())
        self.assertGreaterEqual(kl.item(), -1e-6)  # KL >= 0 (floating point safety)

        # KL between identical distributions should be zero
        kl_zero = _kl_categorical(p, p, z_per_state, z_categories)
        self.assertLess(kl_zero.item(), 1e-6)

    def test_train_layer4_variants(self):
        """Test train_layer4 with consistency weight, large BPTT window, and no new edges."""
        from tacet.experimental.dynamics.decoder import EdgeProbabilityHead
        from tacet.experimental.dynamics.encoders import (
            BagOfTypesEventEncoder,
            ComplexN3PooledEncoder,
        )
        from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig
        from tacet.experimental.dynamics.train import TrainConfig, train_layer4

        # 1. No new edges (covers 'if not pos_edges: continue')
        traj_no_change, event_types = _make_tiny_trajectory(no_change=True)

        all_triples = []
        for snap in traj_no_change.snapshots:
            all_triples.extend(snap.triples())
        all_triples = list(set(all_triples))
        kge = self._build_kge(all_triples)

        kge_dim = kge.cfg.dim
        state_in_dim = 2 * kge_dim
        event_in_dim = len(event_types)

        state_enc = ComplexN3PooledEncoder(kge)
        event_enc = BagOfTypesEventEncoder(event_types)

        z_categories = 2
        z_per_state = 2
        z_dim = z_categories * z_per_state

        rssm_cfg = RSSMConfig(
            z_categories=z_categories,
            z_per_state=z_per_state,
            h_dim=8,
            state_in_dim=state_in_dim,
            event_in_dim=event_in_dim,
            hidden=8,
        )
        rssm = RSSM(rssm_cfg)
        decoder = EdgeProbabilityHead(z_dim=z_dim, kge_dim=kge_dim, hidden=8)

        # Config with consistency_weight > 0 and large bptt_window
        train_cfg = TrainConfig(
            epochs=2,
            consistency_weight=0.1,
            bptt_window=100,  # larger than traj length to trigger final window flush
            seed=42,
        )

        loss_history = train_layer4(
            traj=traj_no_change,
            kge=kge,
            state_enc=state_enc,
            event_enc=event_enc,
            rssm=rssm,
            decoder=decoder,
            cfg=train_cfg,
        )
        self.assertEqual(len(loss_history), 2)
        # Since no_change=True, all transitions should be skipped, epoch loss should be 0.0
        for loss in loss_history:
            self.assertEqual(loss, 0.0)

        # 2. Now run with a normal trajectory but still consistency_weight > 0 and large bptt_window
        traj, _ = _make_tiny_trajectory(no_change=False)
        loss_history_2 = train_layer4(
            traj=traj,
            kge=kge,
            state_enc=state_enc,
            event_enc=event_enc,
            rssm=rssm,
            decoder=decoder,
            cfg=train_cfg,
        )
        self.assertEqual(len(loss_history_2), 2)
        self.assertGreater(loss_history_2[0], 0.0)
