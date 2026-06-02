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


def _make_tiny_trajectory():
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
