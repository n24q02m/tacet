"""Smoke tests for Layer 4 eval module (P1.A MRR, P1.B rollout coherence, P1.C latency).

Skipped on this Windows box because torch DLL does not load (WinError 1114).
On a torch-capable machine (Modal/Linux GPU) this asserts contract / shape guarantees
for all three eval functions; no numeric performance bar is set.
"""

import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


def _make_tiny_trajectory():
    """Build a 4-step Trajectory with small WorldGraphs for smoke testing."""
    from tacet.core.graph import WorldGraph
    from tacet.experimental.dynamics.events import Event, EventBatch
    from tacet.experimental.dynamics.trajectory import Trajectory

    entities = ["A", "B", "C", "D"]
    rel = "rel"

    snapshots = []
    for i in range(5):
        g = WorldGraph(name=f"g{i}")
        for ent in entities:
            g.add_node(ent)
        src = entities[i % len(entities)]
        tgt = entities[(i + 1) % len(entities)]
        g.add_edge(src, rel, tgt)
        snapshots.append(g)

    event_types = ["move", "stay"]
    event_batches = []
    for i in range(4):
        evt = Event(
            timestamp=float(i),
            type=event_types[i % 2],
            actor=entities[i % len(entities)],
        )
        event_batches.append(EventBatch(timestamp=float(i), events=[evt]))

    return Trajectory(snapshots=snapshots, event_batches=event_batches), event_types


def _build_kge(triples):
    from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig

    cfg = TorchKGEConfig(
        dim=8,
        epochs=3,
        device="cpu",
        batch_size=32,
        negatives=2,
        uniform_negatives=2,
        seed=0,
    )
    kge = TorchComplEx(cfg)
    kge.fit(triples)
    return kge


def _build_components(traj, event_types):
    """Return (kge, state_enc, event_enc, rssm, decoder) on tiny dims."""
    from tacet.experimental.dynamics.decoder import EdgeProbabilityHead
    from tacet.experimental.dynamics.encoders import BagOfTypesEventEncoder, ComplexN3PooledEncoder
    from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig

    all_triples = list({t for snap in traj.snapshots for t in snap.triples()})
    kge = _build_kge(all_triples)

    kge_dim = kge.cfg.dim  # 8
    state_in_dim = 2 * kge_dim  # 16 — re+im from ComplexN3PooledEncoder
    event_in_dim = len(event_types)  # 2

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
    return kge, state_enc, event_enc, rssm, decoder


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestEvalSingleStep(unittest.TestCase):
    def setUp(self):
        self.traj, self.event_types = _make_tiny_trajectory()
        (self.kge, self.state_enc, self.event_enc, self.rssm, self.decoder) = _build_components(
            self.traj, self.event_types
        )

    def test_returns_required_keys(self):
        from tacet.experimental.dynamics.eval import eval_single_step

        result = eval_single_step(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            candidates_per_query=10,
        )
        for key in ("MRR", "Hits@1", "Hits@3", "Hits@10", "n"):
            self.assertIn(key, result, f"Missing key {key!r}")

    def test_mrr_in_unit_interval(self):
        from tacet.experimental.dynamics.eval import eval_single_step

        result = eval_single_step(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            candidates_per_query=10,
        )
        self.assertGreaterEqual(result["MRR"], 0.0)
        self.assertLessEqual(result["MRR"], 1.0)

    def test_hits_in_unit_interval(self):
        from tacet.experimental.dynamics.eval import eval_single_step

        result = eval_single_step(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            candidates_per_query=10,
        )
        for key in ("Hits@1", "Hits@3", "Hits@10"):
            self.assertGreaterEqual(result[key], 0.0, f"{key} < 0")
            self.assertLessEqual(result[key], 1.0, f"{key} > 1")

    def test_n_is_nonneg_int(self):
        from tacet.experimental.dynamics.eval import eval_single_step

        result = eval_single_step(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            candidates_per_query=10,
        )
        self.assertIsInstance(result["n"], int)
        self.assertGreaterEqual(result["n"], 0)


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestEvalRolloutCoherence(unittest.TestCase):
    def setUp(self):
        from tacet.core.ontology import Ontology, RelationType

        self.traj, self.event_types = _make_tiny_trajectory()
        (self.kge, self.state_enc, self.event_enc, self.rssm, self.decoder) = _build_components(
            self.traj, self.event_types
        )
        # Minimal ontology that allows the "rel" relation on any types
        self.onto = Ontology()
        self.onto.add_relation_type(RelationType(name="rel"))

    def test_returns_required_keys(self):
        from tacet.experimental.dynamics.eval import eval_rollout_coherence

        result = eval_rollout_coherence(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            self.onto,
            ks=(3, 5, 10),
            edges_per_step=5,
        )
        for k in (3, 5, 10):
            self.assertIn(k, result, f"Missing key {k}")

    def test_rates_in_unit_interval(self):
        from tacet.experimental.dynamics.eval import eval_rollout_coherence

        result = eval_rollout_coherence(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            self.onto,
            ks=(3, 5, 10),
            edges_per_step=5,
        )
        for k, rate in result.items():
            self.assertGreaterEqual(rate, 0.0, f"rate[{k}] < 0")
            self.assertLessEqual(rate, 1.0, f"rate[{k}] > 1")


@unittest.skipUnless(HAS_TORCH, "torch missing")
class TestEvalLatency(unittest.TestCase):
    def setUp(self):
        self.traj, self.event_types = _make_tiny_trajectory()
        (self.kge, self.state_enc, self.event_enc, self.rssm, self.decoder) = _build_components(
            self.traj, self.event_types
        )

    def test_layer4_ms_positive(self):
        from tacet.experimental.dynamics.eval import eval_latency

        result = eval_latency(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            n_queries=5,
        )
        self.assertIn("layer4_ms", result)
        self.assertGreater(result["layer4_ms"], 0.0)

    def test_tier2_ms_and_ratio_when_provided(self):
        from tacet.experimental.dynamics.eval import eval_latency

        call_count = [0]

        def fake_tier2():
            call_count[0] += 1

        result = eval_latency(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            tier2_call=fake_tier2,
            n_queries=5,
        )
        self.assertIn("tier2_ms", result)
        self.assertIn("ratio", result)
        self.assertGreaterEqual(result["tier2_ms"], 0.0)

    def test_tier2_none_returns_none_fields(self):
        from tacet.experimental.dynamics.eval import eval_latency

        result = eval_latency(
            self.traj,
            self.kge,
            self.state_enc,
            self.event_enc,
            self.rssm,
            self.decoder,
            tier2_call=None,
            n_queries=5,
        )
        self.assertIsNone(result["tier2_ms"])
        self.assertIsNone(result["ratio"])
