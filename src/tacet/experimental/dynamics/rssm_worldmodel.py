"""RSSMWorldModel — the real ``WorldModel`` implementation (Layer 4, P5).

The ``tacet.experimental.worldmodel`` module ships only the ``WorldModel`` Protocol and a
deterministic ``IdentityWorldModel`` stand-in, deferring "a real continuous
world model" to future work.  This module provides it: the trained Layer 4
RSSM, adapted to the Protocol so the discrete substrate's planner
(``tacet.experimental.agent.Planner``) can choose plans by their *imagined* future.

State ``S`` is the RSSM latent ``(h, z)`` (a pair of torch tensors); action
``A`` is an :class:`~tacet.experimental.dynamics.events.EventBatch` (the tick's events).
``observe`` encodes a graph into the posterior latent; ``predict`` rolls the
prior one step under an event; ``rollout`` imagines a plan forward and scores
each step by the number of ontology-consistent top edges the decoder predicts
(a cheap, sound proxy reward the planner can maximise).
"""

from __future__ import annotations

from tacet.experimental.dynamics.events import EventBatch
from tacet.experimental.dynamics.trajectory import Trajectory as _DynTrajectory  # noqa: F401
from tacet.experimental.worldmodel import Trajectory


class RSSMWorldModel:
    """Adapt a trained Layer 4 RSSM to the ``tacet.experimental.worldmodel.WorldModel``
    Protocol so a planner can imagine futures with learned graph dynamics."""

    def __init__(
        self,
        rssm,
        decoder,
        state_enc,
        event_enc,
        kge,  # noqa: ANN001
        ontology=None,
        edges_per_step: int = 50,
    ) -> None:
        self.rssm = rssm
        self.decoder = decoder
        self.state_enc = state_enc
        self.event_enc = event_enc
        self.kge = kge
        self.ontology = ontology
        self.edges_per_step = edges_per_step
        self._device = kge.device
        # typing reference for the ontology check (set by observe); the
        # ontology types entities against this graph, so it must be a real
        # populated graph, not an empty one.
        self._ref_graph = None

    def observe(self, observation) -> tuple:  # noqa: ANN001
        """Encode a ``WorldGraph`` into the posterior latent ``(h, z)``."""
        import torch

        self._ref_graph = observation
        h, z = self.rssm.initial(1, device=self._device)
        s = torch.from_numpy(self.state_enc.encode(observation)).unsqueeze(0).to(self._device)
        e0 = torch.zeros(1, self.rssm.cfg.event_in_dim, device=self._device)
        with torch.no_grad():
            z, h = self.rssm.posterior(h, z, e0, s)
        return (h, z)

    def predict(self, state: tuple, action) -> tuple:  # noqa: ANN001
        """Roll the prior one step under ``action`` (an EventBatch)."""
        import torch

        h, z = state
        e = self._encode_action(action)
        with torch.no_grad():
            z2, h2 = self.rssm.prior(h, z, e)
        return (h2, z2)

    def rollout(self, state: tuple, plan: list) -> Trajectory:  # noqa: ANN001
        """Imagine ``plan`` forward; reward = #ontology-consistent top edges."""
        states = [state]
        rewards: list[float] = []
        cur = state
        for action in plan:
            cur = self.predict(cur, action)
            states.append(cur)
            rewards.append(self._consistent_edge_reward(cur))
        return Trajectory(states=states, actions=list(plan), rewards=rewards)

    # -- helpers -----------------------------------------------------------
    def _encode_action(self, action):  # noqa: ANN001
        import torch

        if action is None:
            return torch.zeros(1, self.rssm.cfg.event_in_dim, device=self._device)
        batch = (
            action
            if isinstance(action, EventBatch)
            else EventBatch(timestamp=0.0, events=action if isinstance(action, list) else [action])
        )
        return torch.from_numpy(self.event_enc.encode(batch)).unsqueeze(0).to(self._device)

    def _consistent_edge_reward(self, state: tuple) -> float:
        """Decode top edges at ``state`` and count those the ontology admits.

        A sound, cheap proxy: a good imagined step proposes edges that are
        type-consistent with the ontology.  Returns 0.0 if no ontology given.
        """
        import numpy as np
        import torch

        if self.ontology is None:
            return 0.0
        _, z = state
        ent = list(self.kge.ent.keys())
        rel = list(self.kge.rel.keys())
        n = min(self.edges_per_step * 10, len(ent) * len(rel))
        if n == 0:
            return 0.0
        rng = np.random.default_rng(0)
        si = rng.integers(0, len(ent), n)
        ri = rng.integers(0, len(rel), n)
        oi = rng.integers(0, len(ent), n)
        with torch.no_grad():
            s_e = self.kge._E_re[torch.from_numpy(si).to(self._device)]  # noqa: SLF001
            r_e = self.kge._R_re[torch.from_numpy(ri).to(self._device)]  # noqa: SLF001
            o_e = self.kge._E_re[torch.from_numpy(oi).to(self._device)]  # noqa: SLF001
            sc = self.decoder(z.expand(n, -1), s_e, r_e, o_e).cpu().numpy()
        top = np.argsort(sc)[-min(self.edges_per_step, n) :]
        ok = 0
        from tacet.core.graph import WorldGraph

        # use the observed graph as the typing reference (entities carry their
        # types there); an empty graph would reject every edge.
        ref = self._ref_graph if self._ref_graph is not None else WorldGraph(name="_imagined")
        for idx in top:
            try:
                if self.ontology.allows(
                    ref, ent[int(si[idx])], rel[int(ri[idx])], ent[int(oi[idx])]
                ):
                    ok += 1
            except Exception:  # noqa: BLE001
                pass
        return ok / len(top) if len(top) else 0.0


__all__ = ["RSSMWorldModel"]
