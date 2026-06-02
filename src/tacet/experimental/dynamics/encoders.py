"""Graph + event encoders for Layer 4."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from tacet.core.graph import WorldGraph
from tacet.experimental.dynamics.events import EventBatch


class GraphStateEncoder(Protocol):
    def encode(self, graph: WorldGraph) -> np.ndarray: ...


class EventEncoder(Protocol):
    def encode(self, batch: EventBatch) -> np.ndarray: ...


class ComplexN3PooledEncoder:
    """Mean-pool of (re || im) entity embeddings from a trained TorchComplEx."""

    def __init__(self, kge) -> None:  # noqa: ANN001
        self.kge = kge

    def encode(self, graph: WorldGraph) -> np.ndarray:
        import torch

        ents = sorted({e.source for e in graph.edges} | {e.target for e in graph.edges})
        ids = [self.kge.ent[e] for e in ents if e in self.kge.ent]
        if not ids:
            d = self.kge.cfg.dim
            return np.zeros(2 * d, dtype=np.float32)
        idx = torch.tensor(ids, device=self.kge.device)
        re = self.kge._E_re[idx].mean(dim=0).detach().cpu().numpy()
        im = self.kge._E_im[idx].mean(dim=0).detach().cpu().numpy()
        return np.concatenate([re, im]).astype(np.float32)


class BagOfTypesEventEncoder:
    """Count occurrences of each event type in a batch."""

    def __init__(self, event_types: list[str]) -> None:
        self.event_types = event_types
        self._idx = {t: i for i, t in enumerate(event_types)}
        self.dim = len(event_types)

    def encode(self, batch: EventBatch) -> np.ndarray:
        v = np.zeros(len(self.event_types), dtype=np.float32)
        for e in batch.events:
            j = self._idx.get(e.type)
            if j is not None:
                v[j] += 1.0
        return v


class ActionAsNodeEventEncoder:
    """Action-as-node event encoding (Feng et al. GWM): the event-type
    histogram concatenated with the mean of (actor_re || target_re) ComplEx
    embeddings of the entities each event touches.

    Bag-of-types alone tells the model WHAT kind of events occurred but not
    WHICH entities they touched; grounding each event in its actor/target
    embeddings gives the dynamics the entity-level signal it needs to predict
    *which* edges appear next.
    """

    def __init__(self, event_types: list[str], kge) -> None:  # noqa: ANN001
        self.event_types = event_types
        self._idx = {t: i for i, t in enumerate(event_types)}
        self.kge = kge
        self.dim = len(event_types) + 2 * kge.cfg.dim

    def encode(self, batch: EventBatch) -> np.ndarray:
        n_t = len(self.event_types)
        d = self.kge.cfg.dim
        hist = np.zeros(n_t, dtype=np.float32)
        acc = np.zeros(2 * d, dtype=np.float32)
        cnt = 0
        e_re = self.kge._E_re  # noqa: SLF001
        e_re = e_re.detach().cpu().numpy() if hasattr(e_re, "detach") else np.asarray(e_re)
        for ev in batch.events:
            j = self._idx.get(ev.type)
            if j is not None:
                hist[j] += 1.0
            actor = (
                e_re[self.kge.ent[ev.actor]]
                if ev.actor in self.kge.ent
                else np.zeros(d, dtype=np.float32)
            )
            target = (
                e_re[self.kge.ent[ev.target]]
                if ev.target is not None and ev.target in self.kge.ent
                else np.zeros(d, dtype=np.float32)
            )
            acc += np.concatenate([actor, target]).astype(np.float32)
            cnt += 1
        if cnt:
            acc /= cnt
        return np.concatenate([hist, acc]).astype(np.float32)
