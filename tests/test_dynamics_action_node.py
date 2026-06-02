import types

import numpy as np

from tacet.experimental.dynamics.encoders import ActionAsNodeEventEncoder
from tacet.experimental.dynamics.events import Event, EventBatch


class FakeKGE:
    def __init__(self):
        self.ent = {"A": 0, "B": 1}
        self.cfg = types.SimpleNamespace(dim=2)
        self._E_re = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_action_as_node_dim_and_entity_signal():
    enc = ActionAsNodeEventEncoder(["t1"], FakeKGE())
    assert enc.dim == 5  # n_types(1) + 2*kge_dim(2)
    b = EventBatch(timestamp=0.0, events=[Event(0.0, "t1", "A", "B")])
    v = enc.encode(b)
    assert v.shape == (5,)
    assert v[0] == 1.0  # type histogram
    # entity part = mean of (actor_re || target_re) = ([1,0] || [0,1])
    assert np.allclose(v[1:], [1.0, 0.0, 0.0, 1.0])


def test_action_as_node_unknown_entity_zero():
    enc = ActionAsNodeEventEncoder(["t1"], FakeKGE())
    b = EventBatch(timestamp=0.0, events=[Event(0.0, "t1", "Z", None)])
    v = enc.encode(b)
    # unknown actor + no target -> entity part all zeros
    assert np.allclose(v[1:], [0.0, 0.0, 0.0, 0.0])
