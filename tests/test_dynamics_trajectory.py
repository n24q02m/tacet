from tacet.core.graph import WorldGraph
from tacet.experimental.dynamics.events import Event, EventBatch
from tacet.experimental.dynamics.trajectory import Trajectory, apply_event


def test_apply_event_inserts_edge_with_validity():
    g = WorldGraph(name="test")
    g.add_node("A", "Country")
    g.add_node("B", "Country")
    e = Event(timestamp=5.0, type="visits", actor="A", target="B")
    g2 = apply_event(g, e)
    triples = [(edge.source, edge.relation, edge.target) for edge in g2.edges]
    assert ("A", "visits", "B") in triples


def test_apply_event_null_target():
    g = WorldGraph(name="test")
    g.add_node("A", "Country")
    # target=None should fallback to actor
    e = Event(timestamp=5.0, type="self_reflect", actor="A", target=None)
    g2 = apply_event(g, e)
    triples = [(edge.source, edge.relation, edge.target) for edge in g2.edges]
    assert ("A", "self_reflect", "A") in triples


def test_trajectory_transition_pair():
    g0 = WorldGraph(name="t0")
    g1 = WorldGraph(name="t1")
    batch = EventBatch(timestamp=1.0, events=[])
    traj = Trajectory(snapshots=[g0, g1], event_batches=[batch])
    assert len(traj) == 1
    s, b, ns = traj.transition(0)
    assert s is g0 and ns is g1 and b is batch


def test_trajectory_indexing():
    g0 = WorldGraph(name="t0")
    g1 = WorldGraph(name="t1")
    batch0 = EventBatch(timestamp=1.0, events=[])
    traj = Trajectory(snapshots=[g0, g1], event_batches=[batch0])

    assert traj.at(0) is g0
    assert traj.at(1) is g1
    assert traj.event_at(0) is batch0
