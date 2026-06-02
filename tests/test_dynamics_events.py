from tacet.experimental.dynamics.events import Event, EventBatch


def test_event_construction():
    e = Event(
        timestamp=1.0,
        type="diplomatic_visit",
        actor="Country_A",
        target="Country_B",
        payload={"intensity": 0.8},
    )
    assert e.timestamp == 1.0
    assert e.type == "diplomatic_visit"
    assert e.payload["intensity"] == 0.8


def test_event_batch_groups_by_timestamp():
    e1 = Event(timestamp=1.0, type="visit", actor="A", target="B")
    e2 = Event(timestamp=1.0, type="trade", actor="A", target="C")
    batch = EventBatch(timestamp=1.0, events=[e1, e2])
    assert len(batch.events) == 2
    assert batch.timestamp == 1.0
