from dataclasses import FrozenInstanceError

import pytest

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
    assert e.actor == "Country_A"
    assert e.target == "Country_B"
    assert e.payload["intensity"] == 0.8


def test_event_optional_target():
    e = Event(timestamp=1.0, type="birth", actor="Person_A")
    assert e.target is None
    assert e.payload == {}


def test_event_immutability():
    e = Event(timestamp=1.0, type="visit", actor="A")
    with pytest.raises(FrozenInstanceError):
        e.timestamp = 2.0  # type: ignore


def test_event_equality_ignores_payload():
    e1 = Event(timestamp=1.0, type="visit", actor="A", payload={"a": 1})
    e2 = Event(timestamp=1.0, type="visit", actor="A", payload={"a": 2})
    assert e1 == e2


def test_event_hash_ignores_payload():
    e1 = Event(timestamp=1.0, type="visit", actor="A", payload={"a": 1})
    e2 = Event(timestamp=1.0, type="visit", actor="A", payload={"a": 2})
    assert hash(e1) == hash(e2)


def test_event_batch_construction():
    e1 = Event(timestamp=1.0, type="visit", actor="A", target="B")
    e2 = Event(timestamp=1.0, type="trade", actor="A", target="C")
    batch = EventBatch(timestamp=1.0, events=[e1, e2])
    assert len(batch.events) == 2
    assert batch.timestamp == 1.0
    assert batch.events[0] == e1
    assert batch.events[1] == e2


def test_event_batch_mutability():
    batch = EventBatch(timestamp=1.0, events=[])
    batch.timestamp = 2.0
    assert batch.timestamp == 2.0
    e1 = Event(timestamp=2.0, type="visit", actor="A")
    batch.events.append(e1)
    assert len(batch.events) == 1
