"""Dataset loaders for Layer 4 — ICEWS14 first, others later."""

from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path
from typing import Literal

from tacet.core.graph import WorldGraph
from tacet.experimental.dynamics.events import Event, EventBatch
from tacet.experimental.dynamics.trajectory import Trajectory


def load_icews14_trajectory(
    root: str | Path,
    split: Literal["train", "valid", "test"] = "train",
) -> Trajectory:
    """Load ICEWS14 quadruples and group by day into a Trajectory.

    Each snapshot G_t is the FULL cumulative world state at day t (all edges
    with valid_from <= t), as required by Layer 4 dynamics G_{t+1} = f(G_t, e):
    the downstream encoder pools embeddings of every active node in G_t.

    Performance note: the O(n^2) trap in the naive spec is calling
    WorldGraph.copy() (O(edges)) once per day on a cumulatively growing graph.
    add_edge() deduplication is an O(1) set-membership check, so building one
    cumulative graph from all ~72k quads is O(n).  We then derive each day's
    cumulative snapshot with WorldGraph.slice_at(t) (one linear pass over edges
    per day), which is fast enough for the 365-day ICEWS14 span.
    """
    path = Path(root) / f"{split}.txt"
    quads: list[tuple[str, str, str, float]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            s, r, o, date = parts[:4]
            t = _date_to_float(date)
            quads.append((s, r, o, t))
    quads.sort(key=lambda q: q[3])
    return _build_trajectory(quads)


def _date_to_float(date_str: str) -> float:
    """YYYY-MM-DD -> day index since 2014-01-01."""
    d = datetime.date.fromisoformat(date_str)
    epoch = datetime.date(2014, 1, 1)
    return float((d - epoch).days)


def _build_trajectory(quads: list[tuple[str, str, str, float]]) -> Trajectory:
    """Group by timestamp -> cumulative snapshots derived from one full graph.

    snapshot[0] is the empty initial state; snapshot[k+1] is the FULL world
    state after batch[k], i.e. all edges valid at that day, obtained via
    full.slice_at(t).  This satisfies len(snapshots) == len(batches) + 1 and
    gives the cumulative semantics Layer 4 dynamics requires.
    """
    if not quads:
        return Trajectory(snapshots=[WorldGraph(name="empty")], event_batches=[])

    # Group by day using plain dict — no WorldGraph overhead here
    by_day: dict[float, list[tuple[str, str, str]]] = defaultdict(list)
    for s, r, o, t in quads:
        by_day[t].append((s, r, o))

    # Build ONE cumulative bi-temporal graph (O(n): each unique triple is an
    # O(1) set-membership dedup + append).
    full = WorldGraph(name="icews14_full")
    for s, r, o, t in quads:
        full.add_edge(s, r, o, valid_from=t, valid_to=None)

    snapshots: list[WorldGraph] = [WorldGraph(name="t_init")]
    batches: list[EventBatch] = []

    for t in sorted(by_day):
        events = [Event(timestamp=t, type=r, actor=s, target=o) for s, r, o in by_day[t]]
        batches.append(EventBatch(timestamp=t, events=events))
        # Full cumulative state at day t — all edges with valid_from <= t.
        snapshots.append(full.slice_at(t))

    return Trajectory(snapshots=snapshots, event_batches=batches)
