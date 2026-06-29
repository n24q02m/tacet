import unittest
from pathlib import Path

from tacet.experimental.dynamics.loaders import (
    _build_trajectory,
    _date_to_float,
    load_icews14_trajectory,
)

ICEWS_ROOT = Path("data/ICEWS14")
HAVE_DATA = (ICEWS_ROOT / "train.txt").exists()


def test_date_to_float():
    # 2014-01-01 is epoch (0.0)
    assert _date_to_float("2014-01-01") == 0.0
    # 2014-01-02 is 1.0
    assert _date_to_float("2014-01-02") == 1.0
    # Leap year check (2014 is not, but 2016 is; ICEWS14 is only 2014)
    assert _date_to_float("2014-02-01") == 31.0


def test_build_trajectory_empty():
    traj = _build_trajectory([])
    assert len(traj.snapshots) == 1
    assert traj.snapshots[0].name == "empty"
    assert len(traj.event_batches) == 0


def test_build_trajectory_basic():
    # s, r, o, t
    quads = [
        ("S1", "R1", "O1", 0.0),
        ("S1", "R2", "O2", 0.0),
        ("S2", "R1", "O3", 1.0),
    ]
    traj = _build_trajectory(quads)

    # len(snapshots) == len(batches) + 1
    # batches for t=0.0 and t=1.0
    assert len(traj.event_batches) == 2
    assert len(traj.snapshots) == 3

    # Initial snapshot
    assert len(traj.snapshots[0].edges) == 0

    # Snapshot 1 (after batch 0 at t=0.0)
    assert len(traj.snapshots[1].edges) == 2

    # Snapshot 2 (after batch 1 at t=1.0)
    assert len(traj.snapshots[2].edges) == 3


def test_load_icews14_trajectory_mock(tmp_path):
    # Create a mock ICEWS14 dataset
    data_dir = tmp_path / "ICEWS14"
    data_dir.mkdir()
    train_file = data_dir / "train.txt"

    # format: s \t r \t o \t YYYY-MM-DD
    content = "A\tr1\tB\t2014-01-01\nB\tr2\tC\t2014-01-02\n"
    train_file.write_text(content, encoding="utf-8")

    traj = load_icews14_trajectory(data_dir, split="train")

    assert len(traj.event_batches) == 2
    assert len(traj.snapshots) == 3
    assert traj.event_batches[0].timestamp == 0.0
    assert traj.event_batches[1].timestamp == 1.0


@unittest.skipUnless(HAVE_DATA, "ICEWS14 not seeded locally")
class TestICEWSLoader(unittest.TestCase):
    def test_load_trajectory_train_split(self):
        traj = load_icews14_trajectory(ICEWS_ROOT, split="train")
        self.assertGreater(len(traj), 100)
        self.assertEqual(len(traj.snapshots), len(traj.event_batches) + 1)
        # Snapshots are cumulative, not per-day deltas: later state must
        # contain at least as many edges as an earlier state.
        mid = len(traj) // 2
        self.assertGreaterEqual(len(traj.at(len(traj)).edges), len(traj.at(mid).edges))
