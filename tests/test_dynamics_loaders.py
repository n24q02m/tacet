import unittest
from pathlib import Path

ICEWS_ROOT = Path("data/ICEWS14")
HAVE_DATA = (ICEWS_ROOT / "train.txt").exists()


@unittest.skipUnless(HAVE_DATA, "ICEWS14 not seeded locally")
class TestICEWSLoader(unittest.TestCase):
    def test_load_trajectory_train_split(self):
        from tacet.experimental.dynamics.loaders import load_icews14_trajectory

        traj = load_icews14_trajectory(ICEWS_ROOT, split="train")
        self.assertGreater(len(traj), 100)
        self.assertEqual(len(traj.snapshots), len(traj.event_batches) + 1)
        # Snapshots are cumulative, not per-day deltas: later state must
        # contain at least as many edges as an earlier state.
        mid = len(traj) // 2
        self.assertGreaterEqual(len(traj.at(len(traj)).edges), len(traj.at(mid).edges))
