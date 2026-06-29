"""Tests for episodic memory and feedback-driven curation."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from tacet.experimental.episodic import Episode, EpisodicStore, FeedbackCurator, RuleScore


class TestEpisode(unittest.TestCase):
    def test_feedback_marking(self) -> None:
        ep = Episode(
            id=0,
            timestamp=time.time(),
            head="a",
            relation="r",
            tier=1,
            answers=["b"],
            cost=0.1,
            latency_ms=10.0,
        )
        self.assertEqual(ep.feedback, {})

        ep.mark_correct(by="tester")
        self.assertTrue(ep.feedback["correct"])
        self.assertEqual(ep.feedback["by"], "tester")
        self.assertIn("at", ep.feedback)

        ep.mark_wrong(by="reviewer", reason="incorrect")
        self.assertFalse(ep.feedback["correct"])
        self.assertEqual(ep.feedback["by"], "reviewer")
        self.assertEqual(ep.feedback["reason"], "incorrect")


class TestEpisodicStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = EpisodicStore()

    def test_record_and_len(self) -> None:
        self.assertEqual(len(self.store), 0)
        ep = self.store.record("a", "r", 1, ["b"], 0.1, 10.0, note="test", proof_rules=["rule1"])
        self.assertEqual(len(self.store), 1)
        self.assertEqual(ep.id, 0)
        self.assertEqual(ep.head, "a")
        self.assertEqual(ep.proof_rules, ["rule1"])

    def test_querying(self) -> None:
        t0 = time.time()
        self.store.record("a", "r1", 1, ["b"], 0.1, 10.0)
        time.sleep(0.01)
        t1 = time.time()
        self.store.record("a", "r2", 2, ["c"], 0.2, 20.0)
        self.store.record("x", "r1", 1, ["y"], 0.1, 10.0)

        self.assertEqual(len(self.store.all()), 3)
        self.assertEqual(len(self.store.for_query("a", "r1")), 1)
        self.assertEqual(len(self.store.for_query("a", "r2")), 1)

        # Window query
        window = self.store.in_window(t0, t1)
        self.assertEqual(len(window), 1)
        self.assertEqual(window[0].relation, "r1")

        # Feedback query
        self.assertEqual(len(self.store.with_feedback()), 0)
        self.store.all()[0].mark_correct()
        self.assertEqual(len(self.store.with_feedback()), 1)

    def test_persistence(self) -> None:
        self.store.record("a", "r", 1, ["b"], 0.1, 10.0)
        self.store.record("c", "d", 2, ["e"], 0.2, 20.0)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.store.save_jsonl(tmp_path)

            new_store = EpisodicStore()
            new_store.load_jsonl(tmp_path)

            self.assertEqual(len(new_store), 2)
            self.assertEqual(new_store.all()[0].head, "a")
            self.assertEqual(new_store.all()[1].head, "c")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_summary(self) -> None:
        # Empty summary
        self.assertEqual(self.store.summary(), {"queries": 0})

        # Populated summary
        ep1 = self.store.record("a", "r", 1, ["b"], 0.1, 10.0)
        self.store.record("c", "d", 2, ["e"], 0.3, 30.0)

        ep1.mark_correct()

        summary = self.store.summary()
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["tier_counts"], {1: 1, 2: 1})
        self.assertEqual(summary["avg_cost"], 0.2)
        self.assertEqual(summary["avg_latency_ms"], 20.0)
        self.assertEqual(summary["feedback_received"], 1)
        self.assertEqual(summary["feedback_accuracy"], 1.0)


class TestRuleScore(unittest.TestCase):
    def test_trust_score(self) -> None:
        rs = RuleScore("rule1")
        self.assertEqual(rs.trust, 0.5)  # Initial trust

        rs.positives = 1
        # n=1, p=1.0. trust = max(0, 1.0 - 1/3) = 0.666...
        self.assertAlmostEqual(rs.trust, 2 / 3)

        rs.negatives = 1
        # n=2, p=0.5. trust = max(0, 0.5 - 1/4) = 0.25
        self.assertEqual(rs.trust, 0.25)

        rs.positives = 100
        rs.negatives = 0
        # n=100, p=1.0. trust = 1.0 - 1/102 ~= 0.99
        self.assertGreater(rs.trust, 0.9)


class TestFeedbackCurator(unittest.TestCase):
    def test_curation_flow(self) -> None:
        curator = FeedbackCurator(retire_below=0.4, min_observations=2)

        # Create some episodes with feedback
        ep1 = Episode(0, 0.0, "a", "r", 1, ["b"], 0.1, 10.0, proof_rules=["r1", "r2"])
        ep1.mark_correct()

        ep2 = Episode(1, 0.0, "c", "d", 1, ["e"], 0.1, 10.0, proof_rules=["r1"])
        ep2.mark_wrong()

        ep3 = Episode(2, 0.0, "f", "g", 1, ["h"], 0.1, 10.0, proof_rules=["r2"])
        ep3.mark_wrong()

        curator.absorb([ep1, ep2, ep3])

        # r1: 1 pos, 1 neg -> n=2, trust=0.25
        # r2: 1 pos, 1 neg -> n=2, trust=0.25

        self.assertIn("r1", curator.scores)
        self.assertIn("r2", curator.scores)

        self.assertEqual(curator.rules_to_retire(), ["r1", "r2"])

        # Add more positive feedback for r1
        ep4 = Episode(3, 0.0, "i", "j", 1, ["k"], 0.1, 10.0, proof_rules=["r1"])
        ep4.mark_correct()
        ep5 = Episode(4, 0.0, "l", "m", 1, ["n"], 0.1, 10.0, proof_rules=["r1"])
        ep5.mark_correct()

        curator.absorb([ep4, ep5])
        # r1: 3 pos, 1 neg -> n=4, trust = 3/4 - 1/6 = 0.75 - 0.166 = 0.5833

        self.assertNotIn("r1", curator.rules_to_retire())
        self.assertIn("r1", curator.trusted_rules(threshold=0.5))


if __name__ == "__main__":
    unittest.main()
