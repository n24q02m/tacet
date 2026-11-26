"""Tests for the MetaQA dataset loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tacet.data.metaqa import _parse_question_line, load_metaqa


class TestMetaQALoader(unittest.TestCase):
    def test_parse_question_line(self) -> None:
        q = _parse_question_line("Who starred in [Inception]?\tLeonardo DiCaprio|Tom Hardy", 1)
        assert q is not None
        self.assertEqual(q.head, "Inception")
        self.assertEqual(q.answers, ["Leonardo DiCaprio", "Tom Hardy"])
        self.assertEqual(q.hop, 1)

    def test_parse_handles_missing_topic(self) -> None:
        self.assertIsNone(_parse_question_line("no topic\tanswer", 1))

    def test_parse_handles_blank(self) -> None:
        self.assertIsNone(_parse_question_line("   ", 1))

    def test_loader_raises_on_missing_root(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_metaqa("/nonexistent/path", hop=1)

    def test_loader_reads_a_synthetic_layout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "kb.txt").write_text(
                "Inception|directed_by|Christopher Nolan\n"
                "Inception|starred_actors|Leonardo DiCaprio\n",
                encoding="utf-8",
            )
            hop_dir = root / "1-hop"
            hop_dir.mkdir()
            (hop_dir / "qa_test.txt").write_text(
                "who directed [Inception]?\tChristopher Nolan\n", encoding="utf-8"
            )
            bench = load_metaqa(root, hop=1, split="test")
            self.assertEqual(bench.stats()["kg_triples"], 2)
            self.assertEqual(len(bench.questions), 1)
            self.assertEqual(bench.questions[0].answers, ["Christopher Nolan"])


if __name__ == "__main__":
    unittest.main()
