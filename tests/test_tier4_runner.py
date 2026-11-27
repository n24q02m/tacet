"""Unit and dry-run end-to-end tests for Tier-4 Wave B QLoRA runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

from tacet.core.graph import WorldGraph
from tacet.distill.tier4_runner import (
    BASE_RELATIONS,
    DERIVED_RELATIONS,
    QLoRAConfig,
    _f1_sets,
    build_engines_for_split,
    build_sft_prompt,
    build_unanswerable_eval_queries,
    extract_local_context,
    format_sft_target,
    parse_model_output,
    run_wave_b_matrix,
    summarize_wave_b_jsonl,
)
from tacet.eval.unseen import UnseenBenchmarkConfig, generate_unseen_benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import run_tier4_qlora  # noqa: E402


class TestContextAndFormatting:
    def test_extract_local_context_forbids_derived_and_target_relations(self) -> None:
        g = WorldGraph(name="leak-test")
        g.add_edge("p1", "parent_of", "p2")
        g.add_edge("p2", "parent_of", "p3")
        g.add_edge("p1", "works_at", "c1")
        g.add_edge("p4", "works_at", "c1")
        g.add_edge("p1", "ancestor_of", "p2")
        g.add_edge("p1", "favourite_food", "food_0")
        g.add_edge("p1", "manages", "p5")

        ctx = extract_local_context(g, "p1", "ancestor_of", max_hops=2, max_facts=50)
        rels = {r for _, r, _ in ctx}
        assert rels <= BASE_RELATIONS
        assert rels.isdisjoint(DERIVED_RELATIONS)
        assert ("p4", "works_at", "c1") in ctx
        assert ("p2", "parent_of", "p3") in ctx

        ctx_no_parent = extract_local_context(g, "p1", "parent_of", max_hops=1)
        assert all(r != "parent_of" for _, r, _ in ctx_no_parent)

    def test_prompt_target_and_parser_across_arms(self) -> None:
        ctx = [("p1", "parent_of", "p2")]
        p_a = build_sft_prompt("p1", "ancestor_of", ctx, arm="A")
        p_b = build_sft_prompt("p1", "ancestor_of", ctx, arm="B")
        p_c = build_sft_prompt("p1", "ancestor_of", ctx, arm="C")
        assert '"answers"' in p_a
        assert "FACT/DERIVED" in p_b
        assert '"abstain"' in p_c

        t_a = format_sft_target(
            {"answers": ["p2"], "proof": ["FACT p1 -parent_of-> p2"], "abstain": False},
            "A",
        )
        assert t_a == '{"answers": ["p2"]}'
        assert parse_model_output(t_a) == (["p2"], [], False)

        t_b = format_sft_target(
            {"answers": ["p2"], "proof": ["DERIVED p1 -ancestor_of-> p2 [r]"]},
            "B",
        )
        ans_b, proof_b, abst_b = parse_model_output("<think>noise</think>\n" + t_b)
        assert ans_b == ["p2"]
        assert proof_b == ["DERIVED p1 -ancestor_of-> p2 [r]"]
        assert abst_b is False

        t_c = format_sft_target({"answers": [], "proof": [], "abstain": True}, "C")
        ans_c, proof_c, abst_c = parse_model_output(t_c)
        assert ans_c == []
        assert proof_c == []
        assert abst_c is True

        # String answer / malformed JSON fallback
        assert parse_model_output('{"answers": "p2"}') == (["p2"], [], False)
        assert parse_model_output("{bad json} no answers") == ([], [], False)

    def test_f1_sets_edge_cases(self) -> None:
        assert _f1_sets([], []) == 1.0
        assert _f1_sets(["a"], []) == 0.0
        assert _f1_sets(["a"], ["b"]) == 0.0
        assert _f1_sets(["a", "b"], ["b", "c"]) == pytest.approx(0.5)


class TestEnginesAndDryRunPipeline:
    def test_build_engines_disjoint_and_unanswerable_queries(self) -> None:
        split = generate_unseen_benchmark("unseen_entity", UnseenBenchmarkConfig(seed=0))
        train_engine, eval_engine = build_engines_for_split(split, seed=0)

        for h, _, t in train_engine.closure:
            assert h not in split.eval_entities
            assert t not in split.eval_entities

        eval_rule_names = {r.name for r in eval_engine.rules}
        assert any("superior_of" in name for name in eval_rule_names)
        assert any("colleague_of" in name for name in eval_rule_names)

        unans = build_unanswerable_eval_queries(split, eval_engine, max_queries=15)
        assert len(unans) <= 15
        for h, r in unans:
            assert h in split.eval_entities
            assert not split.truth.get((h, r))
            assert not eval_engine.query(h, r).answered

    def test_dry_run_matrix_resume_and_polars_summary(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "wave_b"
        manifest = run_wave_b_matrix(
            arms=["A", "B", "C"],
            seeds=[0],
            output_dir=out_dir,
            cfg=QLoRAConfig(epochs=1),
            dry_run=True,
            max_train_samples=40,
            max_eval_samples=20,
        )
        assert manifest["dry_run"] is True
        assert len(manifest["runs"]) == 3
        for run_row in manifest["runs"]:
            assert 0.0 <= run_row["deductive_entailment_rate"] <= 1.0
            assert 0.0 <= run_row["overall_exact_match"] <= 1.0
            assert 0.0 <= run_row["abstain_f1"] <= 1.0
            assert 0.0 <= run_row["false_answer_rate"] <= 1.0

        df = summarize_wave_b_jsonl(manifest["runs"])
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert set(df["arm"].to_list()) == {"A", "B", "C"}
        assert summarize_wave_b_jsonl([]).height == 0

        # Resume path skips already completed runs without recomputing
        pred_b = out_dir / "predictions_arm_B_s0.jsonl"
        mtime_before = pred_b.stat().st_mtime_ns
        rc = run_tier4_qlora.main(
            [
                "--arms",
                "A",
                "B",
                "C",
                "--seeds",
                "0",
                "--output-dir",
                str(out_dir),
                "--dry-run",
                "--max-train-samples",
                "40",
                "--max-eval-samples",
                "20",
            ]
        )
        assert rc == 0
        assert pred_b.stat().st_mtime_ns == mtime_before
        stored = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(stored["runs"]) == 3
