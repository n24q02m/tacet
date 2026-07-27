"""Tests for the E16 forbid-target-in-body aggregation.

All fixtures are TINY and HAND-BUILT: the on-disk artifact
(``experiments/results/e16_forbid_target_hop2.json``) is NEVER read here.

What these pin is the DEFINITION, not the numbers. Strict review round 6 found
that the published E16 counts were right by accident of the data rather than by
the classifier: the third rule class happens to be empty on that grid, so folding
it into "junk" gave the same totals. On a workload that produces near-functional
leakage the two definitions diverge, and the leaky rule would be reported as junk
that ``--forbid-target-in-body`` had removed -- which it cannot do. The classifier
test below exercises exactly the case the real grid does not.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from aggregate_e16_forbid import (  # noqa: E402
    DEFAULT_DATASET,
    classify,
    match_key,
    parse_cell,
    read_forbid,
)

TARGET = "q2_directors_of_movies_acted_in_by"
TRUE_RULE = f"syn:{TARGET}<=~starred_actors.directed_by"
SELF_REF = f"syn:{TARGET}<={TARGET}.{TARGET}"
SELF_REF_INV = f"syn:{TARGET}<=~{TARGET}.{TARGET}"
# Base relations only, no mention of the target, but not the composition either:
# the near-functional leakage the paper scopes out of E16 explicitly.
LEAKAGE = f"syn:{TARGET}<=~written_by.directed_by"


# -------------------------------------------------------------------- classify
def test_classify_separates_leakage_from_self_reference():
    """The three classes are distinct, and leakage is NOT counted as self-referential.

    This is the case the published grid never produced. Were the third class
    folded back into "junk", the leaky rule would be reported as junk removed by
    an option that only removes rules naming the target.
    """
    true_installed, self_ref, other = classify([TRUE_RULE, SELF_REF, LEAKAGE])
    assert true_installed is True
    assert self_ref == 1
    assert other == 1


def test_classify_counts_both_self_referential_forms():
    """Direct and inverse-first bodies naming the target both count as self-referential."""
    true_installed, self_ref, other = classify([SELF_REF, SELF_REF_INV])
    assert true_installed is False
    assert self_ref == 2
    assert other == 0


def test_classify_leakage_alone_is_not_an_install_and_not_self_referential():
    """A leaky rule on its own: no install, nothing for the forbid option to remove."""
    assert classify([LEAKAGE]) == (False, 0, 1)


def test_classify_empty_rule_set():
    """A cell that installed nothing is all zeroes, not an install."""
    assert classify([]) == (False, 0, 0)


def test_classify_composition_only_leaves_both_junk_classes_empty():
    """The healthy outcome: the composition installed and nothing else."""
    assert classify([TRUE_RULE]) == (True, 0, 0)


def test_classify_takes_the_composition_from_the_caller():
    """A correct composition over other relations must not be labelled leakage.

    Round 7's finding, and the mirror image of round 6's: with MetaQA's two
    relations hard-coded, a compliance-workload rule that IS the world-correct
    composition falls into `other` and `true_rule_installs` silently reads 0.
    """
    gdpr = "syn:requires_consent<=~processes_data.subject_to_gdpr"
    # with the default (MetaQA) composition it is not recognised
    assert classify([gdpr]) == (False, 0, 1)
    # told which relations compose, it is
    assert classify([gdpr], ("processes_data", "subject_to_gdpr")) == (True, 0, 0)


def test_classify_composition_change_does_not_disturb_self_reference():
    """Self-reference is defined by the head, so it survives a composition swap."""
    assert classify([SELF_REF], ("processes_data", "subject_to_gdpr")) == (False, 1, 0)


# ----------------------------------------------------------------- read_forbid
def _forbid_report(tmp_path, name, dataset, rules):
    """One synthetic forbid-arm report in the on-disk schema (only read fields)."""
    (tmp_path / f"{name}.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "verdict": {"calls_saved_pct": 50.0, "accuracy_cache": 1.0, "accuracy_full": 1.0},
                "arms": [{"arm": "full_distillation", "synthesised_rules": rules}],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_read_forbid_refuses_a_dataset_it_was_not_told_to_expect(tmp_path):
    """Wrong workload stops the run instead of producing a mislabelled artifact."""
    d = _forbid_report(tmp_path, "acme_model-1_hop2_lim300_s0_forbid", "PrivaCI-v2", [])
    with pytest.raises(SystemExit) as e:
        read_forbid(d)
    msg = str(e.value)
    assert "PrivaCI-v2" in msg
    assert DEFAULT_DATASET in msg


def test_read_forbid_accepts_the_expected_dataset(tmp_path):
    """The guard is a check, not a blanket refusal."""
    d = _forbid_report(tmp_path, "acme_model-1_hop2_lim300_s0_forbid", DEFAULT_DATASET, [TRUE_RULE])
    out = read_forbid(d)
    assert list(out.values())[0]["true_rule_installed"] is True


def test_read_forbid_tolerates_reports_without_a_dataset_field(tmp_path):
    """Older reports carry no `dataset`; absence must not be read as a mismatch."""
    (tmp_path / "acme_model-1_hop2_lim300_s0_forbid.json").write_text(
        json.dumps(
            {"verdict": {}, "arms": [{"arm": "full_distillation", "synthesised_rules": []}]}
        ),
        encoding="utf-8",
    )
    assert len(read_forbid(tmp_path)) == 1


# ------------------------------------------------------------------ parse_cell
def test_parse_cell_reads_model_and_seed_from_the_filename():
    """The filename is authoritative; the metadata inside the replay reports is not."""
    assert parse_cell("google_gemini-3.5-flash_hop2_lim300_s1") == ("google/gemini-3.5-flash", 1)
    # a double underscore separates vendor from model, so a vendor whose own name
    # carries an underscore survives instead of being split at the first one
    assert parse_cell("open_router__some_model_hop2_lim300_s3") == ("open_router/some_model", 3)


def test_parse_cell_strips_the_forbid_suffix_so_the_arms_name_one_cell():
    """The treatment arm's files differ only by suffix; both must name the same cell."""
    control = parse_cell("google_gemini-3.5-flash_hop2_lim300_s1")
    forbid = parse_cell("google_gemini-3.5-flash_hop2_lim300_s1_forbid")
    assert control == forbid


def test_parse_cell_short_forms():
    """The first two probe cells were written with short names."""
    assert parse_cell("glm_s2") == ("z-ai/glm-5.2", 2)
    assert parse_cell("sonnet_s0_forbid") == ("anthropic/claude-sonnet-5", 0)


# ------------------------------------------------------------------- match_key
def test_match_key_joins_arms_whose_filenames_differ_only_by_dots():
    """The forbid filenames were written with dots stripped.

    Matching on the raw slug silently paired only the cells whose model names had
    no dots -- 7 of 22 -- so the key drops dots on both sides.
    """
    control = match_key(parse_cell("google_gemini-3.5-flash_hop2_lim300_s1"))
    forbid = match_key(parse_cell("google_gemini-35-flash_hop2_lim300_s1_forbid"))
    assert control == forbid


def test_match_key_does_not_merge_different_seeds_or_models():
    """Dropping dots must not make genuinely different cells collide."""
    a = match_key(parse_cell("google_gemini-3.5-flash_hop2_lim300_s1"))
    b = match_key(parse_cell("google_gemini-3.5-flash_hop2_lim300_s2"))
    c = match_key(parse_cell("google_gemini-3.5-pro_hop2_lim300_s1"))
    assert a != b
    assert a != c


if __name__ == "__main__":
    import unittest

    unittest.main()
