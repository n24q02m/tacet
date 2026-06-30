"""Tests for compliance-domain prompt + answer parsing."""

from tacet.llm.teachers.compliance import (
    COMPLIANCE_PROMPT_TEMPLATE,
    NL_STRATEGY_PROMPT_TEMPLATE,
    parse_compliance_answer,
)


def test_parse_compliance_answer_basics():
    # Standard successful parse
    assert parse_compliance_answer(["permit"]) == ("permit", ())
    # sorted: "art32" comes before "art6"
    assert parse_compliance_answer(["prohibit", "art6", "art32"]) == ("prohibit", ("art32", "art6"))


def test_parse_compliance_answer_edge_cases():
    # Empty or unparseable
    assert parse_compliance_answer([]) == ("abstain", ())
    assert parse_compliance_answer(["weird"]) == ("abstain", ())
    assert parse_compliance_answer(["maybe"]) == ("abstain", ())


def test_parse_compliance_answer_normalization():
    # Case insensitivity for verdict
    assert parse_compliance_answer(["Permit"])[0] == "permit"
    assert parse_compliance_answer(["PROHIBIT"])[0] == "prohibit"

    # Article normalization and deduplication
    answers = ["prohibit", "Article 32", "art6(1)", "ART 32", "not-an-article"]
    verdict, articles = parse_compliance_answer(answers)
    assert verdict == "prohibit"
    # sorted: "art32" comes before "art6"
    assert articles == ("art32", "art6")


def test_parse_compliance_answer_sorting():
    # String sort: "art12" comes before "art5"
    _, articles = parse_compliance_answer(["prohibit", "art6", "art12", "art5"])
    assert articles == ("art12", "art5", "art6")


def test_templates_have_placeholders():
    assert "{head}" in COMPLIANCE_PROMPT_TEMPLATE
    assert "{head}" in NL_STRATEGY_PROMPT_TEMPLATE
    # COMPLIANCE_PROMPT_TEMPLATE specifically says it uses head
    assert "Scenario:\n{head}" in COMPLIANCE_PROMPT_TEMPLATE
