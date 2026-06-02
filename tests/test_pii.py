"""Tests cho PIIRedactor + KGBuilder PII-aware pipeline (G2.4)."""

from __future__ import annotations

import unittest

from tacet.core.ingest import (
    KGBuilder,
    Pattern,
    PIIRedactor,
    RuleBasedExtractor,
)
from tacet.core.ontology import NodeType, Ontology, RelationType


class TestPIIRedactor(unittest.TestCase):
    def setUp(self) -> None:
        self.red = PIIRedactor()

    def test_redacts_email(self) -> None:
        out, rep = self.red.redact("Contact alice@example.com about the report.")
        self.assertNotIn("alice@example.com", out)
        self.assertIn("<EMAIL_1>", out)
        self.assertEqual(rep.counts.get("EMAIL"), 1)

    def test_redacts_us_ssn(self) -> None:
        out, rep = self.red.redact("SSN: 123-45-6789 on the form.")
        self.assertNotIn("123-45-6789", out)
        self.assertIn("<SSN_1>", out)
        self.assertEqual(rep.counts.get("SSN"), 1)

    def test_unique_placeholders_per_occurrence(self) -> None:
        # Two e-mails get distinct placeholders.
        out, rep = self.red.redact("Email a@x.com or b@y.org for details.")
        self.assertIn("<EMAIL_1>", out)
        self.assertIn("<EMAIL_2>", out)
        self.assertEqual(rep.counts.get("EMAIL"), 2)

    def test_preserves_non_pii_text(self) -> None:
        out, rep = self.red.redact("This is a clean sentence.")
        self.assertEqual(out, "This is a clean sentence.")
        self.assertEqual(rep.counts, {})

    def test_redacts_credit_card_shape(self) -> None:
        out, rep = self.red.redact("Card 4111 1111 1111 1111 expires next month.")
        self.assertNotIn("4111 1111 1111 1111", out)
        # CC may be flagged as PHONE too because of the digit-run pattern;
        # both is acceptable — assert at least the CC label fired.
        self.assertIn("CC", rep.counts)


class TestKGBuilderWithRedactor(unittest.TestCase):
    """KGBuilder must pipe text through the redactor before extraction."""

    def _ontology(self) -> Ontology:
        o = Ontology()
        o.add_node_type(NodeType("Person"))
        o.add_node_type(NodeType("Company"))
        o.add_relation_type(RelationType("works_at", frozenset({"Person"}), frozenset({"Company"})))
        return o

    def test_redactor_strips_pii_before_extraction(self) -> None:
        # Pattern that would otherwise capture the e-mail as the "head".
        pat = Pattern(regex=r"(?P<head>\S+) works at (?P<tail>\S+)", relation="works_at")
        ext = RuleBasedExtractor(patterns=[pat])
        ont = self._ontology()
        type_hints = {"works_at": ("Person", "Company")}

        text = "alice@example.com works at acme"
        # Without the redactor, the head is the e-mail.
        plain_builder = KGBuilder(ext, ontology=ont, type_hints=type_hints)
        plain_graph, plain_rep = plain_builder.ingest([text])
        heads = {e.source for e in plain_graph.edges}
        self.assertIn("alice@example.com", heads)
        self.assertEqual(plain_rep.pii_redactions.counts, {})

        # With the redactor wired in, the head is the placeholder and
        # the report records the redaction.
        red_builder = KGBuilder(ext, ontology=ont, type_hints=type_hints, redactor=PIIRedactor())
        red_graph, red_rep = red_builder.ingest([text])
        red_heads = {e.source for e in red_graph.edges}
        self.assertNotIn("alice@example.com", red_heads)
        self.assertIn("<EMAIL_1>", red_heads)
        self.assertEqual(red_rep.pii_redactions.counts.get("EMAIL"), 1)


if __name__ == "__main__":
    unittest.main()
