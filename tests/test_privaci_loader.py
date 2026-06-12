"""Loader tests against a tiny synthetic arrow file with the PrivaCI schema."""

import pyarrow as pa
import pytest

from tacet.data.privaci import PrivaCICase, load_privaci

SCHEMA_COLS = {
    "norm_type": ["prohibit", "permit"],
    "sender": [["Acme"], ["Beta"]],
    "sender_role": [["Data Controller"], ["Processor"]],
    "recipient": [["Sys"], ["Cloud"]],
    "recipient_role": [["Storage"], ["Processor"]],
    "subject": [["Users"], ["Patients"]],
    "subject_role": [["Data Subjects"], ["Data Subjects"]],
    "information_type": [["Passwords"], ["Health Data"]],
    "consent_form": [None, "explicit"],
    "purpose": ["Data Storage", "Care"],
    "followed_articles": [[], ["Article 6"]],
    "violated_articles": [["Article 32", "Article 6(1)"], []],
    "case_content": ["Acme stored passwords unencrypted.", "Beta processed with consent."],
}


@pytest.fixture()
def arrow_dir(tmp_path):
    split = tmp_path / "cases" / "GDPR"
    split.mkdir(parents=True)
    table = pa.table(SCHEMA_COLS)
    with pa.ipc.new_stream(split / "data-00000-of-00001.arrow", table.schema) as w:
        w.write_table(table)
    return tmp_path


def test_load_privaci_basic(arrow_dir):
    cases = load_privaci(arrow_dir, split="GDPR")
    assert len(cases) == 2
    c = cases[0]
    assert isinstance(c, PrivaCICase)
    assert c.case_id == "GDPR-0000"
    assert c.norm_type == "prohibit"
    assert c.information_type == ("passwords",)
    assert c.consent_form == "none"
    assert c.violated_articles == ("art32", "art6")
    assert "passwords" in c.case_content.lower()


def test_load_privaci_normalises_articles(arrow_dir):
    cases = load_privaci(arrow_dir, split="GDPR")
    assert cases[1].followed_articles == ("art6",)
    assert cases[1].violated_articles == ()
    assert cases[1].consent_form == "explicit"


def test_load_privaci_rejects_unknown_split(arrow_dir):
    with pytest.raises(ValueError):
        load_privaci(arrow_dir, split="NOPE")
