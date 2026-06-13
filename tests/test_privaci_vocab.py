"""Normaliser tests against a tiny in-memory vocabulary."""

from tacet.data.privaci import PrivaCICase
from tacet.data.privaci_vocab import normalize_case, normalize_value

VOCAB = {
    "information_type": {
        "categories": ["credentials", "health_data", "other"],
        "aliases": {"passwords": "credentials", "health data": "health_data"},
    },
    "purpose": {
        "categories": ["storage", "marketing", "other"],
        "aliases": {"data storage": "storage"},
    },
    "sender_role": {
        "categories": ["controller", "other"],
        "aliases": {"data controller": "controller"},
    },
    "recipient_role": {"categories": ["processor", "other"], "aliases": {}},
    "subject_role": {
        "categories": ["data_subject", "other"],
        "aliases": {"data subjects": "data_subject"},
    },
}


def _case(**kw):
    base = dict(
        case_id="GDPR-0000",
        norm_type="prohibit",
        sender=("acme",),
        sender_role=("data controller",),
        recipient=("sys",),
        recipient_role=("storage backend",),
        subject=("users",),
        subject_role=("data subjects",),
        information_type=("passwords",),
        consent_form="none",
        purpose=("data storage",),
        followed_articles=(),
        violated_articles=("art32",),
        case_content="Acme stored passwords unencrypted.",
    )
    base.update(kw)
    return PrivaCICase(**base)


def test_normalize_value_hit_and_miss():
    assert normalize_value("information_type", "passwords", VOCAB) == "credentials"
    assert normalize_value("information_type", "unseen thing", VOCAB) == "other"
    # closed slot passes through
    assert normalize_value("consent_form", "explicit", VOCAB) == "explicit"


def test_normalize_case_shapes():
    n = normalize_case(_case(), VOCAB)
    assert n["information_type"] == ("credentials",)
    assert n["purpose"] == ("storage",)
    assert n["sender_role"] == ("controller",)
    assert n["recipient_role"] == ("other",)
    assert n["consent_form"] == ("none",)
