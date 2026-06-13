"""Compliance graph builder tests (synthetic cases + tiny vocab)."""

from tacet.data.privaci import PrivaCICase
from tacet.data.privaci_graph import build_compliance_benchmark

VOCAB = {
    "information_type": {
        "categories": ["credentials", "health_data", "other"],
        "aliases": {"passwords": "credentials", "health data": "health_data"},
    },
    "purpose": {
        "categories": ["storage", "care", "other"],
        "aliases": {"data storage": "storage", "care": "care"},
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


def _case(case_id, **kw):
    base = dict(
        case_id=case_id,
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


def _benchmark():
    cases = [
        _case("GDPR-0000"),
        _case(
            "GDPR-0001",
            norm_type="permit",
            information_type=("health data",),
            purpose=("care",),
            consent_form="explicit",
            violated_articles=(),
        ),
    ]
    return build_compliance_benchmark(cases, VOCAB)


def test_graph_structure_and_sharing():
    bench = _benchmark()
    g = bench.graph
    # both cases share the controller role node -> structural overlap exists
    assert g.out("GDPR-0000", "sender_role") == {"role:controller"}
    assert g.out("GDPR-0001", "sender_role") == {"role:controller"}
    assert g.out("GDPR-0000", "information_type") == {"info_type:credentials"}
    assert g.out("GDPR-0000", "consent_form") == {"consent:none"}
    assert g.out("GDPR-0001", "consent_form") == {"consent:explicit"}


def test_graph_is_well_typed():
    bench = _benchmark()
    assert bench.ontology.validate(bench.graph) == []


def test_oracle_and_workload_no_leak():
    bench = _benchmark()
    assert bench.workload == ("GDPR-0000", "GDPR-0001")
    assert bench.oracle["GDPR-0000"] == ("prohibit", ("art32",))
    assert bench.oracle["GDPR-0001"] == ("permit", ())
    # gold labels never enter the graph
    assert bench.graph.out("GDPR-0000", "violates") == set()
    assert "passwords" in bench.case_content["GDPR-0000"].lower()


def test_violates_relation_gated_by_ontology():
    bench = _benchmark()
    g, onto = bench.graph, bench.ontology
    g.add_node("article:art32", type="article")
    assert onto.allows(g, "GDPR-0000", "violates", "article:art32") is True
    # type gate rejects a violates edge into a non-article node
    assert onto.allows(g, "GDPR-0000", "violates", "role:controller") is False


def test_verdict_relation_gated_by_ontology():
    bench = _benchmark()
    g, onto = bench.graph, bench.ontology
    g.add_node("verdict:permit", type="verdict")
    assert onto.allows(g, "GDPR-0000", "verdict", "verdict:permit") is True
    assert onto.allows(g, "GDPR-0000", "violates", "verdict:permit") is False
