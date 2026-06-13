"""Conjunctive compliance miner tests on synthetic teacher-labelled cases."""

from tacet.distill.compliance_miner import (
    LabeledCase,
    mine_compliance_rules,
)


def _case(i, atoms, verdict, articles=()):
    return LabeledCase(
        case_id=f"C-{i:03d}",
        atoms=frozenset(atoms),
        verdict=verdict,
        articles=tuple(articles),
    )


CRED = ("information_type", "credentials")
NOCONSENT = ("consent_form", "none")
EXPLICIT = ("consent_form", "explicit")
HEALTH = ("information_type", "health_data")
CARE = ("purpose", "medical_care")
MARKETING = ("purpose", "marketing")


def _dataset():
    cases = []
    # planted rule: credentials AND no consent -> art32 (10 cases, pure)
    for i in range(10):
        cases.append(_case(i, [CRED, NOCONSENT], "prohibit", ["art32"]))
    # health data AND no consent -> art9 (6 cases) — makes NOCONSENT alone impure
    for i in range(10, 16):
        cases.append(_case(i, [HEALTH, NOCONSENT, MARKETING], "prohibit", ["art9"]))
    # credentials WITH consent -> permit (8 cases) — makes CRED alone impure
    for i in range(16, 24):
        cases.append(_case(i, [CRED, EXPLICIT, CARE], "permit"))
    # health data for care with consent -> permit (8 cases) — HEALTH alone impure
    for i in range(24, 32):
        cases.append(_case(i, [HEALTH, EXPLICIT, CARE], "permit"))
    # rare noise pattern below support (2 cases)
    for i in range(32, 34):
        cases.append(_case(i, [("purpose", "weird")], "prohibit", ["art99"]))
    return cases


def test_planted_rule_is_mined_with_full_confidence():
    rules = mine_compliance_rules(_dataset(), min_support=5, min_confidence=0.9)
    art32 = [r for r in rules if r.target == "art32"]
    assert art32, "planted art32 rule not mined"
    best = art32[0]
    assert best.confidence == 1.0
    assert best.support == 10
    assert best.rule.head == ("?c", "violates", "article:art32")
    body_rels = {(b[1], b[2]) for b in best.rule.body}
    assert ("consent_form", "consent:none") in body_rels


def test_single_atom_credentials_not_enough():
    # CRED alone covers 18 cases but only 10 are art32 -> conf 0.56 < 0.9
    rules = mine_compliance_rules(_dataset(), min_support=5, min_confidence=0.9)
    for r in rules:
        if r.target == "art32":
            assert len(r.rule.body) >= 2


def test_noise_below_support_excluded():
    rules = mine_compliance_rules(_dataset(), min_support=5, min_confidence=0.9)
    assert not any(r.target == "art99" for r in rules)


def test_permit_rules_mined():
    rules = mine_compliance_rules(_dataset(), min_support=5, min_confidence=0.9)
    permit = [r for r in rules if r.target == "permit"]
    assert permit
    assert permit[0].rule.head == ("?c", "verdict", "verdict:permit")


def test_most_general_pruning():
    rules = mine_compliance_rules(_dataset(), min_support=5, min_confidence=0.9)
    art32_bodies = [set(r.rule.body) for r in rules if r.target == "art32"]
    # no kept art32 rule body is a strict superset of another kept one
    for a in art32_bodies:
        for b in art32_bodies:
            assert not (a < b)


def test_empty_input():
    assert mine_compliance_rules([]) == []
