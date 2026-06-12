"""Phase-0 harness-readiness analysis for the PrivaCI-Bench compliance port.

Measures whether the GDPR case stream has enough structural repetition for
TACET's cost-amortisation mechanism to be evaluable on it. Run against a
checkout of https://github.com/HKUST-KnowComp/PrivaCI-Bench (MIT):

    uv run --with datasets python experiments/privaci_phase0_analysis.py \
        --privaci-dir ../PrivaCI-Bench

Findings on the 2025-04 checkout (3,137 GDPR cases) that gate Phase 1:
- Raw CI-tuple slots are free text (1,300 distinct information_type values),
  so exact-match repetition is low (10.5% in-order stream hit rate); a
  normalisation ontology per slot is mandatory Phase-1 infrastructure.
- The rule-head space is small after normalising article labels: 151 base
  articles, top-10 fully cover 73.7% of violating cases, and 95.8% of cases
  in stream order touch only previously-seen articles.
- Verdict labels are consistent: one CI pattern maps to one norm_type for
  all but a single pattern (majority-vote ceiling 100.0%).
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

RULE_SLOTS = [
    "sender_role",
    "recipient_role",
    "subject_role",
    "information_type",
    "consent_form",
    "purpose",
]


def norm(value):
    if value is None:
        return "none"
    if isinstance(value, list):
        return tuple(sorted(str(x).strip().lower() for x in value)) or ("none",)
    return str(value).strip().lower()


def base_article(label):
    m = re.search(r"article\s*(\d+)", str(label).lower())
    return f"art{m.group(1)}" if m else str(label).lower()[:30]


def pattern(example, fields):
    return tuple(norm(example[f]) for f in fields)


def stream_repeat_rate(ds, fields):
    seen, hits = set(), 0
    for i in range(len(ds)):
        p = pattern(ds[i], fields)
        if p in seen:
            hits += 1
        else:
            seen.add(p)
    return hits / len(ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--privaci-dir", type=Path, required=True, help="Path to a PrivaCI-Bench checkout"
    )
    parser.add_argument("--split", default="GDPR", choices=["GDPR", "AI_ACT", "HIPAA", "ACLU"])
    args = parser.parse_args()

    ds = load_from_disk(str(args.privaci_dir / "HF_cache" / "cases"))[args.split]
    n = len(ds)
    print(f"split={args.split} n={n}")

    print("\n== norm_type distribution ==")
    print(Counter(norm(x) for x in ds["norm_type"]))

    print("\n== field cardinality ==")
    for col in RULE_SLOTS + ["sender", "recipient", "subject"]:
        if col in ds.column_names:
            print(f"  {col}: {len({norm(x) for x in ds[col]})} distinct")

    print("\n== structural repetition (raw values) ==")
    for name, fields in [
        ("full CI pattern", RULE_SLOTS),
        ("info+consent+purpose", ["information_type", "consent_form", "purpose"]),
        ("roles only", ["sender_role", "recipient_role", "subject_role"]),
    ]:
        pats = Counter(pattern(ds[i], fields) for i in range(n))
        singletons = sum(1 for c in pats.values() if c == 1)
        print(
            f"  {name}: distinct={len(pats)}/{n} singletons={singletons} "
            f"stream-hit={stream_repeat_rate(ds, fields):.1%}"
        )

    print("\n== verdict consistency (full CI pattern -> norm_type) ==")
    by_pat = {}
    for i in range(n):
        by_pat.setdefault(pattern(ds[i], RULE_SLOTS), Counter())[norm(ds[i]["norm_type"])] += 1
    ceiling = sum(c.most_common(1)[0][1] for c in by_pat.values()) / n
    conflicts = sum(1 for c in by_pat.values() if len(c) > 1)
    print(f"  majority-vote ceiling={ceiling:.1%} conflicting-patterns={conflicts}")

    print("\n== rule-head space (normalised article labels) ==")
    arts = Counter()
    case_sets = []
    for va in ds["violated_articles"]:
        s = {base_article(a) for a in (va or [])}
        case_sets.append(s)
        arts.update(s)
    violating = [s for s in case_sets if s]
    print(f"  {len(arts)} distinct base articles; top-10: {arts.most_common(10)}")
    for k in (5, 10, 15):
        topk = {a for a, _ in arts.most_common(k)}
        full = sum(1 for s in violating if s <= topk) / len(violating)
        print(f"  top-{k} full-coverage of violating cases: {full:.1%}")
    seen, hits = set(), 0
    for s in case_sets:
        if s and s <= seen:
            hits += 1
        seen |= s
    print(f"  stream: cases fully within previously-seen articles: {hits / len(violating):.1%}")


if __name__ == "__main__":
    main()
