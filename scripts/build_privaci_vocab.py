"""Build the committed normalization vocabulary for PrivaCI-Bench slots.

Two passes with gemini-3.5-flash (cheap, deterministic afterwards):
  pass 1 — propose a closed taxonomy (<= 25 categories) per open slot from a
           sample of distinct raw values;
  pass 2 — map EVERY distinct raw value to exactly one category (batched).

Output: ``src/tacet/data/privaci_vocab.json``
  {slot: {"categories": [...], "aliases": {raw_value: category}}}

The artifact is committed and reviewed by hand; this script is the one-time
bootstrapper, after which normalisation is a deterministic dict lookup (the
"fixed extractor" requirement from the phase-0 analysis).

Run:
    MSYS_NO_PATHCONV=1 skret run -e prod --path=/tacet/prod -- \
        uv run python scripts/build_privaci_vocab.py [--privaci ../PrivaCI-Bench]
"""

import argparse
import json
import os
import re
from pathlib import Path

from tacet.data.privaci import load_privaci
from tacet.llm.metering import MeteredTeacher, PriceTable
from tacet.llm.teachers.llm import GeminiRestTeacher

OPEN_SLOTS = ("information_type", "purpose", "sender_role", "recipient_role", "subject_role")

TAXONOMY_PROMPT = """You are designing a closed taxonomy for normalising free-text
values of the GDPR contextual-integrity slot "{relation}".

Below is a sample of raw values seen in real GDPR enforcement cases. Propose at
most 22 snake_case category names that cover them; include a final catch-all
category named "other". Categories must be mutually exclusive and reusable.

Raw values sample:
{head}

Return ONLY a JSON list of snake_case category name strings, no commentary."""

MAPPING_PROMPT = """You are normalising free-text values of the GDPR
contextual-integrity slot "{relation}" into this closed category list:

{categories}

Map EVERY input value below to exactly one category from the list (use "other"
only when nothing fits). Return ONLY a JSON list of categories, one per input
value, same order and same length as the inputs, no commentary.

Input values (JSON list):
{head}

Answer:"""


def _client():
    return GeminiRestTeacher(
        os.environ["TACET_GEMINI_API_KEY"],
        model="gemini-3.5-flash",
        endpoint="vertex",
        qps=None,
        max_retries=6,
    )


def ask(metered, template, *, head, relation, categories=""):
    t = metered.wrapped
    t._prompt_template = template.replace("{categories}", categories)
    return metered.answer(None, head, relation).answers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--privaci", default="../PrivaCI-Bench")
    ap.add_argument("--batch", type=int, default=80)
    args = ap.parse_args()

    cases = load_privaci(args.privaci, split="GDPR")
    out_path = Path("src/tacet/data/privaci_vocab.json")
    vocab: dict[str, dict] = {}
    metered = MeteredTeacher(_client(), prices=PriceTable.default(), model="gemini-3.5-flash")

    for slot in OPEN_SLOTS:
        distinct = sorted({v for c in cases for v in getattr(c, slot) if v != "none"})
        sample = distinct[:: max(1, len(distinct) // 200)][:200]
        cats = ask(
            metered,
            TAXONOMY_PROMPT,
            head=json.dumps(sample, ensure_ascii=False),
            relation=slot,
        )
        cats = sorted({re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in cats})
        if "other" not in cats:
            cats.append("other")
        print(f"{slot}: {len(distinct)} distinct values -> {len(cats)} categories")

        aliases: dict[str, str] = {}
        for i in range(0, len(distinct), args.batch):
            batch = distinct[i : i + args.batch]
            mapped = ask(
                metered,
                MAPPING_PROMPT,
                head=json.dumps(batch, ensure_ascii=False),
                relation=slot,
                categories=json.dumps(cats),
            )
            if len(mapped) != len(batch):
                print(f"  batch {i}: length mismatch {len(mapped)} != {len(batch)}, retrying once")
                mapped = ask(
                    metered,
                    MAPPING_PROMPT,
                    head=json.dumps(batch, ensure_ascii=False),
                    relation=slot,
                    categories=json.dumps(cats),
                )
            for raw, cat in zip(batch, mapped, strict=False):
                cat = str(cat).strip().lower()
                aliases[raw] = cat if cat in cats else "other"
            missing = [v for v in batch if v not in aliases]
            for v in missing:
                aliases[v] = "other"
        vocab[slot] = {"categories": cats, "aliases": aliases}
        other_n = sum(1 for v in aliases.values() if v == "other")
        print(f"  mapped {len(aliases)} values, other={other_n} ({other_n / len(aliases):.1%})")

    out_path.write_text(
        json.dumps(vocab, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(
        f"wrote {out_path} | total LLM cost ${metered.total_cost_usd:.4f} ({metered.n_calls} calls)"
    )


if __name__ == "__main__":
    main()
