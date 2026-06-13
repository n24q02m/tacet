"""Deterministic slot normaliser backed by the committed PrivaCI vocabulary.

The vocabulary (``privaci_vocab.json``) is bootstrapped once by
``scripts/build_privaci_vocab.py`` and committed; at run time normalisation is
a dict lookup, so the extractor is fixed across experimental arms (the
phase-0 requirement). Unknown values fall back to ``other`` so the normaliser
is total.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from tacet.data.privaci import PrivaCICase

_VOCAB_PATH = Path(__file__).with_name("privaci_vocab.json")

#: Slots normalised through the vocabulary; ``consent_form`` is already a
#: closed three-value set in the raw data and passes through unchanged.
OPEN_SLOTS = ("information_type", "purpose", "sender_role", "recipient_role", "subject_role")


@lru_cache(maxsize=1)
def load_vocab(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else _VOCAB_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_value(slot: str, raw: str, vocab: dict | None = None) -> str:
    vocab = vocab if vocab is not None else load_vocab()
    entry = vocab.get(slot)
    if entry is None:  # closed slot (e.g. consent_form): pass through
        return raw
    return entry["aliases"].get(raw, "other")


def normalize_case(case: PrivaCICase, vocab: dict | None = None) -> dict[str, tuple[str, ...]]:
    """Normalised slot values for one case (deduplicated, sorted)."""
    vocab = vocab if vocab is not None else load_vocab()
    out: dict[str, tuple[str, ...]] = {}
    for slot in OPEN_SLOTS:
        raws = getattr(case, slot)
        out[slot] = tuple(sorted({normalize_value(slot, r, vocab) for r in raws if r != "none"}))
    out["consent_form"] = (case.consent_form,)
    return out


__all__ = ["OPEN_SLOTS", "load_vocab", "normalize_case", "normalize_value"]
