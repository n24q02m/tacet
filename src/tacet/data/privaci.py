"""PrivaCI-Bench loader (compliance cases with contextual-integrity tuples).

Reads the HF arrow files committed in a PrivaCI-Bench checkout
(https://github.com/HKUST-KnowComp/PrivaCI-Bench, MIT) without the heavy
``datasets`` dependency -- ``pyarrow`` reads the IPC stream directly. The
``pyarrow`` import is lazy, mirroring the optional-``httpx`` pattern in
``tacet.llm.teachers.llm``.

Slot values are lowercased/stripped but otherwise raw: the phase-0 analysis
(``experiments/privaci_phase0_analysis.py``) showed they are free text, so
semantic normalisation lives in a separate vocabulary layer, not here.
Article labels are normalised to base numbers (``Article 6(1)`` -> ``art6``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SPLITS = ("GDPR", "AI_ACT", "HIPAA", "ACLU")


def _base_article(label: str) -> str:
    m = re.search(r"article\s*(\d+)", str(label).lower())
    return f"art{m.group(1)}" if m else str(label).strip().lower()[:30]


def _slot(value) -> tuple[str, ...]:  # noqa: ANN001 - arrow rows are untyped
    if value is None:
        return ("none",)
    if isinstance(value, list | tuple):
        vals = sorted({str(v).strip().lower() for v in value if str(v).strip()})
        return tuple(vals) or ("none",)
    return (str(value).strip().lower() or "none",)


@dataclass(frozen=True)
class PrivaCICase:
    case_id: str
    norm_type: str  # "permit" | "prohibit" | "not_applicable"
    sender: tuple[str, ...]
    sender_role: tuple[str, ...]
    recipient: tuple[str, ...]
    recipient_role: tuple[str, ...]
    subject: tuple[str, ...]
    subject_role: tuple[str, ...]
    information_type: tuple[str, ...]
    consent_form: str
    purpose: tuple[str, ...]
    followed_articles: tuple[str, ...]
    violated_articles: tuple[str, ...]
    case_content: str


def load_privaci(root: str | Path, *, split: str = "GDPR") -> list[PrivaCICase]:
    """Load one split from a PrivaCI-Bench checkout (or any directory whose
    ``cases/<split>/data-*.arrow`` follows the same schema)."""
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {_SPLITS}")
    try:
        import pyarrow as pa
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError(
            "load_privaci requires 'pyarrow'. Install with `uv sync --extra experiments`."
        ) from e

    root = Path(root)
    split_dir = root / "HF_cache" / "cases" / split
    if not split_dir.is_dir():
        split_dir = root / "cases" / split  # test fixtures / re-rooted copies
    files = sorted(split_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"no arrow files under {split_dir}")

    tables = []
    for f in files:
        try:
            tables.append(pa.ipc.open_stream(str(f)).read_all())
        except pa.ArrowInvalid:
            tables.append(pa.ipc.open_file(str(f)).read_all())
    table = pa.concat_tables(tables)

    cases: list[PrivaCICase] = []
    for i, row in enumerate(table.to_pylist()):
        cases.append(
            PrivaCICase(
                case_id=f"{split}-{i:04d}",
                norm_type=str(row.get("norm_type") or "not_applicable").strip().lower(),
                sender=_slot(row.get("sender")),
                sender_role=_slot(row.get("sender_role")),
                recipient=_slot(row.get("recipient")),
                recipient_role=_slot(row.get("recipient_role")),
                subject=_slot(row.get("subject")),
                subject_role=_slot(row.get("subject_role")),
                information_type=_slot(row.get("information_type")),
                consent_form=_slot(row.get("consent_form"))[0],
                purpose=_slot(row.get("purpose")),
                followed_articles=tuple(
                    sorted({_base_article(a) for a in (row.get("followed_articles") or [])})
                ),
                violated_articles=tuple(
                    sorted({_base_article(a) for a in (row.get("violated_articles") or [])})
                ),
                case_content=str(row.get("case_content") or ""),
            )
        )
    return cases


__all__ = ["PrivaCICase", "load_privaci"]
