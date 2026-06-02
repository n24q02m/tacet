"""Text-attributed knowledge-graph embeddings.

A multi-modal Tier-2 baseline for TACET: instead of initialising each
entity's embedding with Gaussian noise, we use the entity's textual
description (``Wikidata description``, ``rdfs:comment``, a paragraph
describing a person or place in an internal KG, …) through a language
encoder (``sentence-transformers``, Gemini Embedding 2, Qwen3
Embedding, or any user-supplied ``callable(str) -> np.ndarray``).

The goal is to establish **two claims** in §8.3 of the paper:

1. TACET can exploit available cross-modal signal (language +
   structure) — it is not confined to "structural KGE only".
2. On benchmarks with *cold-start entities* (entities with few edges),
   text-attributed init improves MRR without changing the loss /
   training loop.

Sample usage (sentence-transformers, public)::

    from sentence_transformers import SentenceTransformer
    from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig
    from tacet.kge.kge_textual import seed_kge_from_descriptions

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    desc = {"Q1": "France is a country in Western Europe.",
            "Q2": "Paris is the capital of France.", ...}
    kge = TorchComplEx(TorchKGEConfig(dim=200, score_fn="complex_n3"))
    seed_kge_from_descriptions(kge, desc, encoder=enc.encode)
    kge.fit(triples)

Sample usage (Gemini Embedding 2, ``TACET_GEMINI_API_KEY`` required)::

    from tacet.kge.kge_textual import gemini_encoder, seed_kge_from_descriptions

    enc = gemini_encoder(model="gemini-embedding-001")
    seed_kge_from_descriptions(kge, desc, encoder=enc)

The implementation is intentionally encoder-agnostic: any
``Callable[[list[str]], np.ndarray]`` works; sentence-transformers and
Gemini are convenience helpers shipped with the framework.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

# A single batch of strings ``in`` → ``np.ndarray`` of shape (n, dim).
Encoder = Callable[[list[str]], np.ndarray]


def seed_kge_from_descriptions(
    kge,
    descriptions: dict[str, str],
    encoder: Encoder,
    batch_size: int = 64,
    normalise: bool = True,
) -> dict[str, np.ndarray]:
    """Encode entity descriptions then attach them as text init for ``kge``.

    Returns the dict of (entity_name → np.ndarray) actually computed so
    callers can cache them (encoding is the slow step; if you re-fit
    with different KGE hyperparameters, reuse the cache).
    """
    if not descriptions:
        return {}
    names = list(descriptions.keys())
    texts = [descriptions[n] for n in names]
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = np.asarray(encoder(batch), dtype=np.float32)
        if enc.ndim == 1:
            enc = enc.reshape(1, -1)
        vectors.append(enc)
    arr = np.concatenate(vectors, axis=0)
    if normalise:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr = arr / norms
    out = {n: arr[i] for i, n in enumerate(names)}
    kge.set_text_init(out)
    return out


def gemini_encoder(model: str = "gemini-embedding-001", api_key: str | None = None) -> Encoder:
    """Return a ``list[str] → np.ndarray`` encoder backed by Gemini Embedding 2.

    Lazy-imports ``google-genai``; caller must have installed the
    ``[llm]`` extra (``pip install tacet[llm]``).  Reads the API
    key from ``api_key``, ``TACET_GEMINI_API_KEY`` or
    ``GEMINI_API_KEY``.
    """
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional
        raise ImportError(
            "gemini_encoder requires 'google-genai'.  "
            "Install with `pip install google-genai` "
            "(or `pip install tacet[llm]`)."
        ) from e
    key = api_key or os.environ.get("TACET_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("gemini_encoder: set TACET_GEMINI_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    def _encode(batch: list[str]) -> np.ndarray:
        # google-genai batch embed: one HTTP round trip per call.
        resp = client.models.embed_content(model=model, contents=batch)
        # google-genai returns a list of Embedding objects under .embeddings;
        # we flatten to a (n, dim) matrix.
        vecs = [list(e.values) for e in resp.embeddings]
        return np.asarray(vecs, dtype=np.float32)

    return _encode


def hash_encoder(dim: int = 64, seed: int = 0) -> Encoder:
    """Deterministic hash-based encoder for tests when network / API is unavailable.

    Each string is tokenized simply (lowercase + split on whitespace)
    then accumulated into a vector via hash(token) % dim.  It carries no
    semantics but is deterministic and self-contained enough to verify
    that the pipeline works.
    """

    def _encode(batch: list[str]) -> np.ndarray:
        out = np.zeros((len(batch), dim), dtype=np.float32)
        for i, text in enumerate(batch):
            for tok in text.lower().split() or [text.lower()]:
                # The built-in ``hash`` depends on PYTHONHASHSEED and is
                # not stable across Python processes; we use a simple,
                # stable hash function that does not require hashlib.
                h = (sum(ord(c) for c in tok) * (seed + 1)) % dim
                out[i, h] += 1.0
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return out / norms

    return _encode


__all__ = [
    "Encoder",
    "seed_kge_from_descriptions",
    "gemini_encoder",
    "hash_encoder",
]
