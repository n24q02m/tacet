"""Dataset loaders — standard KG benchmarks and the curated demo graph.

Three things are provided:

* ``load_triples_tsv(path)`` — read a single tab-separated `head\\trelation\\ttail`
  file (the standard FB15k / WN18 / NELL layout) into a list of triples.
* ``load_kg_dataset(root)`` — read a directory laid out as ``train.txt``,
  ``valid.txt``, ``test.txt`` (the layout shipped by FB15k-237, WN18RR,
  YAGO3-10, NELL-995…) into a :class:`KGDataset` with the standard splits.
* ``load_worldgeo()`` — load the curated 15-country world-geography graph
  shipped in the package as ``worldgeo.json``.

Download links for the standard benchmarks (kept out of the repository to
respect their licences and size):

* FB15k-237: https://www.microsoft.com/en-us/download/details.aspx?id=52312
* WN18RR:    https://github.com/TimDettmers/ConvE
* YAGO3-10:  https://github.com/TimDettmers/ConvE
* NELL-995:  https://github.com/wenhuchen/KB-Reasoning-Data

Once unzipped to a directory ``<root>`` containing ``train.txt`` /
``valid.txt`` / ``test.txt``::

    from tacet.data import load_kg_dataset
    ds = load_kg_dataset("data/FB15k-237")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from tacet.core.graph import WorldGraph

Triple = tuple[str, str, str]

_HERE = Path(__file__).resolve().parent


def load_triples_tsv(path: str | Path, sep: str = "\t", limit: int | None = None) -> list[Triple]:
    """Read triples from a single TSV/CSV file (``head<sep>relation<sep>tail``)."""
    out: list[Triple] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(sep)
            if len(parts) < 3:
                continue
            out.append((parts[0], parts[1], parts[2]))
            if limit is not None and len(out) >= limit:
                break
    return out


@dataclass
class KGDataset:
    """A standard KG-completion benchmark split into train/valid/test triples."""

    name: str
    train: list[Triple] = field(default_factory=list)
    valid: list[Triple] = field(default_factory=list)
    test: list[Triple] = field(default_factory=list)

    def all_triples(self) -> list[Triple]:
        return [*self.train, *self.valid, *self.test]

    def stats(self) -> dict[str, int]:
        ents = {x for h, _, t in self.all_triples() for x in (h, t)}
        rels = {r for _, r, _ in self.all_triples()}
        return {
            "entities": len(ents),
            "relations": len(rels),
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
        }

    def to_graph(self) -> WorldGraph:
        """Build a `WorldGraph` containing only the train triples (for KGE warm-up)."""
        return WorldGraph.from_triples(self.train, name=self.name)


def load_kg_dataset(root: str | Path, name: str | None = None) -> KGDataset:
    """Load a KGC benchmark from a directory with ``train/valid/test.txt`` files."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found; see tacet/datasets/__init__.py docstring")
    files = {part: root / f"{part}.txt" for part in ("train", "valid", "test")}
    missing = [p for p, fp in files.items() if not fp.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} split file(s) in {root}; expected train.txt/valid.txt/test.txt"
        )
    return KGDataset(
        name=name or root.name,
        train=load_triples_tsv(files["train"]),
        valid=load_triples_tsv(files["valid"]),
        test=load_triples_tsv(files["test"]),
    )


def split_triples(
    triples: list[Triple], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> KGDataset:
    """Split a triple list into train/valid/test with the given ratios (deterministic)."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    shuffled = list(triples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    return KGDataset(
        name="split",
        train=shuffled[:n_train],
        valid=shuffled[n_train : n_train + n_valid],
        test=shuffled[n_train + n_valid :],
    )


def load_worldgeo() -> WorldGraph:
    """Load the curated 15-country world-geography graph shipped with the package."""
    return WorldGraph.from_json(_HERE / "worldgeo.json")


def worldgeo_dataset(
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> KGDataset:
    """Train/valid/test split of the world-geography graph — handy for KGE smoke tests.

    Note: only 100ish edges total — useful as a sanity-check, not a benchmark.
    Use ``synthetic_kg_dataset`` or download FB15k-237 / WN18RR for real eval.
    """
    triples = [(e.source, e.relation, e.target) for e in load_worldgeo().edges]
    ds = split_triples(triples, ratios=ratios, seed=seed)
    ds.name = "worldgeo"
    return ds


def synthetic_kg_dataset(
    seed: int = 0, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> KGDataset:
    """The synthetic-benchmark graph split into KGC train/valid/test.

    Larger (≈1400 triples, ≈176 entities, 10 relations) and structurally rich
    enough that ComplEx attains a meaningful MRR. Stand-in until the user
    downloads a public benchmark.
    """
    from tacet.eval.benchmark import BenchmarkConfig, generate

    bench = generate(BenchmarkConfig(seed=seed))
    triples = [(e.source, e.relation, e.target) for e in bench.graph.edges]
    ds = split_triples(triples, ratios=ratios, seed=seed)
    ds.name = "synthetic-org"
    return ds


__all__ = [
    "KGDataset",
    "Triple",
    "load_kg_dataset",
    "load_triples_tsv",
    "load_worldgeo",
    "split_triples",
    "synthetic_kg_dataset",
    "worldgeo_dataset",
]
