"""Tier 2 implementation wrapping a KGE predictor and ontology gate."""

from __future__ import annotations

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.kge.kge import ComplEx
from tacet.tiers.base import TierResult


class KGETier:
    """Tier 2: calibrated KGE link prediction gated by threshold and ontology."""

    tier_id: int = 2
    name: str = "kge"

    def __init__(
        self,
        kge: ComplEx,
        graph: WorldGraph,
        ontology: Ontology,
        threshold: float = 0.60,
    ) -> None:
        self.kge = kge
        self.graph = graph
        self.ontology = ontology
        self.threshold = threshold

    def can_handle(self, head: str, relation: str) -> bool:
        ent = getattr(self.kge, "ent", None)
        rel = getattr(self.kge, "rel", None)
        if ent is not None and head not in ent:
            return False
        return not (rel is not None and relation not in rel)

    def _candidates(self, relation: str) -> list[str]:
        rt = self.ontology.relation(relation)
        if rt is None or "*" in rt.range:
            return self.graph.entities()
        return [n for typ in rt.range for n in self.graph.nodes_of_type(typ)]

    def answer(self, head: str, relation: str) -> TierResult | None:
        if not self.can_handle(head, relation):
            return None
        pred = self.kge.predict_tail(head, relation, self._candidates(relation))
        if (
            pred is not None
            and pred.confidence >= self.threshold
            and self.ontology.allows(self.graph, head, relation, pred.tail)
        ):
            return TierResult(
                tier=self.tier_id,
                answers=[pred.tail],
                confidence=pred.confidence,
                proof=[],
                grounded=False,
                note=f"KGE prediction (conf={pred.confidence:.2f})",
            )
        return None
