"""The TACET cascade — an auditable neuro-symbolic reasoning engine.

`TACET` is the public entry point. It routes every query through three tiers
of increasing cost — symbolic, KGE, LLM teacher — and feeds Tier-3 answers
back into the distillation loop, so the routing distribution (and hence the
blended cost) drifts downward as a workload is processed.

    from tacet import TACET, WorldGraph, Ontology
    from tacet.llm.teacher import OracleTeacher

    kg   = WorldGraph.from_json("mykg.json")
    onto = Ontology.induce(kg)
    ak   = TACET(kg, onto, teacher=OracleTeacher(my_oracle))
    ak.warmup()
    ans  = ak.ask("Alice", "located_in")     # -> Answer(tier, answers, proof, cost)
    ...
    ak.consolidate()                          # absorb what the teacher taught
    print(ak.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.core.symbolic import Rule, RuleEngine
from tacet.distill.distill import Distiller
from tacet.kge.kge import ComplEx
from tacet.llm.teacher import Narrator, Teacher
from tacet.serve.config import CascadeConfig

Triple = tuple[str, str, str]


@dataclass
class Answer:
    head: str
    relation: str
    tier: int
    answers: list[str]
    text: str
    confidence: float
    cost: float
    latency_ms: float
    proof: list[str] = field(default_factory=list)
    note: str = ""


class TACET:
    """A 3-tier neuro-symbolic cascade with online LLM distillation."""

    def __init__(
        self,
        graph: WorldGraph,
        ontology: Ontology,
        teacher: Teacher,
        rules: list[Rule] | None = None,
        config: CascadeConfig | None = None,
    ) -> None:
        self.graph = graph
        self.ontology = ontology
        self.teacher = teacher
        self.config = config or CascadeConfig()
        self.engine = RuleEngine(ontology, rules)
        self.kge = ComplEx(self.config.kge)
        self.narrator = Narrator()
        self.distiller = Distiller(
            synth_trigger=self.config.synth_trigger,
            min_confidence=self.config.min_confidence,
            min_support=self.config.min_support,
        )
        self.history: list[Answer] = []
        self._candidate_cache: dict[str, list[str]] = {}
        self._kge_ready = False
        self.synthesised_rules: list[str] = []

    # ----------------------------------------------------------- lifecycle
    def warmup(self, calibration: list[tuple[str, str, list[str], str]] | None = None) -> TACET:
        """Materialise the symbolic closure and train the KGE tier."""
        # snapshot the base relations before any teacher write-back, so rule
        # synthesis only composes original relations (not synthesised ones).
        self.distiller.base_relations = set(self.graph.relations())
        self.engine.materialise(self.graph)
        self.kge = ComplEx(self.config.kge).fit(self.graph.triples())
        if calibration:
            self.kge.calibrate(calibration)
        self._kge_ready = True
        return self

    def consolidate(self) -> TACET:
        """Batch job: re-derive the closure and warm-start the KGE on enriched data."""
        self.engine.materialise(self.graph)
        triples = set(self.graph.triples())
        if self.config.kge_augment:
            triples |= set(self.distiller.kge_augmentation())
        self.kge.partial_fit(sorted(triples), epochs=25)
        self._kge_ready = True
        return self

    # ----------------------------------------------------------- candidates
    def _candidates(self, relation: str) -> list[str]:
        if relation in self._candidate_cache:
            return self._candidate_cache[relation]
        rt = self.ontology.relation(relation)
        if rt is None or "*" in rt.range:
            cands = self.graph.entities()
        else:
            cands = [n for typ in rt.range for n in self.graph.nodes_of_type(typ)]
        self._candidate_cache[relation] = cands
        return cands

    # ----------------------------------------------------------- the router
    def ask(self, head: str, relation: str) -> Answer:
        # ---- Tier 1: symbolic --------------------------------------------
        sym = self.engine.query(head, relation)
        if sym.answered:
            return self._record(
                head, relation, 1, sym.answers, 1.0, proof=sym.proof, note="entailed by rules"
            )

        # ---- Tier 2: KGE link prediction ---------------------------------
        if self._kge_ready:
            pred = self.kge.predict_tail(head, relation, self._candidates(relation))
            if (
                pred is not None
                and pred.confidence >= self.config.l2_threshold
                and self.ontology.allows(self.graph, head, relation, pred.tail)
            ):
                return self._record(
                    head,
                    relation,
                    2,
                    [pred.tail],
                    pred.confidence,
                    note=f"KGE prediction (conf={pred.confidence:.2f})",
                )

        # ---- Tier 3: LLM teacher + distillation --------------------------
        resp = self.teacher.answer(self.graph, head, relation)
        if self.config.distillation:
            self._distil(head, relation, resp.answers)
        return self._record(head, relation, 3, resp.answers, 1.0, note="answered by LLM teacher")

    def _distil(self, head: str, relation: str, answers: list[str]) -> None:
        facts = self.distiller.record(head, relation, answers)
        if self.config.write_back:
            for h, r, t in facts:
                self._type_endpoints(r, h, t)
                self.graph.add_edge(h, r, t, provenance="teacher")
                self.engine.add_fact((h, r, t))
        if self.config.rule_synthesis and self.distiller.ready_to_synthesise(relation):
            mined = self.distiller.synthesise(self.graph, relation)
            changed = False
            for m in mined:
                if self.engine.add_rule(m.rule):
                    self.synthesised_rules.append(m.rule.name)
                    changed = True
            if changed:
                self.engine.materialise(self.graph)

    def _type_endpoints(self, relation: str, source: str, target: str) -> None:
        """Type brand-new write-back endpoints from the relation's declared schema.

        Theorem~1's ontology-preservation guarantee holds while the graph stays
        ontology-consistent. A teacher write-back can introduce a fresh endpoint;
        left untyped it defaults to the catch-all ``Entity`` type and a valid
        rule then derives a type-violating fact, breaking that premise during
        normal operation. Assigning the relation's declared *singleton* domain /
        range type to a **new** endpoint keeps the written-back fact well-typed.
        Existing nodes and ambiguous positions (``*`` or a multi-type set) are
        left untouched, so untyped/schema-free relations are unaffected.
        """
        rt = self.ontology.relation(relation)
        if rt is None:
            return
        self._ensure_type(source, rt.domain)
        self._ensure_type(target, rt.range)

    def _ensure_type(self, node_id: str, type_set: frozenset[str]) -> None:
        if self.graph.node(node_id) is not None:
            return
        concrete = [ty for ty in type_set if ty != "*"]
        if len(concrete) == 1:
            self.graph.add_node(node_id, concrete[0])

    def _record(
        self,
        head: str,
        relation: str,
        tier: int,
        answers: list[str],
        confidence: float,
        proof: list[str] | None = None,
        note: str = "",
    ) -> Answer:
        ans = Answer(
            head=head,
            relation=relation,
            tier=tier,
            answers=sorted(answers),
            text=self.narrator.render(head, relation, sorted(answers), tier, proof),
            confidence=confidence,
            cost=self.config.tier_cost[tier],
            latency_ms=self.config.tier_latency_ms[tier],
            proof=proof or [],
            note=note,
        )
        self.history.append(ans)
        return ans

    # ----------------------------------------------------------- reporting
    def report(self) -> dict[str, object]:
        n = len(self.history)
        by = {1: 0, 2: 0, 3: 0}
        cost = latency = 0.0
        for a in self.history:
            by[a.tier] += 1
            cost += a.cost
            latency += a.latency_ms
        return {
            "queries": n,
            "tier1": by[1],
            "tier2": by[2],
            "tier3": by[3],
            "pct_no_llm": (by[1] + by[2]) / n * 100 if n else 0.0,
            "total_cost": cost,
            "avg_cost": cost / n if n else 0.0,
            "avg_latency_ms": latency / n if n else 0.0,
            "synthesised_rules": list(self.synthesised_rules),
        }
