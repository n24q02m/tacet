#!/usr/bin/env python3
"""Quickstart — the TACET public API in ~20 lines.

python examples/quickstart.py
"""

from __future__ import annotations

from tacet import TACET, CascadeConfig
from tacet.eval.benchmark import BenchmarkConfig, generate
from tacet.kge.kge import KGEConfig
from tacet.llm.teacher import OracleTeacher

# 1. a knowledge graph + ontology + an LLM teacher (here: an offline oracle).
bench = generate(BenchmarkConfig(seed=0))
teacher = OracleTeacher(bench.oracle, entity_pool=bench.entity_pool)

# 2. build the cascade and warm it up (compile rules, train the KGE tier).
tacet = TACET(
    bench.graph,
    bench.ontology,
    teacher,
    rules=bench.given_rules,
    config=CascadeConfig(kge=KGEConfig(epochs=80)),
)
tacet.warmup(calibration=bench.calibration)

# 3. ask questions — each Answer carries its tier, cost and (for Tier 1) a proof.
for head, relation in bench.workload[:6]:
    ans = tacet.ask(head, relation)
    print(f"T{ans.tier}  ${ans.cost:.4f}  {ans.text.splitlines()[0]}")

# 4. process the rest of the workload, consolidating periodically.
for i, (head, relation) in enumerate(bench.workload[6:], start=6):
    tacet.ask(head, relation)
    if (i + 1) % 100 == 0:
        tacet.consolidate()

print("\nreport:", tacet.report())
