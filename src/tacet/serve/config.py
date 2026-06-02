"""Configuration — cost model and cascade hyper-parameters.

The per-tier cost / latency figures are the *model constants* used throughout
the paper's experiments. They reflect realistic 2026 ratios — a frontier LLM
reasoning call is ~3 orders of magnitude more expensive than a graph query —
but the qualitative results depend only on `c3 >> c2 > c1`, not the exact
values. Override `CascadeConfig` to plug in measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tacet.kge.kge import KGEConfig

# USD per query, by tier.
TIER_COST = {1: 0.0001, 2: 0.0010, 3: 0.0500}
# Milliseconds per query, by tier.
TIER_LATENCY_MS = {1: 3.0, 2: 15.0, 3: 900.0}


@dataclass
class CascadeConfig:
    """Hyper-parameters of the 3-tier cascade."""

    l2_threshold: float = 0.60  # min Tier-2 confidence to accept a prediction
    distillation: bool = True  # master switch (off => static cascade baseline)
    write_back: bool = True  # mechanism 1: fact write-back
    kge_augment: bool = True  # mechanism 2: KGE training augmentation
    rule_synthesis: bool = True  # mechanism 3: rule mining
    synth_trigger: int = 10  # teacher facts on a relation before mining
    min_confidence: float = 0.95  # rule-mining confidence threshold
    min_support: int = 3  # rule-mining support threshold
    kge: KGEConfig = field(default_factory=KGEConfig)
    tier_cost: dict[int, float] = field(default_factory=lambda: dict(TIER_COST))
    tier_latency_ms: dict[int, float] = field(default_factory=lambda: dict(TIER_LATENCY_MS))
