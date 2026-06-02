"""TACET — an auditable causal-temporal neuro-symbolic reasoning engine.

With machine-checkable proof trees and online rule distillation.

A neuro-symbolic cascade that answers knowledge-graph questions through three
tiers of increasing cost (symbolic rules, KGE link prediction, an LLM teacher)
and distils the teacher's reasoning down into the cheap tiers online, so the
blended cost per query falls toward the symbolic floor as a workload streams.

Public API:

    from tacet import TACET, WorldGraph, Ontology
    from tacet.llm.teacher import OracleTeacher, CallableTeacher
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("tacet")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

from tacet.cascade.router import TACET, Answer
from tacet.core.causal import CausalModel, Variable, backdoor_set, counterfactual
from tacet.core.graph import Edge, Node, WorldGraph
from tacet.core.ingest import (
    CallableExtractor,
    Extractor,
    IngestionReport,
    KGBuilder,
    Pattern,
    RuleBasedExtractor,
)
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule, RuleEngine
from tacet.core.temporal import TemporalEngine, TemporalQuery, allen_relation, temporal_edge
from tacet.distill.concepts import (
    InducedRelation,
    InducedType,
    induce_node_types,
    induce_relations,
    revise_ontology,
)
from tacet.distill.distill import Distiller, mine_rules
from tacet.experimental.agent import Action, Goal, GroundedAction, Plan, Planner
from tacet.experimental.episodic import Episode, EpisodicStore, FeedbackCurator, RuleScore
from tacet.experimental.federation import FederatedGraph, Provenance, merge, trust_weighted
from tacet.experimental.worldmodel import IdentityWorldModel, Trajectory, WorldModel
from tacet.kge.kge import ComplEx, KGEConfig
from tacet.llm.teacher import CallableTeacher, Narrator, OracleTeacher, Teacher
from tacet.serve.config import CascadeConfig
from tacet.serve.settings import Settings, load_settings

__all__ = [
    "TACET",
    "Action",
    "Answer",
    "CallableExtractor",
    "CallableTeacher",
    "CascadeConfig",
    "CausalModel",
    "ComplEx",
    "Distiller",
    "Edge",
    "Episode",
    "EpisodicStore",
    "Extractor",
    "FederatedGraph",
    "FeedbackCurator",
    "Goal",
    "GroundedAction",
    "IdentityWorldModel",
    "InducedRelation",
    "InducedType",
    "IngestionReport",
    "KGBuilder",
    "KGEConfig",
    "Narrator",
    "Node",
    "NodeType",
    "Ontology",
    "OracleTeacher",
    "Pattern",
    "Plan",
    "Planner",
    "Provenance",
    "RelationType",
    "Settings",
    "Rule",
    "RuleBasedExtractor",
    "RuleEngine",
    "RuleScore",
    "Teacher",
    "TemporalEngine",
    "TemporalQuery",
    "Trajectory",
    "Variable",
    "WorldGraph",
    "WorldModel",
    "__version__",
    "allen_relation",
    "backdoor_set",
    "counterfactual",
    "induce_node_types",
    "induce_relations",
    "load_settings",
    "merge",
    "mine_rules",
    "revise_ontology",
    "temporal_edge",
    "trust_weighted",
]
