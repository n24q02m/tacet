"""TACET — an auditable neuro-symbolic reasoning engine.

With machine-checkable proof trees and online rule distillation.

A neuro-symbolic cascade that answers knowledge-graph questions through tiers
of increasing cost (symbolic rules, KGE link prediction, an LLM teacher)
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

from tacet.cascade.config import CascadeConfig
from tacet.cascade.router import TACET, Answer
from tacet.core.graph import Edge, Node, WorldGraph
from tacet.core.ontology import NodeType, Ontology, RelationType
from tacet.core.symbolic import Rule, RuleEngine
from tacet.distill.distill import Distiller, mine_rules
from tacet.kge.kge import ComplEx, KGEConfig
from tacet.llm.teacher import CallableTeacher, Narrator, OracleTeacher, Teacher

__all__ = [
    "TACET",
    "Answer",
    "CallableTeacher",
    "CascadeConfig",
    "ComplEx",
    "Distiller",
    "Edge",
    "KGEConfig",
    "Narrator",
    "Node",
    "NodeType",
    "Ontology",
    "OracleTeacher",
    "RelationType",
    "Rule",
    "RuleEngine",
    "Teacher",
    "WorldGraph",
    "__version__",
    "mine_rules",
]
