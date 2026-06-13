# Changelog

## 0.1.0 — first public release
Initial public release of TACET, the reference implementation for the paper
"Cost-Amortised Reasoning via Self-Distilling Neuro-Symbolic Cascades: From Knowledge-Graph QA to Regulatory-Compliance Checking."

- Sound Datalog reasoning with machine-checkable proof trees (`tacet.core.symbolic`)
- Causal utility functions — Pearl-framework back-door / front-door / IV detection and counterfactuals over a discrete SCM; library helpers, not integrated into the cascade (`tacet.core.causal`)
- Bi-temporal reasoning with Allen interval relations (`tacet.core.temporal`)
- Online distillation of teacher knowledge into Datalog-checkable Horn rules (`tacet.distill`)
- Three-tier cascade with KGE link prediction and LLM teachers (`tacet.cascade`, `tacet.kge`, `tacet.llm`)
- Auditability evaluation and ProofWriter / synthetic-KGQA benchmarks (`tacet.eval`, `tacet.data`)
- `tacet.experimental`: a graph forward-dynamics forecaster retained as a documented negative result (not a core contribution)
