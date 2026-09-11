# Appendix note — DRAFT for v9: Recursive targets and the pure-self-loop guard

> **Status:** `DRAFT — NOT PUBLISHED`. Cutting a Zenodo v9 with this note is a separate
> publication decision. Nothing in this file is a claim until that gate opens.
> Prepared from committed, replayable `$0` artifacts only (E18 synthetic grid + E18-W2
> classification replay). No provider was contacted at any point.

## 1. What the published record says today

TACET v8 (current) documents the structural junk rejection (E16) with its registered
scope limitation, stated before the result: forbidding the mining target from length-2
bodies removes the self-referential junk rule without gold, but "forecloses genuinely
recursive targets (`ancestor <= ancestor.parent`)", so the flag must stay opt-in and the
POSITIVE result licenses "safe on non-recursive targets" only.

## 2. New result 1 — the blunt ban measurably forecloses recursion (E18, synthetic)

On a seeded random DAG (n=150, edge_p=0.03) whose target is the genuine transitive
closure `anc` over base `parent`, with oracle teachers (complete answer sets, 30 sampled
heads per seed), grid 3 seeds x 3 gammas (0.50/0.70/0.90) x 3 arms
(published / `e16_ban` / refined pure-self-loop guard), locked decision rule applied:

- the `e16_ban` arm loses forward-chaining derivation coverage over the full closure on
  **9/9 recursive cells** (e.g. 0.7065 -> 0.5038, 0.6044 -> 0.4650, 0.7907 -> 0.5014);
- the refined arm (ban **only** the pure all-target body `target x target`, both
  inversions; keep mixed `target x base` legs) preserves published-arm coverage on
  **9/9** cells and drops exactly the pure self-loop installs, nothing else;
- `unsound_derivations = 0` in every cell: on a genuinely transitive target even the
  pure self-loop is sound (a subset of the closure) — it is redundant, not harmful,
  which is why coverage-preservation and junk-removal can coexist here.

Artifact: `experiments/results/e18_recursive_target_synthetic.json`
(commit `333b9670`, branch `codex/e18-recursive-boundary-20260911`).

## 3. New result 2 — the refined guard subsumes E16 on the real grid (E18-W2, half i)

Re-classifying every `synthesised_rules` entry of the committed E16 MetaQA 2-hop
artifact (22 cells, both arms; recording SHA-256 `4b52d91d...`):

- control arm: **14/14** target-containing rules classify `pure_self_loop`
  (`mixed_recursive` = 0), spread over exactly the 7 published junk cells;
- true compositions: **10/10** cells kept, every composition rule `base_only`;
- forbid arm: 0 target-containing rules, 10/10 true installs;
- totals reconcile with the published aggregates exactly, per cell as well as in
  aggregate (per-cell reconciliation empty).

Decision (locked before the run): **REFINED-GUARD-SUBSUMES**. Combined with section 2:
the refined pure-self-loop guard keeps E16's benefit on the non-recursive grid exactly,
and is coverage-preserving on genuinely recursive targets. The blunt
`--forbid-target-in-body` flag stays opt-in and non-recursive-only.

Artifact: `experiments/results/e16_metaqa_reread_refined_guard.json`
(commit `2ba90bb3`, same branch; contract-tested in `tests/test_e16_replay_refined.py`).

## 4. Explicit non-claims

1. **Generative half (W2 half ii) is open.** Re-mining the 22 recorded cells under the
   refined guard byte-identically requires the E11 recorded answer sessions, which are
   not in the public repo and not on the authoring machine. Until they exist, no
   generative claim is made; the classification replay above operates on committed
   artifacts only. (`TECHNICAL_EXTERNAL`, registered in advance.)
2. **Near-functional leakage is untouched.** This note addresses only structural
   self-reference and recursive targets. The E17 admission-mechanism question for
   near-functional leakage remains closed-inconclusive and is not cited as support.
3. **Scope:** synthetic DAG + oracle (result 1), classification replay (result 2),
   miner-level. No deployment, no product, no benchmark beyond MetaQA 2-hop.

## 5. Reproducibility

Both results are deterministic and replayable at `$0` from the seeds and the committed
artifact: `experiments/run_e18_recursive.py`,
`experiments/replay_e16_refined.py`, contract tests
`tests/test_e18_recursive.py`, `tests/test_e16_replay_refined.py`
(full suite 524 collected at `2ba90bb3`).
