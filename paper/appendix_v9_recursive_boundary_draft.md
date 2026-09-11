# Appendix note — DRAFT for v9: Recursive targets and the pure-self-loop guard

> **Status:** `DRAFT — NOT PUBLISHED`. Cutting a Zenodo v9 with this note is a separate
> publication decision. Nothing in this file is a claim until that gate opens.
> Prepared from committed, replayable `$0` artifacts only (E18 synthetic grid + E18-W2
> classification replay + E18-W2 generative replay). No provider was contacted at any point.

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

## 4. New result 3 — generative re-mining on the real grid (E18-W2, half ii)

Re-mining all 22 recorded cells at gamma 0.50 under the refined guard — the teacher
answers replayed from the E11 recordings (recovered from the authoring archive; used
untracked, never committed to the public repo), zero provider calls:

- control arm reproduces the published shape exactly under the refined taxonomy
  (**14 `pure_self_loop` rules over exactly the 7 published junk cells; true installs
  10/10**);
- refined arm: **0 `pure_self_loop`, 0 `mixed_recursive`** across all 22 cells; true
  compositions **10/10 kept** with identical per-rule world precision; no valid cell
  with full accuracy below cache accuracy; 0 invalid cells.

Decision (locked before the run): **GENERATIVE-SUBSUMES** — the refined guard's junk
removal on the real grid is not a classification artifact of committed rule lists; it
holds when the miner runs fresh under the guard. Artifact:
`experiments/results/e18_w2_generative_refined.json` (commit `889df9cd`, same branch;
runner plumbing and contract test in the same commit).

## 5. Explicit non-claims
1. **Near-functional leakage is untouched.** This note addresses only structural
   self-reference and recursive targets. The E17 admission-mechanism question for
   near-functional leakage remains closed-inconclusive and is not cited as support.
2. **Scope:** synthetic DAG + oracle (result 1), classification replay (result 2),
   generative replay (result 3), miner-level. No deployment, no product, no benchmark
   beyond MetaQA 2-hop.
3. **Data provenance:** the E11 answer recordings stay in the authoring archive; the
   public artifact carries per-cell outcomes only, never raw paid answers.

## 6. Reproducibility
All three results are deterministic and replayable at `$0` from the seeds and the
committed artifacts: `experiments/run_e18_recursive.py`,
`experiments/replay_e16_refined.py`, `experiments/run_real_kg_controlled.py`
(replay mode), contract tests `tests/test_e18_recursive.py`,
`tests/test_e16_replay_refined.py`, `tests/test_controlled_selfloop_plumbing.py`
(full suite 526 collected at `889df9cd`).

## 7. Main-text deltas to apply ONLY at the v9 cut

Kept here so the publication edit is turnkey; none of these touch `main` before the
Zenodo v9 exists (the published record stays v8-consistent until then):

1. The E16 registered-limitation paragraph in the main text gains a pointer to this
   appendix: the blunt ban stays opt-in, and the refined pure-self-loop guard is the
   replacement for workloads with genuinely recursive targets (results 1-3).
2. Abstract/claims sync: "safe on non-recursive targets only" becomes "junk removal
   preserved exactly on non-recursive grids and extended to genuinely recursive
   targets via the refined guard", citing results 1-3.
3. `CITATION.cff`: version v8 -> v9, version DOI -> the new Zenodo version DOI minted
   at the cut (concept DOI unchanged).
4. README claim paragraph + reproducibility commands sync to v9 (suite count at the
   cut commit, the three artifact files, the three runners).

## 8. What v9 deliberately does NOT contain

- Any E17 admission-mechanism claim (closed-inconclusive; not cited).
- The E11 answer recordings (authoring archive only; outcomes-only artifact).
- Anything from the ideas portfolio (E13/E14 replay analyses, E15, VIGIL intake) —
  none of it is approved research output yet.
