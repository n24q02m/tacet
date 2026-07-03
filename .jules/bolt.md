## 2024-06-30 - Optimize path-mining relation lookup
**Learning:** Found a performance bottleneck specific to this codebase's architecture in `src/tacet/distill/concepts.py` where `induce_relations` rebuilds forward adjacency structures repeatedly. By pre-computing these maps once, the complexity dropped dramatically from $O(|R|^2 \times \text{pairs})$ to $O(|R| \times \text{pairs})$. The benchmark showed an improvement from 32s to 26s for 20 dense relations.
**Action:** Always check for repeated graph traversal allocations or rebuilds inside nested loops when dealing with multi-relational graphs.

## 2024-07-03 - Avoid O(R^2) dictionary creation in rule synthesis
**Learning:** Found a performance bottleneck specific to this codebase's architecture in `src/tacet/distill/distill.py` where `mine_rules` rebuilt forward adjacency structures repeatedly in its length-2 combinatorial loops. The complexity of computing the adjacency dict scaled at $O(|R|^2 \times \text{pairs})$, but by precomputing it for all `R` and its inverses beforehand, we avoid expensive dictionary initialization in the inner loops, optimizing graph mining execution times.
**Action:** Pre-compute lookup structures (like adj maps) outside combinatorial loops to prevent redundant object creation scaling quadratically during rule mining or induction tasks.
