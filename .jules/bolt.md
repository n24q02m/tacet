## 2024-06-30 - Optimize path-mining relation lookup
**Learning:** Found a performance bottleneck specific to this codebase's architecture in `src/tacet/distill/concepts.py` where `induce_relations` rebuilds forward adjacency structures repeatedly. By pre-computing these maps once, the complexity dropped dramatically from $O(|R|^2 \times \text{pairs})$ to $O(|R| \times \text{pairs})$. The benchmark showed an improvement from 32s to 26s for 20 dense relations.
**Action:** Always check for repeated graph traversal allocations or rebuilds inside nested loops when dealing with multi-relational graphs.
## 2024-07-01 - Optimize path-mining adjacency lookups in distill
**Learning:** Found a performance bottleneck in `src/tacet/distill/distill.py` similar to the one in `concepts.py` where `mine_rules_with_stats` rebuilt forward adjacency structures repeatedly in combinatorial length-2 path loops. By pre-computing these maps once, the complexity dropped from $O(|R|^2 \times \text{pairs})$ to $O(|R| \times \text{pairs})$.
**Action:** Pre-compute graph adjacency representations outside of multi-relational combinatoric nested loops whenever possible.
