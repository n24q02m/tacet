## 2024-06-30 - Optimize path-mining relation lookup
**Learning:** Found a performance bottleneck specific to this codebase's architecture in `src/tacet/distill/concepts.py` where `induce_relations` rebuilds forward adjacency structures repeatedly. By pre-computing these maps once, the complexity dropped dramatically from $O(|R|^2 \times \text{pairs})$ to $O(|R| \times \text{pairs})$. The benchmark showed an improvement from 32s to 26s for 20 dense relations.
**Action:** Always check for repeated graph traversal allocations or rebuilds inside nested loops when dealing with multi-relational graphs.
## 2024-07-01 - Optimize path-mining for rule synthesis
**Learning:** In `src/tacet/distill/distill.py`, `mine_rules_with_stats` used to reconstruct adjacency maps $O(|R|^2)$ times within nested loops while proposing length-2 Horn rules. By pre-computing these maps once outside the loop and additionally filtering head entities earlier rather than via a delayed set intersection, the time on a dense benchmark graph decreased from 3.47s to 0.73s.
**Action:** When evaluating graph combinations ($R_1 \land R_2$) in a combinatorial loop, always lift structural calculations (like direct and inverse adjacency maps) to the top of the loop hierarchy.
## 2024-05-24 - Faster Node Signatures
**Learning:** In sparse graphs with many relation types, computing structural signatures by iterating over all relations (`graph.relations()`) and then fetching edges is O(|Relations| * Degree). This is an anti-pattern when the internal graph implementation already indexes edges by relation.
**Action:** When computing properties that only depend on existing edges, access the internal adjacency dictionaries (`graph._out` and `graph._in`) directly. This guarantees O(Degree) traversal time.
