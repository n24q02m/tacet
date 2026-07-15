## 2024-05-24 - Faster Node Signatures
**Learning:** In sparse graphs with many relation types, computing structural signatures by iterating over all relations (`graph.relations()`) and then fetching edges is O(|Relations| * Degree). This is an anti-pattern when the internal graph implementation already indexes edges by relation.
**Action:** When computing properties that only depend on existing edges, access the internal adjacency dictionaries (`graph._out` and `graph._in`) directly. This guarantees O(Degree) traversal time.
