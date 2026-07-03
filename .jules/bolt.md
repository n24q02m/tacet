## 2024-06-30 - Optimize path-mining relation lookup
**Learning:** Found a performance bottleneck specific to this codebase's architecture in `src/tacet/distill/concepts.py` where `induce_relations` rebuilds forward adjacency structures repeatedly. By pre-computing these maps once, the complexity dropped dramatically from $O(|R|^2 \times \text{pairs})$ to $O(|R| \times \text{pairs})$. The benchmark showed an improvement from 32s to 26s for 20 dense relations.
**Action:** Always check for repeated graph traversal allocations or rebuilds inside nested loops when dealing with multi-relational graphs.
## 2024-07-24 - [Pre-computing adjacency maps in combinatorial mining]
**Learning:** In `distill.py` rule synthesis, the length-2 path miner redundantly built adjacency maps `_adj(_directed(...))` inside an O(N^2) combinatorial loop, resulting in a performance bottleneck as relation numbers scale.
**Action:** Always pre-compute structural index structures, like forward and backward adjacency lists, outside nested combinatorial loops when mining length-2 horn rules or paths.
