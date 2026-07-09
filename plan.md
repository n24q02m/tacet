1. **Understand the problem**:
   - `src/tacet/distill/distill.py` has a performance bottleneck in `mine_rules_with_stats`.
   - The loop calculating length-2 rules iterates through all possible `(r1, inv1)` and `(r2, inv2)` combinations.
   - For every combination, it calculates `p1 = _adj(_directed(idx[r1], inv1))` and `p2 = _adj(_directed(idx[r2], inv2))`.
   - Rebuilding these adjacency maps inside the combinatorial loop is an $O(N^2)$ operation over the relations where it only needs to be $O(N)$ if we cache the structures.
   - This was exactly the bottleneck documented in `.jules/bolt.md` (which mentioned `src/tacet/distill/concepts.py` `induce_relations`).

2. **Propose solution**:
   - Instead of calculating `_adj(_directed(idx[r], inv))` on every iteration of the nested loop, we pre-calculate this into a cache mapping `(relation, inversion)` to its corresponding adjacency dictionary.
   - Specifically, before the length-2 body loops, compute:
     ```python
     adj_cache: dict[tuple[str, bool], dict[str, set[str]]] = {}
     for r in relations:
         for inv in (False, True):
             adj_cache[(r, inv)] = _adj(_directed(idx[r], inv))
     ```
   - Then simply lookup `p1 = adj_cache[(r1, inv1)]` and `p2 = adj_cache[(r2, inv2)]` in the inner loops.

3. **Write changes**:
   - Update `src/tacet/distill/distill.py` using `replace_with_git_merge_diff`.
   - Ensure to retain existing typing and logic.

4. **Testing**:
   - Run the full test suite using `uv run pytest` to ensure no functionality is broken.
   - Run format and lint checks (`uv run ruff format .` and `uv run ruff check .`).

5. **Pre-commit and PR**:
   - Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.
   - Submit the PR with the required Bolt PR format and measurements.
