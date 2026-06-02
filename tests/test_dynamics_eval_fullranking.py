import numpy as np

from tacet.experimental.dynamics.eval import filtered_rank


def test_filtered_rank_excludes_other_true_tails():
    # scores indexed by entity id; gold=2 (0.5). Entities 1 (0.9) and 4 (0.8)
    # outscore it. Filtering true tail 4 leaves only 1 above gold -> rank 2.
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    assert filtered_rank(scores, gold_idx=2, filter_idx={4}) == 2


def test_filtered_rank_top_is_rank_one():
    scores = np.array([0.1, 0.2, 0.9])
    assert filtered_rank(scores, gold_idx=2, filter_idx=set()) == 1


def test_filtered_rank_no_filter_counts_all_better():
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    # gold=2 beaten by 1 and 4 -> rank 3
    assert filtered_rank(scores, gold_idx=2, filter_idx=set()) == 3
