"""Test exact rank preservation in batched evaluate vs original sequential."""

import numpy as np

from tacet.kge.kge import ComplEx


def test_evaluate_exactness():
    # Capture original evaluate output
    def evaluate_original(model, test, filter_triples=None):
        filter_triples = filter_triples or set()
        all_ent = np.arange(len(model.ent))
        filter_map = {}
        for fh, fr, ft in filter_triples:
            if (fh, fr) not in filter_map:
                filter_map[(fh, fr)] = set()
            filter_map[(fh, fr)].add(ft)
        ranks = []
        for h, r, t in test:
            if h not in model.ent or r not in model.rel or t not in model.ent:
                continue
            hi, ri, ti = model.ent[h], model.rel[r], model.ent[t]
            scores = model._phi_idx(np.full(len(all_ent), hi), np.full(len(all_ent), ri), all_ent)
            for ft in filter_map.get((h, r), set()):
                if ft != t and ft in model.ent:
                    scores[model.ent[ft]] = -np.inf
            rank = int(np.sum(scores > scores[ti]) + 1)
            ranks.append(rank)
        return ranks

    model = ComplEx()
    ne = 1000
    model.ent = {f"e{i}": i for i in range(ne)}
    model.rel = {"r1": 0, "r2": 1}
    np.random.seed(42)
    model.E_re = np.random.rand(ne, 64)
    model.E_im = np.random.rand(ne, 64)
    model.R_re = np.random.rand(2, 64)
    model.R_im = np.random.rand(2, 64)

    test_triples = [
        (f"e{np.random.randint(ne)}", "r1", f"e{np.random.randint(ne)}") for _ in range(200)
    ]

    orig_ranks = evaluate_original(model, test_triples)

    # We redefine the evaluate logic temporarily to yield ranks to compare exactly
    # because the actual model.evaluate aggregates metrics
    valid_test = [
        (h, r, t)
        for h, r, t in test_triples
        if h in model.ent and r in model.rel and t in model.ent
    ]
    batch_size = 256
    tr = model.E_re.T
    ti_im = model.E_im.T

    batched_ranks = []
    for i in range(0, len(valid_test), batch_size):
        batch = valid_test[i : i + batch_size]
        hi = np.array([model.ent[h] for h, _, _ in batch])
        ri = np.array([model.rel[r] for _, r, _ in batch])
        ti = np.array([model.ent[t] for _, _, t in batch])

        hr, hi_im_val = model.E_re[hi], model.E_im[hi]
        rr, ri_im_val = model.R_re[ri], model.R_im[ri]

        a = hr * rr - hi_im_val * ri_im_val
        b = hr * ri_im_val + hi_im_val * rr

        scores = np.dot(a, tr) + np.dot(b, ti_im)

        for j, (h, r, t) in enumerate(batch):
            rank = int(np.sum(scores[j] > scores[j, ti[j]]) + 1)
            batched_ranks.append(rank)

    assert orig_ranks == batched_ranks, f"Ranks mismatch: {orig_ranks} != {batched_ranks}"
