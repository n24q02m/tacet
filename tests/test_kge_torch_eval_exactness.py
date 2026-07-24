"""Test exact rank preservation in batched evaluate vs original sequential."""

import numpy as np
import pytest

from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_evaluate_torch_exactness():
    def evaluate_original(model, test, filter_triples=None):
        filter_triples = filter_triples or set()
        ne = len(model.ent)
        all_ent_idx = torch.arange(ne, device=model.device)
        ranks = []
        with torch.no_grad():
            for h, r, t in test:
                if h not in model.ent or r not in model.rel or t not in model.ent:
                    continue
                hi, ri, ti = model.ent[h], model.rel[r], model.ent[t]
                hh = torch.full((ne,), hi, device=model.device)
                rr = torch.full((ne,), ri, device=model.device)
                scores = model._phi_idx(hh, rr, all_ent_idx).cpu().numpy()
                rank = int((scores > scores[ti]).sum() + 1)
                ranks.append(rank)
        return ranks

    model = TorchComplEx(TorchKGEConfig(score_fn="complex", device="cpu"))
    ne = 1000
    model.ent = {f"e{i}": i for i in range(ne)}
    model.rel = {"r1": 0, "r2": 1}
    torch.manual_seed(42)
    model._E_re = torch.rand(ne, 64)
    model._E_im = torch.rand(ne, 64)
    model._R_re = torch.rand(2, 64)
    model._R_im = torch.rand(2, 64)

    test_triples = [
        (f"e{np.random.randint(ne)}", "r1", f"e{np.random.randint(ne)}") for _ in range(200)
    ]

    orig_ranks = evaluate_original(model, test_triples)

    valid_test = [
        (h, r, t)
        for h, r, t in test_triples
        if h in model.ent and r in model.rel and t in model.ent
    ]
    batch_size = 256

    batched_ranks = []
    with torch.no_grad():
        tr = model._E_re
        ti_im = model._E_im
        for i in range(0, len(valid_test), batch_size):
            batch = valid_test[i : i + batch_size]
            hi = torch.tensor([model.ent[h] for h, _, _ in batch], device=model.device)
            ri = torch.tensor([model.rel[r] for _, r, _ in batch], device=model.device)
            ti = np.array([model.ent[t] for _, _, t in batch])

            hr, hi_im_val = model._E_re[hi], model._E_im[hi]
            rr, ri_im_val = model._R_re[ri], model._R_im[ri]
            a = hr * rr - hi_im_val * ri_im_val
            b = hr * ri_im_val + hi_im_val * rr
            scores = (a @ tr.T + b @ ti_im.T).cpu().numpy()

            for j, (h, r, t) in enumerate(batch):
                rank = int(np.sum(scores[j] > scores[j, ti[j]]) + 1)
                batched_ranks.append(rank)

    assert orig_ranks == batched_ranks, f"Ranks mismatch: {orig_ranks} != {batched_ranks}"
