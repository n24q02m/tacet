import unittest

import numpy as np
import torch

from tacet.kge.kge import ComplEx, KGEConfig
from tacet.kge.kge_torch import _HAS_TORCH, TorchComplEx, TorchKGEConfig


class TestKGEBitIdentical(unittest.TestCase):
    def test_numpy_bit_identical_ranks(self):
        triples = [(f"e{i % 20}", f"r{i % 5}", f"e{(i + 1) % 20}") for i in range(200)]
        test_triples = [(f"e{i % 20}", f"r{i % 5}", f"e{(i + 1) % 20}") for i in range(100)]

        model = ComplEx(KGEConfig(dim=50, epochs=5, seed=42))
        model.fit(triples)

        # Capture old behaviour: unbatched looped evaluation.
        all_ent = np.arange(len(model.ent))
        ranks_old = []
        for h, r, t in test_triples:
            if h not in model.ent or r not in model.rel or t not in model.ent:
                continue
            hi, ri, ti = model.ent[h], model.rel[r], model.ent[t]
            scores = model._phi_idx(np.full(len(all_ent), hi), np.full(len(all_ent), ri), all_ent)
            rank = int(np.sum(scores > scores[ti]) + 1)
            ranks_old.append(rank)

        # Monkey patch evaluate back to new version (to grab ranks)
        def evaluate_ranks(self, test, filter_triples=None):
            filter_triples = filter_triples or set()
            filter_map = {}
            for fh, fr, ft in filter_triples:
                if (fh, fr) not in filter_map:
                    filter_map[(fh, fr)] = set()
                filter_map[(fh, fr)].add(ft)
            ranks = []

            valid_test = [
                (h, r, t) for h, r, t in test if h in self.ent and r in self.rel and t in self.ent
            ]
            if not valid_test:
                return []

            all_ent_idx = np.arange(len(self.ent))[None, :]
            batch_size = 128
            for start in range(0, len(valid_test), batch_size):
                batch = valid_test[start : start + batch_size]
                b_h = np.array([self.ent[h] for h, r, t in batch])[:, None]
                b_r = np.array([self.rel[r] for h, r, t in batch])[:, None]
                b_t = np.array([self.ent[t] for h, r, t in batch])

                scores = self._phi_idx(b_h, b_r, all_ent_idx)

                for i, (h, r, t) in enumerate(batch):
                    for ft in filter_map.get((h, r), set()):
                        if ft != t and ft in self.ent:
                            scores[i, self.ent[ft]] = -np.inf

                    ti = b_t[i]
                    rank = int(np.sum(scores[i] > scores[i, ti]) + 1)
                    ranks.append(rank)
            return ranks

        model.evaluate_ranks = evaluate_ranks.__get__(model)
        ranks_new = model.evaluate_ranks(test_triples)
        self.assertEqual(ranks_old, ranks_new)

    @unittest.skipUnless(_HAS_TORCH, "Torch required")
    def test_torch_bit_identical_ranks(self):
        triples = [(f"e{i % 20}", f"r{i % 5}", f"e{(i + 1) % 20}") for i in range(200)]
        test_triples = [(f"e{i % 20}", f"r{i % 5}", f"e{(i + 1) % 20}") for i in range(100)]

        cfg = TorchKGEConfig(dim=50, epochs=5, seed=42, device="cpu", score_fn="complex")
        model = TorchComplEx(cfg)
        model.fit(triples)

        ne = len(model.ent)
        all_ent_idx = torch.arange(ne, device=model.device)
        ranks_old = []
        with torch.no_grad():
            for h, r, t in test_triples:
                if h not in model.ent or r not in model.rel or t not in model.ent:
                    continue
                hi, ri, ti = model.ent[h], model.rel[r], model.ent[t]
                hh = torch.full((ne,), hi, device=model.device)
                rr = torch.full((ne,), ri, device=model.device)
                scores = model._phi_idx(hh, rr, all_ent_idx).cpu().numpy()
                rank = int((scores > scores[ti]).sum() + 1)
                ranks_old.append(rank)

        def evaluate_ranks(self, test, filter_triples=None):
            filter_triples = filter_triples or set()
            filter_map = {}
            for fh, fr, ft in filter_triples:
                if (fh, fr) not in filter_map:
                    filter_map[(fh, fr)] = set()
                filter_map[(fh, fr)].add(ft)
            ranks = []

            valid_test = [
                (h, r, t) for h, r, t in test if h in self.ent and r in self.rel and t in self.ent
            ]
            if not valid_test:
                return []

            batch_size = 128
            all_ent_idx2 = torch.arange(ne, device=self.device).unsqueeze(0)
            with torch.no_grad():
                for start in range(0, len(valid_test), batch_size):
                    batch = valid_test[start : start + batch_size]
                    b_h = torch.tensor(
                        [self.ent[h] for h, r, t in batch], device=self.device
                    ).unsqueeze(1)
                    b_r = torch.tensor(
                        [self.rel[r] for h, r, t in batch], device=self.device
                    ).unsqueeze(1)
                    b_t = torch.tensor([self.ent[t] for h, r, t in batch], device=self.device)

                    scores = self._phi_idx(b_h, b_r, all_ent_idx2).cpu().numpy()
                    b_t_np = b_t.cpu().numpy()

                    for i, (h, r, t) in enumerate(batch):
                        for ft in filter_map.get((h, r), set()):
                            if ft != t and ft in self.ent:
                                scores[i, self.ent[ft]] = -np.inf

                        ti = b_t_np[i]
                        rank = int(np.sum(scores[i] > scores[i, ti]) + 1)
                        ranks.append(rank)
            return ranks

        model.evaluate_ranks = evaluate_ranks.__get__(model)
        ranks_new = model.evaluate_ranks(test_triples)
        self.assertEqual(ranks_old, ranks_new)
