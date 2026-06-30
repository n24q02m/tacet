"""Tier 2 — learned structural prediction (ComplEx knowledge-graph embedding).

A vectorised NumPy implementation of ComplEx (Trouillon et al., 2016). Entities
and relations are complex vectors; a triple is scored by the multilinear form
phi(h, r, t) = Re(<h, r, conj(t)>). ComplEx is chosen over TransE because the
synthetic and real graphs here contain *symmetric* relations, which TransE
cannot represent — ComplEx models symmetric, antisymmetric and inverse
patterns alike.

The model is trained with negative sampling and a binary cross-entropy loss
via Adam. For the cascade it exposes `predict_tail`, which ranks type-valid
candidates and attaches a *temperature-calibrated* confidence (Guo et al.,
2017) that the router thresholds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Triple = tuple[str, str, str]


@dataclass
class Prediction:
    tail: str
    confidence: float
    ranking: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class KGEConfig:
    """ComplEx hyper-parameters.

    ``negatives`` controls type-correct (relation-pool) negatives, which
    teach within-type discrimination — exactly what the cascade's Tier-2
    candidate ranking needs. ``uniform_negatives`` adds uniformly-corrupted
    negatives, which teach the model to suppress *out-of-type* entities and
    is essential for the standard open-world link-prediction protocol
    (filtered MRR / Hits@k across all entities).
    """

    dim: int = 64
    epochs: int = 100
    lr: float = 0.05
    negatives: int = 6
    uniform_negatives: int = 4
    batch_size: int = 0  # 0 = full-batch; set 2048+ for FB15k-scale data
    reg: float = 1e-6
    seed: int = 0


def _scatter_add(idx: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Sum ``values`` rows into ``n`` buckets indexed by ``idx``.

    Two strategies, chosen by memory footprint:

    * **One-hot matmul** for small ``n`` — a single BLAS call, very fast.
      Memory: ``L * n`` floats.
    * **`np.add.at`** for large ``n`` (the FB15k-237 / WN18RR scale) —
      slower but uses only ``L`` extra floats, so the model trains in
      bounded memory regardless of entity count.

    The threshold ``L * n ≤ 4e7`` (~300 MB at 8 B/float) was chosen so a
    14 k-entity benchmark with a 2 M-row batch automatically takes the
    safe path.
    """
    out = np.zeros((n, values.shape[1]))
    if len(idx) * n <= 40_000_000:
        onehot = np.zeros((len(idx), n))
        onehot[np.arange(len(idx)), idx] = 1.0
        np.matmul(onehot.T, values, out=out)
        return out
    np.add.at(out, idx, values)
    return out


class ComplEx:
    """ComplEx KGE model: phi(h,r,t) = Re(sum_d h_d * r_d * conj(t_d))."""

    def __init__(self, config: KGEConfig | None = None) -> None:
        self.cfg = config or KGEConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.ent: dict[str, int] = {}
        self.rel: dict[str, int] = {}
        self.E_re: np.ndarray = np.zeros((0, 0))
        self.E_im: np.ndarray = np.zeros((0, 0))
        self.R_re: np.ndarray = np.zeros((0, 0))
        self.R_im: np.ndarray = np.zeros((0, 0))
        self.loss_history: list[float] = []
        self.temperature: float = 1.0

    # ------------------------------------------------------------- scoring
    def _phi_idx(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Vectorised score for index arrays h, r, t of equal length."""
        hr, hi = self.E_re[h], self.E_im[h]
        rr, ri = self.R_re[r], self.R_im[r]
        tr, ti = self.E_re[t], self.E_im[t]
        a = hr * rr - hi * ri
        b = hr * ri + hi * rr
        return np.sum(a * tr + b * ti, axis=-1)

    def score(self, h: str, r: str, t: str) -> float:
        if h not in self.ent or r not in self.rel or t not in self.ent:
            return float("-inf")
        return float(
            self._phi_idx(
                np.array([self.ent[h]]), np.array([self.rel[r]]), np.array([self.ent[t]])
            )[0]
        )

    # ------------------------------------------------------------- training
    def fit(self, triples: list[Triple], epochs: int | None = None) -> ComplEx:
        """Train ComplEx from scratch with Adam + negative sampling (BCE loss).

        Fully vectorised: every epoch builds one batch of positives and
        type-correct negatives, scores it, and scatters gradients via a
        one-hot matmul.
        """
        ents = sorted({x for h, _, t in triples for x in (h, t)})
        rels = sorted({r for _, r, _ in triples})
        self.ent = {e: i for i, e in enumerate(ents)}
        self.rel = {r: i for i, r in enumerate(rels)}
        ne, nr, d = len(ents), len(rels), self.cfg.dim
        if ne < 2 or nr < 1 or not triples:
            return self
        scale = 1.0 / np.sqrt(d)
        self.E_re = self._rng.normal(0, scale, (ne, d))
        self.E_im = self._rng.normal(0, scale, (ne, d))
        self.R_re = self._rng.normal(0, scale, (nr, d))
        self.R_im = self._rng.normal(0, scale, (nr, d))
        self._train(triples, epochs if epochs is not None else self.cfg.epochs)
        return self

    def partial_fit(self, triples: list[Triple], epochs: int = 25) -> ComplEx:
        """Warm-start: keep converged embeddings, extend for new symbols, refine.

        Used by `TACET.consolidate` so absorbing teacher facts costs a short
        refinement rather than a full retrain (continuous learning).
        """
        if not self.ent:
            return self.fit(triples, epochs)
        d, scale = self.cfg.dim, 1.0 / np.sqrt(self.cfg.dim)
        for e in sorted({x for h, _, t in triples for x in (h, t)}):
            if e not in self.ent:
                self.ent[e] = len(self.ent)
                self.E_re = np.vstack([self.E_re, self._rng.normal(0, scale, (1, d))])
                self.E_im = np.vstack([self.E_im, self._rng.normal(0, scale, (1, d))])
        for r in sorted({r for _, r, _ in triples}):
            if r not in self.rel:
                self.rel[r] = len(self.rel)
                self.R_re = np.vstack([self.R_re, self._rng.normal(0, scale, (1, d))])
                self.R_im = np.vstack([self.R_im, self._rng.normal(0, scale, (1, d))])
        self._train(triples, epochs)
        return self

    def _train(self, triples: list[Triple], epochs: int) -> None:
        """Run `epochs` of Adam over `triples` on the current parameters.

        Mini-batches when ``cfg.batch_size > 0`` (recommended at >10 k
        triples); falls back to full-batch otherwise to preserve the
        backwards-compatible behaviour for the synthetic benchmark.
        """
        ne, nr = len(self.ent), len(self.rel)
        h = np.array([self.ent[a] for a, _, _ in triples])
        r = np.array([self.rel[b] for _, b, _ in triples])
        t = np.array([self.ent[c] for _, _, c in triples])
        p, k = len(triples), self.cfg.negatives
        ku = self.cfg.uniform_negatives
        if p == 0:
            return

        # relation-specific (type-correct) negative pools.
        tail_pool = [
            np.array(sorted({t[i] for i in range(p) if r[i] == ri}) or [0]) for ri in range(nr)
        ]
        head_pool = [
            np.array(sorted({h[i] for i in range(p) if r[i] == ri}) or [0]) for ri in range(nr)
        ]

        params = [self.E_re, self.E_im, self.R_re, self.R_im]
        m = [np.zeros_like(x) for x in params]
        v = [np.zeros_like(x) for x in params]
        b1, b2, eps = 0.9, 0.999, 1e-8

        batch = self.cfg.batch_size if self.cfg.batch_size > 0 else p
        step = 0
        idx_all = np.arange(p)

        for _ in range(1, epochs + 1):
            self._rng.shuffle(idx_all)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, p, batch):
                bidx = idx_all[start : start + batch]
                bh = h[bidx]
                br = r[bidx]
                bt = t[bidx]
                bp = len(bidx)

                # type-correct negatives — sample per relation present in batch.
                nh = np.repeat(bh[:, None], k, axis=1)
                nt = np.repeat(bt[:, None], k, axis=1)
                for ri in np.unique(br):
                    mask = br == ri
                    rows = np.where(mask)[0]
                    tp, hp = tail_pool[int(ri)], head_pool[int(ri)]
                    nt[rows] = tp[self._rng.integers(0, len(tp), (len(rows), k))]
                    nh[rows] = hp[self._rng.integers(0, len(hp), (len(rows), k))]
                ch = self._rng.random((bp, k)) < 0.5
                nh = np.where(ch, nh, bh[:, None])
                nt = np.where(ch, bt[:, None], nt)

                # uniform negatives
                if ku:
                    ut = self._rng.integers(0, ne, (bp, ku))
                    uh_rand = self._rng.integers(0, ne, (bp, ku))
                    cu = self._rng.random((bp, ku)) < 0.5
                    uh = np.where(cu, uh_rand, bh[:, None])
                    ut = np.where(cu, bt[:, None], ut)
                else:
                    uh = np.zeros((bp, 0), dtype=int)
                    ut = np.zeros((bp, 0), dtype=int)

                H = np.concatenate([bh, nh.ravel(), uh.ravel()])
                T = np.concatenate([bt, nt.ravel(), ut.ravel()])
                R = np.concatenate(
                    [
                        br,
                        np.repeat(br[:, None], k, axis=1).ravel(),
                        np.repeat(br[:, None], ku, axis=1).ravel(),
                    ]
                )
                labels = np.concatenate([np.ones(bp), np.zeros(bp * (k + ku))])

                hr, hi = self.E_re[H], self.E_im[H]
                rr, ri = self.R_re[R], self.R_im[R]
                tr, ti = self.E_re[T], self.E_im[T]
                a = hr * rr - hi * ri
                b = hr * ri + hi * rr
                phi = np.sum(a * tr + b * ti, axis=1)
                sig = 1.0 / (1.0 + np.exp(-np.clip(phi, -30, 30)))
                epoch_loss += float(np.mean(np.logaddexp(0.0, phi) - labels * phi))
                n_batches += 1

                dphi = (sig - labels)[:, None]
                g_h_re = dphi * (rr * tr + ri * ti)
                g_h_im = dphi * (-ri * tr + rr * ti)
                g_t_re = dphi * a
                g_t_im = dphi * b
                g_r_re = dphi * (hr * tr + hi * ti)
                g_r_im = dphi * (-hi * tr + hr * ti)

                ent_idx = np.concatenate([H, T])
                grads = [
                    _scatter_add(ent_idx, np.concatenate([g_h_re, g_t_re]), ne) / bp,
                    _scatter_add(ent_idx, np.concatenate([g_h_im, g_t_im]), ne) / bp,
                    _scatter_add(R, g_r_re, nr) / bp,
                    _scatter_add(R, g_r_im, nr) / bp,
                ]
                step += 1
                for i, (param, grad) in enumerate(zip(params, grads, strict=True)):
                    grad = grad + self.cfg.reg * param
                    m[i] = b1 * m[i] + (1 - b1) * grad
                    v[i] = b2 * v[i] + (1 - b2) * grad * grad
                    mhat = m[i] / (1 - b1**step)
                    vhat = v[i] / (1 - b2**step)
                    param -= self.cfg.lr * mhat / (np.sqrt(vhat) + eps)

            self.loss_history.append(epoch_loss / max(n_batches, 1))

    # ------------------------------------------------------------- inference
    def predict_tail(
        self, head: str, relation: str, candidates: list[str], top_k: int = 5
    ) -> Prediction | None:
        """Rank `candidates` as the tail of (head, relation) with calibrated confidence."""
        if head not in self.ent or relation not in self.rel:
            return None
        usable = [c for c in candidates if c in self.ent and c != head]
        if not usable:
            return None
        h = np.full(len(usable), self.ent[head])
        r = np.full(len(usable), self.rel[relation])
        t = np.array([self.ent[c] for c in usable])
        scores = self._phi_idx(h, r, t)
        order = np.argsort(-scores)
        z = scores[order] / self.temperature
        z = z - z.max()
        probs = np.exp(z) / np.exp(z).sum()
        return Prediction(
            tail=usable[order[0]],
            confidence=float(probs[0]),
            ranking=[
                (usable[order[i]], float(scores[order[i]])) for i in range(min(top_k, len(usable)))
            ],
        )

    # ------------------------------------------------------------- calibration
    def calibrate(self, val: list[tuple[str, str, list[str], str]]) -> float:
        """Temperature scaling: fit T minimising NLL on validation queries.

        `val` is a list of (head, relation, candidates, true_tail).
        """
        rows = []
        for head, relation, cands, truth in val:
            if head not in self.ent or relation not in self.rel:
                continue
            usable = [c for c in cands if c in self.ent]
            if truth not in usable:
                continue
            h = np.full(len(usable), self.ent[head])
            r = np.full(len(usable), self.rel[relation])
            t = np.array([self.ent[c] for c in usable])
            rows.append((self._phi_idx(h, r, t), usable.index(truth)))
        if not rows:
            return self.temperature

        best_t, best_nll = 1.0, float("inf")
        for temp in np.linspace(0.1, 6.0, 60):
            nll = 0.0
            for scores, idx in rows:
                z = scores / temp
                z = z - z.max()
                p = np.exp(z) / np.exp(z).sum()
                nll -= np.log(max(p[idx], 1e-12))
            if nll < best_nll:
                best_nll, best_t = nll, float(temp)
        self.temperature = best_t
        return best_t

    # ------------------------------------------------------------- evaluation
    def evaluate(
        self, test: list[Triple], filter_triples: set[Triple] | None = None
    ) -> dict[str, float]:
        """Filtered tail-ranking metrics: MRR and Hits@{1,3,10}."""
        filter_triples = filter_triples or set()
        all_ent = np.arange(len(self.ent))
        # Pre-index filter triples for fast lookup
        filter_map: dict[tuple[str, str], set[str]] = {}
        for fh, fr, ft in filter_triples:
            if (fh, fr) not in filter_map:
                filter_map[(fh, fr)] = set()
            filter_map[(fh, fr)].add(ft)
        ranks: list[int] = []
        for h, r, t in test:
            if h not in self.ent or r not in self.rel or t not in self.ent:
                continue
            hi, ri, ti = self.ent[h], self.rel[r], self.ent[t]
            scores = self._phi_idx(np.full(len(all_ent), hi), np.full(len(all_ent), ri), all_ent)
            for ft in filter_map.get((h, r), set()):
                if ft != t and ft in self.ent:
                    scores[self.ent[ft]] = -np.inf
            rank = int(np.sum(scores > scores[ti]) + 1)
            ranks.append(rank)
        if not ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        arr = np.array(ranks)
        return {
            "MRR": float(np.mean(1.0 / arr)),
            "Hits@1": float(np.mean(arr <= 1)),
            "Hits@3": float(np.mean(arr <= 3)),
            "Hits@10": float(np.mean(arr <= 10)),
        }
