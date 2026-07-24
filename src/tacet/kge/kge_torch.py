"""PyTorch KGE backend — same public API as ``kge.ComplEx``, runs on GPU.

Use when scaling to FB15k-237 / WN18RR / WikiKG and friends. The NumPy
``ComplEx`` is fine for tens of thousands of triples; for the hundreds of
thousands to millions a Tier-2 production system actually serves you want
mini-batch SGD on accelerator hardware.

Mirror the NumPy class API exactly so callers swap in one line:

    from tacet.kge.kge import ComplEx           # CPU / NumPy
    from tacet.kge.kge_torch import TorchComplEx as ComplEx   # GPU / PyTorch

Three scorers + losses are shipped (``score_fn`` config):

* ``complex`` (default, Trouillon et al. 2016) — multilinear ComplEx scoring
  with BCE-with-logits loss over type-correct + uniform negatives.  Cheap and
  stable; matches the prior published behaviour of the framework.
* ``complex_n3`` (Lacroix et al. 2018) — same multilinear ComplEx scoring but
  trained with the **1-vs-N cross-entropy loss** (softmax NLL over the true
  tail + the sampled negatives) plus the **N3 nuclear-norm regularizer**
  ``λ·Σ(|h|³ + |r|³ + |t|³)`` on positive triples.  This is the recipe that
  pushed ComplEx to literature SOTA on FB15k-237 and WN18RR; expect MRR
  improvement vs the BCE baseline at the same dim / epochs.
* ``rotate`` (Sun et al. 2019, ICLR) — rotation-in-complex-plane scoring with
  the **margin-based loss + self-adversarial negative sampling** from the
  paper (γ - d for positive, weighted sum of σ(d - γ) for negatives where
  weights are softmax(α · -d) over the negative scores).  This is the
  recipe that lets RotatE express composition / inversion / asymmetry.

PyTorch is imported lazily; if torch is not installed, instantiation raises
``ImportError`` with the install command.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:  # noqa: SIM105
    import torch  # type: ignore[import-not-found]

    _HAS_TORCH = True
except (ImportError, OSError):  # pragma: no cover - optional
    # OSError catches Windows DLL initialisation failures (WinError 1114
    # when torch's ``c10.dll`` cannot load — typically a missing
    # ``VC++ Redistributable``).  Treat the backend as unavailable rather
    # than crashing the whole tacet package.
    _HAS_TORCH = False

import numpy as np

Triple = tuple[str, str, str]


@dataclass
class Prediction:
    tail: str
    confidence: float
    ranking: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class TorchKGEConfig:
    dim: int = 64
    epochs: int = 200
    lr: float = 5e-3
    batch_size: int = 1024
    negatives: int = 6
    uniform_negatives: int = 4
    reg: float = 1e-6  # Adam weight_decay (BCE path); 0 disables
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps
    score_fn: str = "complex"  # complex | complex_n3 | rotate
    margin: float = 12.0  # γ for RotatE (Sun 2019: 12-24 typical)
    alpha: float = 1.0  # Self-adversarial temperature (RotatE)
    n3_lambda: float = 1e-2  # Lacroix N3 regularizer weight (complex_n3)
    log_every: int = 0  # 0 = quiet


def _resolve_device(name: str):
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("torch not installed")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class TorchComplEx:
    """GPU-trainable ComplEx (and optional RotatE) — public API matches the NumPy version."""

    def __init__(self, config: TorchKGEConfig | None = None) -> None:
        if not _HAS_TORCH:  # pragma: no cover - optional
            raise ImportError(
                "TorchComplEx requires PyTorch. "
                "Install with `pip install torch` (or use the `kge.ComplEx` NumPy backend)."
            )
        self.cfg = config or TorchKGEConfig()
        torch.manual_seed(self.cfg.seed)
        self.device = _resolve_device(self.cfg.device)
        self.ent: dict[str, int] = {}
        self.rel: dict[str, int] = {}
        self._E_re: torch.Tensor | None = None
        self._E_im: torch.Tensor | None = None
        self._R_re: torch.Tensor | None = None
        self._R_im: torch.Tensor | None = None
        self.loss_history: list[float] = []
        self.temperature: float = 1.0
        # Optional text-attributed init: dict[entity_name -> np.ndarray of
        # shape (text_dim,)].  Used by ``fit`` to seed ``_E_re`` with a
        # linearly-projected text embedding before structural training
        # proceeds normally — see ``kge_textual`` for the factory + tests.
        self._text_init: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------- scoring
    def _phi_idx(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hr, hi = self._E_re[h], self._E_im[h]
        rr, ri = self._R_re[r], self._R_im[r]
        tr, ti = self._E_re[t], self._E_im[t]
        if self.cfg.score_fn == "rotate":
            # constrain |r|=1 ⇒ r = (cos θ, sin θ); approximate by L2-normalising.
            mag = (rr * rr + ri * ri).clamp_min(1e-12).sqrt()
            rr_n, ri_n = rr / mag, ri / mag
            re_diff = hr * rr_n - hi * ri_n - tr
            im_diff = hr * ri_n + hi * rr_n - ti
            d = (re_diff.abs() + im_diff.abs()).sum(-1)
            return self.cfg.margin - d
        # ComplEx multilinear
        a = hr * rr - hi * ri
        b = hr * ri + hi * rr
        return (a * tr + b * ti).sum(-1)

    def score(self, h: str, r: str, t: str) -> float:
        if h not in self.ent or r not in self.rel or t not in self.ent:
            return float("-inf")
        with torch.no_grad():
            hh = torch.tensor([self.ent[h]], device=self.device)
            rr = torch.tensor([self.rel[r]], device=self.device)
            tt = torch.tensor([self.ent[t]], device=self.device)
            return float(self._phi_idx(hh, rr, tt).item())

    def set_text_init(self, embeddings: dict[str, np.ndarray]) -> TorchComplEx:
        """Provide pre-computed text embeddings (entity → vector) for init.

        The mapping is consumed at the next ``fit`` call: entities present
        in the dict have their ``_E_re`` row seeded from a linear
        projection of the text embedding to the KGE dimension; entities
        absent from the dict (or the call as a whole) get the random
        init used in vanilla ComplEx.  ``_E_im`` always starts random so
        the multilinear interaction has a chance to learn asymmetry.
        """
        self._text_init = dict(embeddings)
        return self

    # ------------------------------------------------------------------- training
    def fit(self, triples: list[Triple], epochs: int | None = None) -> TorchComplEx:
        ents = sorted({x for h, _, t in triples for x in (h, t)})
        rels = sorted({r for _, r, _ in triples})
        self.ent = {e: i for i, e in enumerate(ents)}
        self.rel = {r: i for i, r in enumerate(rels)}
        ne, nr, d = len(ents), len(rels), self.cfg.dim
        if ne < 2 or nr < 1 or not triples:
            return self
        scale = 1.0 / d**0.5
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)

        def _new(rows: int) -> torch.Tensor:
            t = torch.empty(rows, d).normal_(0.0, scale, generator=gen)
            return t.to(self.device).requires_grad_(True)

        self._E_re = _new(ne)
        self._E_im = _new(ne)
        self._R_re = _new(nr)
        self._R_im = _new(nr)

        # Text-attributed init: project pre-computed text vectors to the
        # KGE dimension and seed ``_E_re`` for the entities we have a
        # description for.  We use a fixed-seed random projection
        # (Johnson-Lindenstrauss-style) so the seam is reproducible
        # without adding learnable text-encoder weights.
        if self._text_init:
            self._seed_from_text(ents, d, scale, gen)

        self._train(triples, epochs if epochs is not None else self.cfg.epochs)
        return self

    def _seed_from_text(self, ents: list[str], d: int, scale: float, gen) -> None:
        """Project per-entity text embeddings to KGE dim, write into ``_E_re``."""
        # Determine the source text-embedding dimension from the first
        # provided vector; require all provided vectors to match.
        first = next(iter(self._text_init.values()))
        text_dim = int(first.shape[-1])
        # Reproducible random projection (text_dim → d).  scale matches
        # the random init so the magnitudes stay comparable.
        proj_rng = np.random.default_rng(self.cfg.seed + 7919)
        proj = proj_rng.standard_normal((text_dim, d)).astype(np.float32) * scale
        proj_t = torch.from_numpy(proj).to(self.device)
        with torch.no_grad():
            for ent_name in ents:
                vec = self._text_init.get(ent_name)
                if vec is None:
                    continue
                if int(vec.shape[-1]) != text_dim:
                    raise ValueError(
                        f"text embedding for {ent_name!r} has dim "
                        f"{vec.shape[-1]} != expected {text_dim}"
                    )
                idx = self.ent[ent_name]
                t = torch.from_numpy(np.asarray(vec, dtype=np.float32)).to(self.device)
                self._E_re[idx] = t @ proj_t

    def partial_fit(self, triples: list[Triple], epochs: int = 25) -> TorchComplEx:
        if not self.ent:
            return self.fit(triples, epochs)
        d, scale = self.cfg.dim, 1.0 / self.cfg.dim**0.5

        def _extend(table: torch.Tensor, n_new: int) -> torch.Tensor:
            if n_new <= 0:
                return table
            with torch.no_grad():
                rows = torch.empty(n_new, d).normal_(0.0, scale).to(self.device)
            new = torch.cat([table.detach(), rows], dim=0)
            new.requires_grad_(True)
            return new

        new_ents = [e for f in triples for e in (f[0], f[2]) if e not in self.ent]
        new_ents = sorted(set(new_ents))
        for e in new_ents:
            self.ent[e] = len(self.ent)
        if new_ents:
            self._E_re = _extend(self._E_re, len(new_ents))
            self._E_im = _extend(self._E_im, len(new_ents))
        new_rels = sorted({r for _, r, _ in triples if r not in self.rel})
        for r in new_rels:
            self.rel[r] = len(self.rel)
        if new_rels:
            self._R_re = _extend(self._R_re, len(new_rels))
            self._R_im = _extend(self._R_im, len(new_rels))
        self._train(triples, epochs)
        return self

    def _train(self, triples: list[Triple], epochs: int) -> None:
        ne, nr = len(self.ent), len(self.rel)
        h = torch.tensor([self.ent[a] for a, _, _ in triples], device=self.device)
        r = torch.tensor([self.rel[b] for _, b, _ in triples], device=self.device)
        t = torch.tensor([self.ent[c] for _, _, c in triples], device=self.device)
        p, k, ku = len(triples), self.cfg.negatives, self.cfg.uniform_negatives
        if p == 0:
            return
        # relation-specific pools live on CPU; sampling done on CPU then moved.
        h_cpu = h.cpu().numpy()
        r_cpu = r.cpu().numpy()
        t_cpu = t.cpu().numpy()
        tail_pool = [
            np.array(sorted({int(t_cpu[i]) for i in range(p) if r_cpu[i] == ri}) or [0])
            for ri in range(nr)
        ]
        head_pool = [
            np.array(sorted({int(h_cpu[i]) for i in range(p) if r_cpu[i] == ri}) or [0])
            for ri in range(nr)
        ]
        rel_rows = [np.where(r_cpu == ri)[0] for ri in range(nr)]
        rng = np.random.default_rng(self.cfg.seed + 1)

        score_fn = self.cfg.score_fn
        # ComplEx-N3 and RotatE bring their own regulariser / margin loss, so
        # disable Adam weight_decay for them — combining both regularisers
        # double-penalises and hurts MRR (verified empirically on FB15k-237).
        weight_decay = self.cfg.reg if score_fn == "complex" else 0.0
        params = [self._E_re, self._E_im, self._R_re, self._R_im]
        opt = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=weight_decay)
        bce = torch.nn.BCEWithLogitsLoss()
        log_sigmoid = torch.nn.LogSigmoid()
        # ``self.cfg.margin`` is consumed by ``_phi_idx`` directly when
        # ``score_fn == "rotate"``; we only need ``alpha`` and ``n3_lambda``
        # at the loss-dispatch site below.
        alpha = float(self.cfg.alpha)
        n3_lambda = float(self.cfg.n3_lambda)

        batch = self.cfg.batch_size if self.cfg.batch_size > 0 else p
        idx = np.arange(p)
        total_neg = k + ku

        for epoch in range(1, epochs + 1):
            rng.shuffle(idx)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, p, batch):
                bidx = idx[start : start + batch]
                bh = h_cpu[bidx]
                br = r_cpu[bidx]
                bt = t_cpu[bidx]
                bp = len(bidx)

                # type-correct negatives (per relation)
                nh = np.repeat(bh[:, None], k, axis=1)
                nt = np.repeat(bt[:, None], k, axis=1)
                for ri, _rows in enumerate(rel_rows):
                    mask = br == ri
                    if not mask.any():
                        continue
                    rows = np.where(mask)[0]
                    tp, hp = tail_pool[ri], head_pool[ri]
                    nt[rows] = tp[rng.integers(0, len(tp), (len(rows), k))]
                    nh[rows] = hp[rng.integers(0, len(hp), (len(rows), k))]
                ch = rng.random((bp, k)) < 0.5
                nh = np.where(ch, nh, bh[:, None])
                nt = np.where(ch, bt[:, None], nt)

                # uniform negatives — needed for an open-world ranking signal
                if ku:
                    ut = rng.integers(0, ne, (bp, ku))
                    uh_rand = rng.integers(0, ne, (bp, ku))
                    cu = rng.random((bp, ku)) < 0.5
                    uh = np.where(cu, uh_rand, bh[:, None])
                    ut = np.where(cu, bt[:, None], ut)
                else:
                    uh = np.zeros((bp, 0), dtype=np.int64)
                    ut = np.zeros((bp, 0), dtype=np.int64)

                # ---- positives ---------------------------------------------------
                H_pos = torch.from_numpy(bh).to(self.device)
                R_pos = torch.from_numpy(br).to(self.device)
                T_pos = torch.from_numpy(bt).to(self.device)
                phi_pos = self._phi_idx(H_pos, R_pos, T_pos)  # [bp]

                # ---- negatives ---------------------------------------------------
                if total_neg > 0:
                    nh_all = np.concatenate([nh, uh], axis=1).reshape(-1)
                    nt_all = np.concatenate([nt, ut], axis=1).reshape(-1)
                    nr_all = np.repeat(br[:, None], total_neg, axis=1).reshape(-1)
                    H_neg = torch.from_numpy(nh_all).to(self.device)
                    R_neg = torch.from_numpy(nr_all).to(self.device)
                    T_neg = torch.from_numpy(nt_all).to(self.device)
                    phi_neg = self._phi_idx(H_neg, R_neg, T_neg).view(bp, total_neg)
                else:
                    phi_neg = phi_pos.new_zeros(bp, 0)

                # ---- loss dispatch ----------------------------------------------
                opt.zero_grad()
                if score_fn == "rotate":
                    # Margin-based + self-adversarial neg sampling (Sun et al. 2019).
                    # phi already encodes γ - d (positive: high = close; negative
                    # we want σ(d - γ) = σ(-phi)).  Self-adv weights softmax
                    # the negative *scores* (constant γ cancels under softmax).
                    pos_loss = -log_sigmoid(phi_pos).mean()
                    if total_neg > 0:
                        with torch.no_grad():
                            weights = torch.softmax(alpha * phi_neg, dim=-1)
                        neg_loss = -(weights * log_sigmoid(-phi_neg)).sum(-1).mean()
                        loss = pos_loss + neg_loss
                    else:
                        loss = pos_loss
                elif score_fn == "complex_n3":
                    # 1-vs-N softmax NLL (positive vs sampled negatives) + N3 reg.
                    if total_neg > 0:
                        all_phi = torch.cat([phi_pos.unsqueeze(-1), phi_neg], dim=-1)
                        targets = torch.zeros(bp, dtype=torch.long, device=self.device)
                        loss = torch.nn.functional.cross_entropy(all_phi, targets)
                    else:
                        loss = -log_sigmoid(phi_pos).mean()
                    # N3 nuclear-norm regulariser on the embeddings used by the
                    # positives this batch (Lacroix 2018 §3).
                    n3 = (
                        self._E_re[H_pos].abs().pow(3).sum()
                        + self._E_im[H_pos].abs().pow(3).sum()
                        + self._R_re[R_pos].abs().pow(3).sum()
                        + self._R_im[R_pos].abs().pow(3).sum()
                        + self._E_re[T_pos].abs().pow(3).sum()
                        + self._E_im[T_pos].abs().pow(3).sum()
                    ) / bp
                    loss = loss + n3_lambda * n3
                else:
                    # Default ComplEx BCE — matches the prior published behaviour.
                    if total_neg > 0:
                        phi_all = torch.cat([phi_pos, phi_neg.reshape(-1)], dim=0)
                        labels_t = torch.cat(
                            [
                                torch.ones(bp, device=self.device),
                                torch.zeros(bp * total_neg, device=self.device),
                            ]
                        )
                    else:
                        phi_all = phi_pos
                        labels_t = torch.ones(bp, device=self.device)
                    loss = bce(phi_all, labels_t)

                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            self.loss_history.append(epoch_loss / max(n_batches, 1))
            if self.cfg.log_every and epoch % self.cfg.log_every == 0:
                print(f"  epoch {epoch:4d}  loss={self.loss_history[-1]:.4f}")

    # ------------------------------------------------------------------- inference
    def predict_tail(
        self, head: str, relation: str, candidates: list[str], top_k: int = 5
    ) -> Prediction | None:
        if head not in self.ent or relation not in self.rel:
            return None
        usable = [c for c in candidates if c in self.ent and c != head]
        if not usable:
            return None
        with torch.no_grad():
            h = torch.full((len(usable),), self.ent[head], device=self.device)
            r = torch.full((len(usable),), self.rel[relation], device=self.device)
            t = torch.tensor([self.ent[c] for c in usable], device=self.device)
            scores = self._phi_idx(h, r, t).cpu().numpy()
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

    # ------------------------------------------------------------------- calibration
    def calibrate(self, val: list[tuple[str, str, list[str], str]]) -> float:
        rows = []
        for head, relation, cands, truth in val:
            if head not in self.ent or relation not in self.rel:
                continue
            usable = [c for c in cands if c in self.ent]
            if truth not in usable:
                continue
            with torch.no_grad():
                h = torch.full((len(usable),), self.ent[head], device=self.device)
                r = torch.full((len(usable),), self.rel[relation], device=self.device)
                t = torch.tensor([self.ent[c] for c in usable], device=self.device)
                scores = self._phi_idx(h, r, t).cpu().numpy()
            rows.append((scores, usable.index(truth)))
        if not rows:
            return self.temperature
        best_t, best_nll = 1.0, float("inf")
        for temp in np.linspace(0.1, 6.0, 60):
            nll = 0.0
            for scores, idx in rows:
                z = scores / temp
                z = z - z.max()
                pp = np.exp(z) / np.exp(z).sum()
                nll -= np.log(max(pp[idx], 1e-12))
            if nll < best_nll:
                best_nll, best_t = nll, float(temp)
        self.temperature = best_t
        return best_t

    # ------------------------------------------------------------------- evaluation
    def evaluate(
        self, test: list[Triple], filter_triples: set[Triple] | None = None
    ) -> dict[str, float]:
        filter_triples = filter_triples or set()
        ne = len(self.ent)
        # Pre-index filter triples for fast lookup
        filter_map: dict[tuple[str, str], set[str]] = {}
        for fh, fr, ft in filter_triples:
            if (fh, fr) not in filter_map:
                filter_map[(fh, fr)] = set()
            filter_map[(fh, fr)].add(ft)
        all_ent_idx = torch.arange(ne, device=self.device).unsqueeze(0)
        ranks: list[int] = []

        valid_test = [
            (h, r, t) for h, r, t in test if h in self.ent and r in self.rel and t in self.ent
        ]
        if not valid_test:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

        # ⚡ Bolt Optimization: Batch test triples and score all entities simultaneously
        # via broadcasting in `_phi_idx`. This significantly speeds up row-by-row
        # looping while guaranteeing bit-identical ranks by reusing the exact formula.
        batch_size = 128
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

                scores = self._phi_idx(b_h, b_r, all_ent_idx).cpu().numpy()
                b_t_np = b_t.cpu().numpy()

                for i, (h, r, t) in enumerate(batch):
                    for ft in filter_map.get((h, r), set()):
                        if ft != t and ft in self.ent:
                            scores[i, self.ent[ft]] = -np.inf

                    ti = b_t_np[i]
                    rank = int(np.sum(scores[i] > scores[i, ti]) + 1)
                    ranks.append(rank)

        arr = np.array(ranks)
        return {
            "MRR": float((1.0 / arr).mean()),
            "Hits@1": float((arr <= 1).mean()),
            "Hits@3": float((arr <= 3).mean()),
            "Hits@10": float((arr <= 10).mean()),
        }


def build_kge_from_settings(settings):  # noqa: ANN001
    """Return either ``TorchComplEx`` or NumPy ``ComplEx`` based on settings.

    ``settings.kge_model`` selects the scorer/loss variant when the Torch
    backend is active: ``complex`` (BCE, default), ``complex_n3`` (Lacroix
    2018 N3 regulariser + 1-vs-N NLL), or ``rotate`` (Sun 2019 margin loss
    + self-adversarial negatives).
    """
    from tacet.kge.kge import ComplEx, KGEConfig

    backend = settings.kge_backend
    if backend == "torch" or (backend == "auto" and _HAS_TORCH):
        model_name = getattr(settings, "kge_model", "complex") or "complex"
        return TorchComplEx(
            TorchKGEConfig(
                dim=settings.kge_dim,
                epochs=settings.kge_epochs,
                device=settings.kge_device,
                seed=0,
                score_fn=model_name,
            )
        )
    return ComplEx(KGEConfig(dim=settings.kge_dim, epochs=settings.kge_epochs))


__all__ = [
    "Prediction",
    "TorchComplEx",
    "TorchKGEConfig",
    "build_kge_from_settings",
]
