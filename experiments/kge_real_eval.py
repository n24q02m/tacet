"""Train and evaluate the Tier-2 ComplEx KGE on a standard link-prediction benchmark.

Reports filtered MRR / Hits@1 / Hits@3 / Hits@10 on the test split — the
standard knowledge-graph-completion protocol (Bordes 2013).

Run on the shipped curated world-geography KG::

    python experiments/kge_real_eval.py

Or on a downloaded standard benchmark (FB15k-237, WN18RR, …) laid out as
``train.txt``, ``valid.txt``, ``test.txt`` in ``<root>``::

    python experiments/kge_real_eval.py --root path/to/FB15k-237 --epochs 100

The script writes a JSON report next to its `--out` argument.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from tacet.data import KGDataset, load_kg_dataset, synthetic_kg_dataset, worldgeo_dataset
from tacet.kge.kge import ComplEx, KGEConfig


def evaluate(
    ds: KGDataset,
    cfg: KGEConfig,
    *,
    backend: str = "numpy",
    device: str = "cpu",
    model_name: str = "complex",
    n3_lambda: float = 1e-2,
    margin: float = 12.0,
    alpha: float = 1.0,
) -> dict:
    """Train a KGE on ``ds.train`` and return filtered link-prediction metrics.

    ``model_name`` only affects the Torch backend (NumPy backend uses the
    classic ComplEx + BCE recipe).  Supported: ``complex`` (default),
    ``complex_n3`` (Lacroix 2018: NLL + N3 regulariser, recommended for
    standard KGC benchmarks), ``rotate`` (Sun 2019: margin + self-adv).
    """
    t0 = time.time()
    if backend == "torch":
        from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig

        model: object = TorchComplEx(
            TorchKGEConfig(
                dim=cfg.dim,
                epochs=cfg.epochs,
                negatives=cfg.negatives,
                batch_size=cfg.batch_size if cfg.batch_size > 0 else 1024,
                lr=cfg.lr,
                seed=cfg.seed,
                device=device,
                score_fn=model_name,
                n3_lambda=n3_lambda,
                margin=margin,
                alpha=alpha,
            )
        ).fit(ds.train)
    else:
        if model_name != "complex":
            raise SystemExit(
                f"--model={model_name!r} requires --backend=torch "
                "(NumPy backend only supports the BCE ComplEx recipe)."
            )
        model = ComplEx(cfg).fit(ds.train)
    train_time = time.time() - t0
    all_known = set(ds.all_triples())
    t0 = time.time()
    metrics = model.evaluate(ds.test, filter_triples=all_known)
    eval_time = time.time() - t0
    return {
        "dataset": ds.name,
        "stats": ds.stats(),
        "kge": {
            "backend": backend,
            "device": device,
            "model": model_name,
            "dim": cfg.dim,
            "epochs": cfg.epochs,
            "negatives": cfg.negatives,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "n3_lambda": n3_lambda,
            "margin": margin,
            "alpha": alpha,
        },
        "train_time_s": round(train_time, 2),
        "eval_time_s": round(eval_time, 2),
        "final_loss": round(model.loss_history[-1], 4) if model.loss_history else None,
        **{k: round(v, 4) for k, v in metrics.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=None,
        help="dataset root with train.txt/valid.txt/test.txt (overrides --dataset)",
    )
    ap.add_argument(
        "--dataset",
        choices=("synthetic", "worldgeo"),
        default="synthetic",
        help="builtin dataset (synthetic: ≈1400 triples; worldgeo: ≈100, sanity only)",
    )
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--negatives", type=int, default=8)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = full-batch; recommend 2048+ at FB15k scale (>10k entities)",
    )
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    ap.add_argument("--device", default="cpu", help="(torch only) cpu | cuda | mps | auto")
    ap.add_argument(
        "--model",
        choices=("complex", "complex_n3", "rotate"),
        default="complex",
        help="(torch only) scorer + loss variant — "
        "complex (BCE, default), complex_n3 (Lacroix), "
        "rotate (Sun, margin + self-adv)",
    )
    ap.add_argument(
        "--n3-lambda", type=float, default=1e-2, help="(complex_n3) N3 regulariser weight λ"
    )
    ap.add_argument(
        "--margin", type=float, default=12.0, help="(rotate) margin γ (Sun 2019: 12-24 typical)"
    )
    ap.add_argument(
        "--alpha", type=float, default=1.0, help="(rotate) self-adversarial temperature α"
    )
    ap.add_argument("--out", default="experiments/results/kge_real.json")
    args = ap.parse_args()

    if args.root is not None:
        ds = load_kg_dataset(args.root)
    elif args.dataset == "worldgeo":
        ds = worldgeo_dataset(seed=args.seed)
    else:
        ds = synthetic_kg_dataset(seed=args.seed)
    cfg = KGEConfig(
        dim=args.dim,
        epochs=args.epochs,
        negatives=args.negatives,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    report = evaluate(
        ds,
        cfg,
        backend=args.backend,
        device=args.device,
        model_name=args.model,
        n3_lambda=args.n3_lambda,
        margin=args.margin,
        alpha=args.alpha,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
