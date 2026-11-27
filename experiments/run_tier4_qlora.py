"""Thin CLI launcher for TACET v2 Tier 4 Wave B QLoRA experiments (Arms A/B/C)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import structlog  # noqa: E402

from tacet.distill.export import Arm  # noqa: E402
from tacet.distill.tier4_runner import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    QLoRAConfig,
    run_wave_b_matrix,
)

log = structlog.get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for Wave B QLoRA training and evaluation."""
    parser = argparse.ArgumentParser(description="Run TACET v2 Tier 4 Wave B QLoRA benchmark.")
    parser.add_argument("--arms", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/tier4_wave_b"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run configured Wave B arms/seeds and print aggregated summary."""
    args = parse_args(argv)
    arms: list[Arm] = list(args.arms)
    cfg = QLoRAConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
    )
    manifest = run_wave_b_matrix(
        arms=arms,
        seeds=args.seeds,
        output_dir=args.output_dir,
        cfg=cfg,
        dry_run=args.dry_run,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    log.info(
        "wave_b.finished",
        output_dir=str(args.output_dir),
        n_runs=len(manifest["runs"]),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
