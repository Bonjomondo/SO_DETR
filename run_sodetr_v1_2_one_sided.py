#!/usr/bin/env python3
"""Run the SO-DETR V1.2 one-sided Expanded-IoU quality experiment.

V1.2/Q1P preserves the published SO-DETR Expanded-IoU baseline ratio for
medium/large targets while only allowing extra tolerance for sufficiently
small targets. It reuses the V1.1 ratio generator and lower-bounds the
adaptive quality ratio by the fixed baseline ratio.

Historical V1.1 Q0/Q1/Q2/Q3 semantics are intentionally unchanged.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from sodetr_formal_coco import add_formal_coco_arguments
from sodetr_reports import add_report_arguments


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_sodetr_visdrone.py"


def existing_file(value: str) -> Path:
    """Resolve a command-line path and fail early when the file is missing."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SO-DETR V1.2/Q1P: one-sided scale-adaptive Expanded-IoU "
            "for query-quality supervision only."
        )
    )
    parser.add_argument("--model", choices=("r18", "r50", "ev2"), default="r18")
    parser.add_argument("--data", type=existing_file, required=True)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--expanded-iou-fixed-ratio", type=float, default=1.25)
    parser.add_argument("--expanded-iou-alpha", type=float, default=0.5)
    parser.add_argument("--expanded-iou-tau", type=float, default=0.01)
    parser.add_argument("--expanded-iou-max-ratio", type=float, default=1.5)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    add_formal_coco_arguments(parser)
    add_report_arguments(parser)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    """Build the explicit Q1P training command."""
    if args.expanded_iou_fixed_ratio <= 0:
        raise ValueError("--expanded-iou-fixed-ratio must be positive.")
    if args.expanded_iou_alpha < 0:
        raise ValueError("--expanded-iou-alpha must be non-negative.")
    if args.expanded_iou_tau <= 0:
        raise ValueError("--expanded-iou-tau must be positive.")
    if args.expanded_iou_max_ratio < args.expanded_iou_fixed_ratio:
        raise ValueError(
            "--expanded-iou-max-ratio must be >= --expanded-iou-fixed-ratio "
            "for one-sided adaptation."
        )

    run_name = f"sodetr-v1.2-{args.model}-q1p-one-sided-quality-seed{args.seed}"
    default_description = (
        "V1.2 Q1P: one-sided adaptive query-quality Expanded-IoU; "
        f"adaptive ratio is lower-bounded by the fixed baseline ratio "
        f"{args.expanded_iou_fixed_ratio:g}; regression remains fixed."
    )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        args.model,
        "--data",
        str(args.data),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--optimizer",
        args.optimizer,
        "--lr0",
        str(args.lr0),
        "--lrf",
        str(args.lrf),
        "--weight-decay",
        str(args.weight_decay),
        "--mixup",
        str(args.mixup),
        "--close-mosaic",
        str(args.close_mosaic),
        "--expanded-iou-mode",
        "adaptive-quality",
        "--expanded-iou-fixed-ratio",
        str(args.expanded_iou_fixed_ratio),
        "--expanded-iou-alpha",
        str(args.expanded_iou_alpha),
        "--expanded-iou-tau",
        str(args.expanded_iou_tau),
        # Key V1.2 change: never let the adaptive quality ratio fall below
        # the published SO-DETR fixed ratio.
        "--expanded-iou-min-ratio",
        str(args.expanded_iou_fixed_ratio),
        "--expanded-iou-max-ratio",
        str(args.expanded_iou_max_ratio),
        "--project",
        args.project,
        "--name",
        run_name,
        "--experiment-description",
        args.experiment_description or default_description,
    ]

    for enabled, flag in (
        (args.cache, "--cache"),
        (args.amp, "--amp"),
        (args.non_deterministic, "--non-deterministic"),
        (args.no_pretrained, "--no-pretrained"),
        (args.exist_ok, "--exist-ok"),
    ):
        if enabled:
            command.append(flag)

    command.extend(["--reports-dir", str(args.reports_dir)])
    for value, flag in (
        (args.coco_anno, "--coco-anno"),
        (args.baseline, "--baseline"),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if not args.formal_coco_eval:
        command.append("--no-formal-coco-eval")
    if not args.reports:
        command.append("--no-reports")
    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print("\n[Q1P] one-sided adaptive query-quality Expanded-IoU", flush=True)
    print(
        "ratio = clamp(1 + alpha * exp(-area / tau), "
        "fixed_ratio, max_ratio); regression = fixed_ratio",
        flush=True,
    )
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
