#!/usr/bin/env python3
"""Run the SO-DETR V1.1 Expanded-IoU ablation experiments sequentially."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_sodetr_visdrone.py"
EXPERIMENTS = {
    "Q0": "fixed",
    "Q1": "adaptive-quality",
    "Q2": "adaptive-regression",
    "Q3": "adaptive-both",
}


def existing_file(value: str) -> Path:
    """Resolve a command-line path and fail early when the file is missing."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SO-DETR V1.1 Q0-Q3 sequentially and stop on the first failure."
    )
    parser.add_argument(
        "--only",
        choices=("all", *EXPERIMENTS),
        default="all",
        help="Run all experiments or one Q0-Q3 experiment.",
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
    parser.add_argument("--expanded-iou-min-ratio", type=float, default=1.0)
    parser.add_argument("--expanded-iou-max-ratio", type=float, default=1.5)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, experiment: str, mode: str) -> List[str]:
    """Build one explicit training command for reproducible logging."""
    run_name = f"sodetr-v1.1-{args.model}-{experiment.lower()}-{mode}-seed{args.seed}"
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
        mode,
        "--expanded-iou-fixed-ratio",
        str(args.expanded_iou_fixed_ratio),
        "--expanded-iou-alpha",
        str(args.expanded_iou_alpha),
        "--expanded-iou-tau",
        str(args.expanded_iou_tau),
        "--expanded-iou-min-ratio",
        str(args.expanded_iou_min_ratio),
        "--expanded-iou-max-ratio",
        str(args.expanded_iou_max_ratio),
        "--project",
        args.project,
        "--name",
        run_name,
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
    return command


def main() -> None:
    args = parse_args()
    selected = EXPERIMENTS.items() if args.only == "all" else [(args.only, EXPERIMENTS[args.only])]
    for experiment, mode in selected:
        command = build_command(args, experiment, mode)
        print(f"\n[{experiment}] expanded_iou_mode={mode}", flush=True)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
