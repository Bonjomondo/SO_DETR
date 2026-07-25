#!/usr/bin/env python3
"""Reproducible SO-DETR V1.1 training entry point for VisDrone.

V1.1 adds scale-adaptive Expanded-IoU supervision while keeping the model
architecture and inference graph unchanged. The recommended setting adapts
only encoder query-quality targets and leaves the original regression loss
untouched for a controlled ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parent
MODEL_CONFIGS: Dict[str, Path] = {
    "r18": ROOT / "ultralytics/cfg/models/A-Test-M-R18.yaml",
    "r50": ROOT / "ultralytics/cfg/models/A-Test-r50-M.yaml",
    "ev2": ROOT / "ultralytics/cfg/models/A-Test-M-EV2.yaml",
}
EXPANDED_IOU_MODES = (
    "fixed",
    "adaptive-quality",
    "adaptive-regression",
    "adaptive-both",
)


def existing_file(value: str) -> Path:
    """Resolve a command-line path and fail early when the file is missing."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SO-DETR V1.1 with scale-adaptive Expanded-IoU on VisDrone."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="r18",
        help="Published SO-DETR model variant. V1.1 development model is r18.",
    )
    parser.add_argument(
        "--model-cfg",
        type=existing_file,
        default=None,
        help="Optional custom model YAML. Overrides --model when supplied.",
    )
    parser.add_argument(
        "--data",
        type=existing_file,
        required=True,
        help="VisDrone dataset YAML in Ultralytics/YOLO format.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument(
        "--expanded-iou-mode",
        choices=EXPANDED_IOU_MODES,
        default="adaptive-quality",
        help=(
            "Where to use the scale-adaptive ratio. adaptive-quality is the "
            "recommended V1.1 experiment; fixed exactly reproduces V1.0 loss behavior."
        ),
    )
    parser.add_argument("--expanded-iou-fixed-ratio", type=float, default=1.25)
    parser.add_argument("--expanded-iou-alpha", type=float, default=0.5)
    parser.add_argument("--expanded-iou-tau", type=float, default=0.01)
    parser.add_argument("--expanded-iou-min-ratio", type=float, default=1.0)
    parser.add_argument("--expanded-iou-max-ratio", type=float, default=1.5)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument(
        "--name",
        default=None,
        help="Run name. Defaults to sodetr-v1.1-<model>-<mode>-seed<seed>.",
    )
    parser.add_argument("--cache", action="store_true", help="Cache the dataset.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP. V1.1 default is disabled.")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic training. Deterministic mode is enabled by default.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize the backbone from available pretrained weights.",
    )
    parser.add_argument(
        "--resume",
        type=existing_file,
        default=None,
        metavar="CHECKPOINT",
        help="Resume from a last.pt checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import RTDETR

    if args.expanded_iou_fixed_ratio <= 0:
        raise ValueError("--expanded-iou-fixed-ratio must be positive.")
    if args.expanded_iou_alpha < 0:
        raise ValueError("--expanded-iou-alpha must be non-negative.")
    if args.expanded_iou_tau <= 0:
        raise ValueError("--expanded-iou-tau must be positive.")
    if args.expanded_iou_min_ratio <= 0:
        raise ValueError("--expanded-iou-min-ratio must be positive.")
    if args.expanded_iou_max_ratio < args.expanded_iou_min_ratio:
        raise ValueError("--expanded-iou-max-ratio must be >= --expanded-iou-min-ratio.")

    model_cfg = args.model_cfg or MODEL_CONFIGS[args.model]
    if not model_cfg.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_cfg}")

    run_name = args.name or f"sodetr-v1.1-{args.model}-{args.expanded_iou_mode}-seed{args.seed}"

    # For a true resume, construct the model from the checkpoint and let the
    # trainer restore its saved optimizer/scheduler state where supported.
    model_source = args.resume if args.resume is not None else model_cfg
    model = RTDETR(str(model_source))

    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "deterministic": not args.non_deterministic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "mixup": args.mixup,
        "close_mosaic": args.close_mosaic,
        "expanded_iou_mode": args.expanded_iou_mode,
        "expanded_iou_fixed_ratio": args.expanded_iou_fixed_ratio,
        "expanded_iou_alpha": args.expanded_iou_alpha,
        "expanded_iou_tau": args.expanded_iou_tau,
        "expanded_iou_min_ratio": args.expanded_iou_min_ratio,
        "expanded_iou_max_ratio": args.expanded_iou_max_ratio,
        "cache": args.cache,
        "amp": args.amp,
        "pretrained": not args.no_pretrained,
        "project": args.project,
        "name": run_name,
        "exist_ok": args.exist_ok,
        "save": True,
        "save_period": -1,
        "plots": True,
        "verbose": True,
    }
    if args.resume is not None:
        train_args["resume"] = True

    print("SO-DETR V1.1 scale-adaptive Expanded-IoU")
    print(f"  model: {model_source}")
    print(f"  data:  {args.data}")
    print(f"  mode:  {args.expanded_iou_mode}")
    print(
        "  ratio: "
        f"1 + {args.expanded_iou_alpha} * exp(-area / {args.expanded_iou_tau}), "
        f"clamped to [{args.expanded_iou_min_ratio}, {args.expanded_iou_max_ratio}]"
    )
    print(f"  run:   {Path(args.project) / run_name}")
    model.train(**train_args)


if __name__ == "__main__":
    main()
