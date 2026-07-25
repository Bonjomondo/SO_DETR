#!/usr/bin/env python3
"""Backward-compatible entry point for the SO-DETR V1.0 trainer.

Use ``python train_exp.py --help`` or invoke ``train_sodetr_visdrone.py``
directly. The old machine-specific absolute paths have been removed.
"""

from train_sodetr_visdrone import main


if __name__ == "__main__":
    main()
