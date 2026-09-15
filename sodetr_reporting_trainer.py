"""RT-DETR trainer with persistent experiment telemetry, including DDP rank zero."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import torch

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import de_parallel, get_flops
from sodetr_reports import read_json, write_json


class ReportingRTDETRTrainer(RTDETRTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_start", self._report_start)
        self.add_callback("on_fit_epoch_end", self._report_epoch)
        self.add_callback("on_train_end", self._report_end)

    def _save_state(self):
        if RANK in (-1, 0):
            self._report_state["training_seconds"] = self._previous_seconds + time.monotonic() - self._report_start_time
            self._report_state["updated_at"] = datetime.now(timezone.utc).isoformat()
            if self.device.type == "cuda":
                self._report_state["peak_gpu_memory_gib_rank0"] = torch.cuda.max_memory_allocated(self.device) / 2**30
            write_json(self.save_dir / "training_state.json", self._report_state)

    def _report_start(self, trainer):
        if RANK not in (-1, 0):
            return
        old = read_json(self.save_dir / "training_state.json") if self.resume else {}
        self._previous_seconds = old.get("training_seconds") or 0
        self._report_start_time = time.monotonic()
        model = de_parallel(self.model)
        # FLOPs profiling may execute dropout; preserve the training RNG state.
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type == "cuda" else []):
            gflops = get_flops(model, self.args.imgsz) or None
        self._report_state = {
            "status": "running", "stop_reason": None,
            "started_at": old.get("started_at", datetime.now(timezone.utc).isoformat()),
            "model": str(model.yaml.get("yaml_file", self.args.model)),
            "parameters": sum(p.numel() for p in model.parameters()),
            "gflops": gflops,
            "best_epoch": old.get("best_epoch"),
            "epochs_completed": self.start_epoch,
            "timing_scope": "recorded training sessions, includes validation; excludes formal COCOeval",
        }
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        write_json(self.save_dir / "evaluation_state.json", {"status": "pending"})
        # Stale metrics from an overwritten/resumed run are ignored by the collector.
        self._save_state()

    def _report_epoch(self, trainer):
        if RANK not in (-1, 0) or not hasattr(self, "_report_state"):
            return
        self._report_state["epochs_completed"] = self.epoch + 1
        # Same criterion used by save_model, including tied fitness.
        if self.fitness is not None and self.fitness == self.best_fitness:
            self._report_state["best_epoch"] = self.epoch + 1
        self._save_state()

    def _report_end(self, trainer):
        if RANK not in (-1, 0):
            return
        self._report_state.update(status="completed", stop_reason=(
            "epochs_completed" if self.epoch + 1 >= self.epochs else "early_stopping" if self.stop else "unknown"))
        self._save_state()

    def _do_train(self, world_size=1):
        if RANK in (-1, 0):
            initial = read_json(self.save_dir / "training_state.json") if self.resume else {}
            initial.update(status="running", stop_reason=None)
            write_json(self.save_dir / "training_state.json", initial)
            write_json(self.save_dir / "evaluation_state.json", {"status": "pending"})
        try:
            super()._do_train(world_size)
        except BaseException as exc:
            if RANK in (-1, 0) and hasattr(self, "_report_state"):
                self._report_state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                          stop_reason=type(exc).__name__, error=str(exc))
                self._save_state()
            elif RANK in (-1, 0):
                initial.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               stop_reason=type(exc).__name__, error=str(exc))
                write_json(self.save_dir / "training_state.json", initial)
            raise
