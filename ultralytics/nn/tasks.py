# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Compatibility bootstrap for the SO-DETR Ultralytics task definitions.

The upstream SO-DETR snapshot mixes an Ultralytics 8.0.201 loss module with
newer YOLOv10 compatibility references in ``tasks.py``.  Those references
resolve ``E2EDetectLoss`` before a checkpoint is loaded, but that class is not
present in this repository, so final validation of ``best.pt`` fails.

Keep the original implementation byte-for-byte in ``_tasks_unpatched.py`` and
execute it in this module after removing only the incomplete YOLOv10 fragments.
Executing in this module namespace preserves checkpoint class paths such as
``ultralytics.nn.tasks.RTDETRDetectionModel``.
"""

from pathlib import Path

_IMPL_PATH = Path(__file__).with_name("_tasks_unpatched.py")
_SOURCE = _IMPL_PATH.read_text(encoding="utf-8")

_REPLACEMENTS = {
    '''    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
''': '''    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)
''',
    '                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10\n': "",
    '                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10\n': "",
}

for _old, _new in _REPLACEMENTS.items():
    if _old not in _SOURCE:
        raise ImportError(f"SO-DETR compatibility target was not found in {_IMPL_PATH}: {_old!r}")
    _SOURCE = _SOURCE.replace(_old, _new, 1)

exec(compile(_SOURCE, str(_IMPL_PATH), "exec"), globals(), globals())
