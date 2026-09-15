"""Automatic formal COCOeval for SO-DETR training entry points."""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


COCO_STAT_NAMES = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
)


def add_formal_coco_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--formal-coco-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After training, evaluate weights/best.pt with formal pycocotools "
            "COCOeval. Use --no-formal-coco-eval to disable it."
        ),
    )
    parser.add_argument(
        "--coco-anno",
        default=None,
        help=(
            "COCO-format validation GT JSON. When omitted, infer it from the "
            "dataset YAML or SODETR_COCO_ANNO."
        ),
    )


def _resolve_relative(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def resolve_coco_annotation(data_yaml: str | Path, explicit: str | Path | None) -> Path:
    data_yaml = Path(data_yaml).expanduser().resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_yaml}")

    candidates: list[Path] = []
    configured = explicit or os.environ.get("SODETR_COCO_ANNO")
    if configured:
        selected = _resolve_relative(configured, data_yaml.parent)
        if not selected.is_file():
            raise FileNotFoundError(f"Explicit COCO annotation not found: {selected}")
        return selected

    for key in ("coco_anno", "anno_json", "val_json"):
        if config.get(key):
            candidates.append(_resolve_relative(config[key], data_yaml.parent))

    dataset_root = config.get("path")
    if dataset_root:
        root = _resolve_relative(dataset_root, data_yaml.parent)
        candidates.extend(
            [
                root / "coco_annotations" / "val.json",
                root / "data_val.json",
                root / "annotations" / "data_val.json",
                root / "annotations" / "instances_val.json",
                root / "annotations" / "instances_val2017.json",
                root / "annotations" / "instances_val2019.json",
            ]
        )

    candidates.extend(
        [
            data_yaml.parent / "coco_annotations" / "val.json",
            data_yaml.parent / "data_val.json",
            data_yaml.parent / "annotations" / "data_val.json",
            data_yaml.parent / "annotations" / "instances_val.json",
        ]
    )

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in checked) or "(no candidates)"
    raise FileNotFoundError(
        "Formal COCOeval is enabled, but the validation COCO annotation JSON "
        "was not found. Pass --coco-anno, set SODETR_COCO_ANNO, or add "
        f"coco_anno to the dataset YAML.\nChecked:\n  - {searched}"
    )


def prepare_formal_coco_eval(
    data_yaml: str | Path,
    explicit_annotation: str | Path | None,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    try:
        from pycocotools.coco import COCO  # noqa: F401
        from pycocotools.cocoeval import COCOeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Automatic formal COCOeval is enabled, but pycocotools is unavailable. "
            "Install pycocotools or pass --no-formal-coco-eval."
        ) from exc
    return resolve_coco_annotation(data_yaml, explicit_annotation)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _dataset_names(data_yaml: Path) -> dict[int, str]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = config.get("names", {}) if isinstance(config, dict) else {}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def _normalize_predictions(
    raw_json: Path,
    annotation_json: Path,
    data_yaml: Path,
    output_json: Path,
) -> dict[str, Any]:
    ground_truth = _read_json(annotation_json)
    predictions = _read_json(raw_json)
    if not isinstance(ground_truth, dict) or not isinstance(predictions, list):
        raise ValueError("Invalid COCO ground-truth or prediction JSON structure.")

    images = ground_truth.get("images")
    categories = ground_truth.get("categories")
    if not isinstance(images, list) or not isinstance(categories, list):
        raise ValueError("COCO ground truth must contain images and categories lists.")

    gt_image_ids = {image["id"] for image in images}
    name_to_id = {Path(str(image["file_name"])).name: image["id"] for image in images}
    stem_to_id: dict[str, Any] = {}
    ambiguous_stems: set[str] = set()
    for image in images:
        stem = Path(str(image["file_name"])).stem
        previous = stem_to_id.get(stem)
        if previous is None:
            stem_to_id[stem] = image["id"]
        elif previous != image["id"]:
            stem_to_id.pop(stem, None)
            ambiguous_stems.add(stem)

    converted = 0
    missing: set[str] = set()
    for prediction in predictions:
        original = prediction.get("image_id")
        if original in gt_image_ids:
            continue

        if isinstance(original, str):
            try:
                numeric_id = int(original)
            except ValueError:
                numeric_id = None
            if numeric_id in gt_image_ids:
                prediction["image_id"] = numeric_id
                converted += 1
                continue

        basename = Path(str(original)).name
        stem = Path(basename).stem
        if stem in ambiguous_stems:
            raise ValueError(f"Ambiguous prediction image stem: {stem}")
        mapped = name_to_id.get(basename, stem_to_id.get(stem))
        if mapped is None:
            missing.add(str(original))
        else:
            prediction["image_id"] = mapped
            converted += 1

    if missing:
        raise ValueError(
            "Prediction image IDs do not correspond to the COCO set. "
            f"Examples: {sorted(missing)[:10]}"
        )

    gt_category_ids = {category["id"] for category in categories}
    prediction_category_ids = {prediction.get("category_id") for prediction in predictions}
    if None in prediction_category_ids:
        raise ValueError("A prediction is missing category_id.")

    category_mapping: dict[int, Any] = {}
    # This repository exports YOLO class indices, even when they happen to
    # overlap COCO category IDs (e.g. a batch with no class zero).
    if prediction_category_ids:
        data_names = _dataset_names(data_yaml)
        gt_name_to_id = {str(category["name"]): category["id"] for category in categories}
        for prediction_id in prediction_category_ids:
            if not isinstance(prediction_id, int) or prediction_id not in data_names:
                category_mapping.clear()
                break
            name = data_names[prediction_id]
            if name not in gt_name_to_id:
                category_mapping.clear()
                break
            category_mapping[prediction_id] = gt_name_to_id[name]
        if len(category_mapping) != len(prediction_category_ids):
            raise ValueError(
                "Prediction category IDs do not match the COCO ground truth, and "
                "a safe name-based mapping could not be inferred."
            )
        for prediction in predictions:
            prediction["category_id"] = category_mapping[prediction["category_id"]]

    mapped_image_ids = {prediction["image_id"] for prediction in predictions}
    if not mapped_image_ids.issubset(gt_image_ids):
        raise ValueError("Normalized image IDs are still incompatible with the COCO set.")

    _write_json(output_json, predictions)
    return {
        "prediction_entries": len(predictions),
        "prediction_images": len(mapped_image_ids),
        "ground_truth_images": len(gt_image_ids),
        "converted_image_entries": converted,
        "category_mapping": {str(key): value for key, value in category_mapping.items()},
    }


def _single_device(device: str) -> str:
    return str(device[0] if isinstance(device, (list, tuple)) else device).split(",", 1)[0].strip()


def run_formal_coco_eval(
    data_yaml: str | Path,
    annotation_json: str | Path,
    train_save_dir: str | Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict[str, float | None]:
    from ultralytics import RTDETR

    data_yaml = Path(data_yaml).resolve()
    annotation_json = Path(annotation_json).resolve()
    train_save_dir = Path(train_save_dir).resolve()
    best_weights = train_save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training best checkpoint not found: {best_weights}")

    eval_project = train_save_dir / "formal_coco"
    eval_name = "best"
    eval_dir = eval_project / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Never consume predictions left behind by an earlier validation.
    (eval_dir / "predictions.json").unlink(missing_ok=True)
    eval_model = RTDETR(str(best_weights))
    exported = {}
    eval_model.add_callback("on_val_end", lambda validator: exported.update(predictions=validator.jdict))
    eval_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=_single_device(device),
        save_json=True,
        plots=False,
        half=False,
        max_det=300,
        project=str(eval_project),
        name=eval_name,
        exist_ok=True,
    )

    raw_json = eval_dir / "predictions.json"
    if not raw_json.is_file() and exported.get("predictions") == []:
        _write_json(raw_json, [])
    if not raw_json.is_file():
        raise FileNotFoundError(f"Prediction JSON not found: {raw_json}")
    normalized_json = eval_dir / "predictions_coco.json"
    mapping = _normalize_predictions(raw_json, annotation_json, data_yaml, normalized_json)

    record = evaluate_predictions(annotation_json, normalized_json, {
        "weights": str(best_weights),
        "weights_mtime_ns": best_weights.stat().st_mtime_ns,
        "data": str(data_yaml),
        "raw_predictions": str(raw_json.resolve()),
        "mapping": mapping,
        "inference": {"imgsz": imgsz, "max_det": 300, "half": False, "split": "val"},
    })
    metrics_json = eval_dir / "coco_metrics.json"
    _write_json(metrics_json, record)
    print(f"[Formal COCOeval] saved: {metrics_json}", flush=True)
    return record["metrics"]


def evaluate_predictions(annotation_json: Path, predictions_json: Path,
                         metadata: dict | None = None) -> dict:
    """Evaluate saved, normalized predictions without loading a model or GPU."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(annotation_json))
    predictions = _read_json(predictions_json)
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        # pycocotools.loadRes([]) indexes the first item and raises IndexError.
        detections = COCO()
        detections.dataset = {"images": ground_truth.dataset["images"],
                              "categories": ground_truth.dataset["categories"], "annotations": []}
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    def valid_mean(values):
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else None

    metrics = {name: float(value) if value >= 0 else None
               for name, value in zip(COCO_STAT_NAMES, evaluator.stats)}
    precision = evaluator.eval["precision"]  # [IoU, recall, category, area, maxDets]
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        per_class[str(category_id)] = {
            "name": ground_truth.cats[category_id]["name"],
            "AP": valid_mean(values),
            "AP50": valid_mean(values[np.isclose(evaluator.params.iouThrs, .5)]),
            "AP75": valid_mean(values[np.isclose(evaluator.params.iouThrs, .75)]),
        }
    return {
        **(metadata or {}),
        "protocol": "pycocotools.COCOeval bbox",
        "maxDets": [1, 10, 100],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "annotations": str(annotation_json.resolve()),
        "annotation_sha256": hashlib.sha256(annotation_json.read_bytes()).hexdigest(),
        "normalized_predictions": str(predictions_json.resolve()),
        "metrics": metrics,
        "metrics_percent": {name: value * 100 if value is not None else None
                            for name, value in metrics.items()},
        "per_class": per_class,
    }


def enrich_saved_evaluation(metrics_json: Path) -> dict:
    """Backfill per-class AP from an existing prediction export on CPU."""
    record = _read_json(metrics_json)
    annotation = Path(record["annotations"])
    predictions = metrics_json.parent / "predictions_coco.json"
    if not predictions.is_file():
        predictions = Path(record["normalized_predictions"])
    enriched = evaluate_predictions(annotation, predictions, record)
    _write_json(metrics_json, enriched)
    return enriched
