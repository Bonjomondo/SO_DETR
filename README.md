<h2 align="center">SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection</h2>

This repository is a research fork of the official SO-DETR implementation. The `SO-DETR-V1.1` branch adds scale-adaptive Expanded-IoU query-quality supervision on top of the clean, reproducible V1.0 baseline.

## V1.1 scale-adaptive Expanded-IoU

V1.1 replaces the fixed `ratio=1.25` query-quality target with a per-target ratio:

```text
ratio = clamp(1 + alpha * exp(-normalized_area / tau), min_ratio, max_ratio)
```

The recommended defaults are `alpha=0.5`, `tau=0.01`, `min_ratio=1.0`, and `max_ratio=1.5`. Small targets receive more tolerant query-quality supervision, while large targets approach ordinary IoU geometry. The default V1.1 experiment applies adaptation only to the query-quality target; the published SO-DETR regression loss remains fixed at `ratio=1.25`.

This change affects training only. It adds no parameters, FLOPs, or inference latency.

### Quick start

Install PyTorch/torchvision for the CUDA version on the server, then install the remaining dependencies:

```bash
pip install -r requirements_v1.txt
```

Create a local dataset configuration:

```bash
cp configs/visdrone-local.example.yaml configs/visdrone-local.yaml
# Edit the `path` field in configs/visdrone-local.yaml.
```

Run the recommended V1.1 R18 experiment:

```bash
python train_sodetr_visdrone.py \
  --model r18 \
  --data configs/visdrone-local.yaml \
  --device 0,1 \
  --epochs 400 \
  --patience 40 \
  --batch 8 \
  --imgsz 640 \
  --workers 8 \
  --seed 0 \
  --close-mosaic 0 \
  --mixup 0.2 \
  --lrf 1.0 \
  --expanded-iou-mode adaptive-quality
```

The same protocol is encoded as the V1.1 trainer defaults, so this is equivalent:

```bash
python train_sodetr_visdrone.py \
  --model r18 \
  --data configs/visdrone-local.yaml
```

Run Q0-Q3 sequentially:

```bash
python run_sodetr_v1_1_ablation.py \
  --only all \
  --model r18 \
  --data configs/visdrone-local.yaml \
  --device 0,1
```

See [V1.1_SCALE_ADAPTIVE_IOU.md](V1.1_SCALE_ADAPTIVE_IOU.md) for the design boundary, ablation mapping, commands, and acceptance criteria. The unchanged baseline is documented in [V1.0_BASELINE.md](V1.0_BASELINE.md).

## Published model configurations

| Variant | Model YAML | Intended role in this fork |
|---|---|---|
| SO-DETR-R18 | `ultralytics/cfg/models/A-Test-M-R18.yaml` | Main development baseline |
| SO-DETR-R50 | `ultralytics/cfg/models/A-Test-r50-M.yaml` | Teacher and large-model reference |
| SO-DETR-EV2 | `ultralytics/cfg/models/A-Test-M-EV2.yaml` | Lightweight deployment reference |

## Original project update

- **2025-01-17:** released SO-DETR-R50, SO-DETR-R18, SO-DETR-EV2, and the original response-distillation code.

## Results reported by the original authors

### VisDrone-2019-DET

| Model | Backbone | Input Size | Params (M) | GFLOPs | AP | AP50 |
|---|---|---:|---:|---:|---:|---:|
| SO-DETR | EfficientFormerV2 | 640×640 | 12.1 | 33.3 | 28.2 | 46.7 |
| SO-DETR (Distilled) | EfficientFormerV2 | 640×640 | 12.1 | 33.3 | 28.8 | 47.5 |
| SO-DETR | ResNet18 | 640×640 | 20.5 | 64.3 | 29.9 | 49.0 |
| SO-DETR | ResNet50 | 640×640 | 44.4 | 161.4 | 31.5 | 51.5 |

### UAVVaste

| Model | Params (M) | GFLOPs | AP | AP50 |
|---|---:|---:|---:|---:|
| SO-DETR-R50 | 44.4 | 161.4 | 37.5 | 76.4 |
| SO-DETR-R18 | 20.5 | 64.3 | 35.1 | 72.1 |
| SO-DETR-EV2 | 12.1 | 33.3 | 33.7 | 70.6 |
| SO-DETR-EV2 (Distilled) | 12.1 | 33.3 | 36.9 | 73.6 |

These numbers are retained as source-reported reference values. V1.0 and V1.1 experiments in this fork should be recorded separately under the standardized training protocol.
