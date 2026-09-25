<h2 align="center">SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection</h2>

This repository is a research fork of the official SO-DETR implementation. The `SO-DETR-V1.1` branch contains the clean V1.0 baseline, the historical V1.1 scale-adaptive Expanded-IoU experiments, and the V1.2/Q1P one-sided follow-up.

## V1.1 scale-adaptive Expanded-IoU

V1.1 replaces the fixed `ratio=1.25` query-quality target with a per-target ratio:

```text
ratio = clamp(1 + alpha * exp(-normalized_area / tau), min_ratio, max_ratio)
```

The V1.1 defaults are `alpha=0.5`, `tau=0.01`, `min_ratio=1.0`, and `max_ratio=1.5`. Small targets receive more tolerant query-quality supervision, while sufficiently large targets can receive a ratio below the published `1.25` baseline. The historical V1.1 Q1 experiment applies adaptation only to the query-quality target; the published SO-DETR regression loss remains fixed at `ratio=1.25`.

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

Run the historical V1.1 R18 Q1 experiment:

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

## V1.2 / Q1P one-sided adaptive Expanded-IoU

The seed-0 V1.1 result shows a small `APs` increase for Q1 together with lower `APm` and nearly unchanged total AP. Because the V1.1 curve can fall below `1.25`, Q1 changes supervision for both small and larger targets.

V1.2/Q1P isolates the small-target side of that hypothesis by keeping the adaptive quality ratio at or above the published baseline:

```text
ratio = clamp(
    1 + alpha * exp(-normalized_area / tau),
    fixed_ratio,
    max_ratio,
)
```

With the default settings, Q1P uses `[1.25, 1.5]` for query-quality supervision and keeps regression fixed at `1.25`. Historical Q1 remains `[1.0, 1.5]`; old results are not reinterpreted or overwritten.

Run Q1P seed0 with:

```bash
python run_sodetr_v1_2_one_sided.py \
  --model r18 \
  --data configs/visdrone-local.yaml \
  --device 0,1 \
  --seed 0
```

The launcher records a separate run name such as:

```text
sodetr-v1.2-r18-q1p-one-sided-quality-seed0
```

See [V1.2_ONE_SIDED_ADAPTIVE_IOU.md](V1.2_ONE_SIDED_ADAPTIVE_IOU.md) for the motivation, exact crossover analysis, direct command, and recommended multi-seed sequence.

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

These numbers are retained as source-reported reference values. V1.0, V1.1, and V1.2 experiments in this fork should be recorded separately under the standardized training protocol.

## Results in this fork

SO-DETR-R18 (20.59 M parameters, 65.32 GFLOPs), VisDrone-2019-DET `val`, seed 0, formal
COCOeval on `weights/best.pt`. Values are AP in percent.

| Run | `expanded_iou_mode` | Training | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Q0 | `fixed` | 336 epochs (best 284) | 29.13 | 47.86 | 29.78 | 20.58 | 39.99 | 42.23 |
| Q1 | `adaptive-quality` | 400 epochs (best 382) | 28.90 | 48.16 | 29.36 | 21.05 | 39.06 | 45.73 |
| Q2 | `adaptive-regression` | 388 epochs, early stop (best 348) | 28.65 | 47.71 | 28.86 | 20.55 | 38.79 | 41.17 |
| Q3 | `adaptive-both` | in progress | — | — | — | — | — | — |
| Q1P | `adaptive-quality`, min ratio = 1.25 | not run yet | — | — | — | — | — | — |

These are single seed-0 runs where available. Delta-to-baseline is intentionally not reported here: Q0
predates structured telemetry (training status and stop reason unknown), and the runs are
not yet verified as configuration- and protocol-identical. Treat this table as measurements,
not as evidence that one mode is better. Run seeds 1 and 2 before drawing conclusions.

Machine-generated tables, per-class AP, plots and provenance links live in
[`reports/experiments.md`](reports/experiments.md), which is regenerated by
`sodetr_reports.py` and must not be edited by hand.

## 自动实验报告

训练结束后默认自动 COCOeval 并更新 `reports/experiments.md`、`experiments.xlsx`、JSON/CSV 与三张对比图。支持实验说明、每类别 AP、兼容基线的百分点比较和评估失败重试。

已有实验可运行 `python sodetr_reports.py --backfill-per-class` 补齐每类别 AP 并生成报告。完整用法和指标口径见 [EXPERIMENT_REPORTS.md](EXPERIMENT_REPORTS.md)。
