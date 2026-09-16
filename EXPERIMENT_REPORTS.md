# 自动评估与实验报告

从 `train_sodetr_visdrone.py`、兼容入口 `train_exp.py` 或 Q0–Q3 批量入口启动训练，默认在正常结束或早停后评估 `weights/best.pt`，随后更新 `reports/`。训练失败、手动中断和评估失败也会尽可能刷新状态报告，并保留原始异常退出。

```bash
conda activate UAV_CV
pip install -r requirements_v1.txt

python train_sodetr_visdrone.py \
  --data datasets/VisDrone2019-DET-YOLO/data.yaml \
  --coco-anno "$(pwd)/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json" \
  --device 0 --name Q1 --expanded-iou-mode adaptive-quality \
  --experiment-description 'Q1：仅质量监督使用尺度自适应 Expanded-IoU；其余训练配置与 Q0 一致'
```

`--coco-anno` 相对路径按**数据集 YAML 所在目录**解析，因此上述参数建议使用绝对路径（例如 `"$(pwd)/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json"`）；也可在数据集 YAML 中添加 `coco_anno: annotations/VisDrone2019-DET_val_coco.json`，然后省略命令行参数。

批量入口 `run_sodetr_v1_1_ablation.py` 同样支持 `--coco-anno`、`--reports-dir`、`--baseline`、`--no-reports`、`--no-formal-coco-eval`。每个实验结束后更新一次报告；沿用原有“首个失败即停止”的行为。使用脚本的 `--device 0,1` 启动多卡训练时，由启动进程在训练子进程退出后统一评估、汇总；遥测由 rank 0 写入。

## 已有实验、补算与重试

仅重新汇总，无需 GPU，不加载模型：

```bash
python sodetr_reports.py
python sodetr_reports.py --runs-dir runs/train --output reports --baseline Q0
```

为旧实验补算每类别 AP：读取已经归一化的 `predictions_coco.json` 和指标文件中记录的 GT，在 CPU 上执行 COCOeval，不重新推理。只补缺少 `per_class` 的实验：

```bash
python sodetr_reports.py --backfill-per-class
```

从可信的本地 `best.pt` 恢复模型路径、参数量和最佳轮次，需要训练环境；常规汇总不会反序列化 checkpoint。未记录的训练耗时、停止原因和历史显存仍留空：

```bash
python sodetr_reports.py --backfill-checkpoint-info
```

评估失败后单独重跑，使用该实验 `args.yaml` 的配置及 `best.pt`，可指定设备和 GT；失败状态会进入报告，命令以非零状态退出：

```bash
python sodetr_reports.py --evaluate Q1 --device 0 \
  --coco-anno "$(pwd)/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json"
```

## PyTorch 2.6+ 的 `weights_only` 兼容

`torch.load` 从 PyTorch 2.6 起默认 `weights_only=True`，会拒绝反序列化 checkpoint 中保存的自定义类（例如 `ultralytics.nn.tasks.RTDETRDetectionModel`），报错形式为 `_pickle.UnpicklingError: Weights only load failed`。本仓库对可信的本地 checkpoint 显式传入 `weights_only=False`：

- `ultralytics/nn/tasks.py::torch_safe_load`：训练、验证、推理的加载入口；
- `ultralytics/utils/torch_utils.py::strip_optimizer`：`final_eval()` 的收尾步骤；
- `sodetr_reports.py::backfill_checkpoint_info`：显式补算。

注意 Ultralytics 保存的是模型对象而非纯 `state_dict`，因此即使已经过 `strip_optimizer`，`weights_only=True` 仍然会失败，这是预期行为。

若训练已正常结束、只是收尾阶段抛出 `UnpicklingError`，`training_state.json` 会被记为 `failed`，而报告把该状态视为过期、不纳入比较。此时权重通常完好，可先重跑评估，再把状态改回 `completed`（保留原始异常文本备查）：

```bash
python sodetr_reports.py --evaluate Q2 --device 0 \
  --coco-anno "$(pwd)/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json"
```

## 输出与实验说明

每个实验目录新增 `training_state.json`（状态、实际轮数、最佳轮次、训练会话累计耗时、参数量、估算 GFLOPs、rank 0 峰值 CUDA allocated 显存）、`evaluation_state.json` 和 `experiment.json`。正式指标仍在 `formal_coco/best/coco_metrics.json`，增加每类别 AP/AP50/AP75、评估时间、GT SHA256、推理配置和 checkpoint 修改时间。无有效 GT 的指标使用 `null`。

可以直接编辑某个实验的 `experiment.json` 后重新汇总；后续不传 `--experiment-description` 时保留已有说明：

```json
{
  "experiment": "Q1",
  "description": "仅质量监督自适应；本次用于验证小目标 AP 的改善"
}
```

`reports/experiments.json` 是本次汇总的统一快照，包含原始指标、配置、来源、缺失提示和训练曲线。其他格式均从该快照生成，不应直接修改生成文件：

- `experiments.md`：主结果、相对基线变化、实验说明、来源文件链接、三张图。
- `experiments.csv`：每个实验一行，UTF-8 BOM，可用 Excel 打开。
- `experiments.xlsx`：总览、COCO指标、PerClass AP、训练曲线、实验配置、文件索引、图表、口径说明；支持冻结表头、筛选、排序，绿色/红色标记 AP 提升/下降，黄色粗体标记可比较组内的最佳 AP。
- `plots/map_comparison.png`、`small_medium_large.png`、`learning_curves.png`：总体 AP、目标尺度 AP 和六个训练指标曲线。

报告目录通过文件锁串行更新；先完整生成到临时目录，再逐文件替换，避免导出失败破坏已有文件。扫描单个损坏实验不会阻断其他实验。报告目录默认在 Git 中忽略。

## 指标口径和限制

正式 COCOeval 为 bbox、`maxDets=[1,10,100]`，预测阶段保留最多 300 个框。完整 12 项指标及每类别 AP 均从该评估产生。JSON/CSV 中指标为 **0–1 原始值**，Markdown/Excel/图表中为**百分数**；`delta_*_pp` 为**百分点**，例如 29.1 → 30.9 是 +1.8 pp。

Precision/Recall 来自训练 `results.csv` 中最佳轮次的内部验证结果。新训练记录 checkpoint 选择时的轮次；旧实验未补 checkpoint 信息时，按 `0.1*mAP50 + 0.9*mAP50-95` 从 CSV 估算并明确标记。内部 P/R 不能当作 COCO 的 AR100。

自动比较要求固定模式的基线唯一，模型、数据集、训练配置和评估协议一致。`--baseline` 可消除多个基线的歧义，但不会绕过一致性检查。缺少模型身份、历史推理配置或 GT 指纹时，新旧实验可能无法自动配对，此时留空差值并给出提示；可以用 `--backfill-checkpoint-info` 恢复模型信息，用 `--evaluate` 以当前协议重新评估补全评估来源。不同组不进行总体名次排序，也不自动宣称某种方法有效。

历史未记录的耗时和停止原因不能从 CSV 可靠恢复，留空。耗时涵盖已记录的训练会话及内部验证，不含正式 COCOeval；历史恢复后未记录的更早会话耗时不包括在内。GFLOPs 是仓库模型工具的估计，无法估计时留空；显存为 rank 0 的 PyTorch allocated 峰值，并非所有显卡显存总和。强制杀进程/断电无法执行收尾，状态可能保留为 running，需要人工核实。

本版未执行新一轮完整 GPU 训练。自动衔接以模拟训练测试覆盖，COCOeval 与报告导出在现有 Q0 和 CPU 小型数据集上验证。
