#!/usr/bin/env bash
set -e

# 进入脚本所在的项目根目录
cd "$(dirname "$0")"

# 默认扫描 runs/train，也可指定目录，例如: ./update_reports.sh runs/train
RUNS_DIR="${1:-runs/train}"
OUTPUT_DIR="${2:-reports}"

echo "🔄 正在扫描目录 [$RUNS_DIR] 并更新实验报告..."

python3 sodetr_reports.py --runs-dir "$RUNS_DIR" --output "$OUTPUT_DIR"

echo "✅ 更新完成！已重新生成 $OUTPUT_DIR/experiments.md、表格与图表。"

