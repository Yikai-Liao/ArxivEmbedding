#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
本地批量生成 Arxiv Embedding 并上传至 Hugging Face。

用法：
  ./batch_embed_local.sh [选项]

选项：
  --years <年份列表>      逗号分隔的年份列表（默认：2025）
  --repo-id <仓库ID>       Hugging Face 数据集仓库 ID（默认：lyk/ArxivEmbedding）
  --batch-size <大小>      每次入队的文本数量（默认：128，建议 GPU 可提高）
  --task-dir <目录>        任务文件输出目录（默认：local_matrix_tasks）
  --artifacts-dir <目录>   嵌入结果输出目录（默认：local_artifacts）
  --max-tasks <数量>       单次处理的最大任务数（默认：1000000000）
  --engine <名称>          embed 推理引擎（torch 或 optimum，默认：torch）
  --device <设备>          推理设备（默认：cuda）
  --skip-sync              已同步依赖时跳过 `uv sync`
  -h, --help               显示此帮助信息

环境变量：
  HF_TOKEN 必须已设置，用于访问 Hugging Face。

示例：
  HF_TOKEN=xxxx ./batch_embed_local.sh --years 2025 --batch-size 256
EOF
}

YEARS="2025"
REPO_ID="lyk/ArxivEmbedding"
BATCH_SIZE=128
TASK_DIR="temp/local_matrix_tasks"
ARTIFACTS_DIR="temp/local_artifacts"
MAX_TASKS=1000000000
ENGINE="torch"
DEVICE="cuda"
SKIP_SYNC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --years)
      YEARS="$2"; shift 2 ;;
    --repo-id)
      REPO_ID="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --task-dir)
      TASK_DIR="$2"; shift 2 ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"; shift 2 ;;
    --max-tasks)
      MAX_TASKS="$2"; shift 2 ;;
    --engine)
      ENGINE="$2"; shift 2 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --skip-sync)
      SKIP_SYNC=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[错误] 未检测到 HF_TOKEN 环境变量。" >&2
  exit 1
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

if [[ $SKIP_SYNC -eq 0 ]]; then
  echo "[步骤] 使用 uv 同步依赖..."
  if [[ -f uv.lock ]]; then
    uv sync --locked --all-extras --dev
  else
    uv sync --all-extras --dev
  fi
fi

MATRIX_ID=0
TASK_FILE="$TASK_DIR/matrix_${MATRIX_ID}_tasks.json"
OUTPUT_DIR="$ARTIFACTS_DIR/matrix-output-${MATRIX_ID}"

mkdir -p "$TASK_DIR"

echo "[步骤] 生成任务列表 (年份: $YEARS)..."
uv run python script/local_split_tasks.py \
  --repo-id "$REPO_ID" \
  --matrix-count 1 \
  --years "$YEARS" \
  --config-file config.toml \
  --output-dir "$TASK_DIR" \
  --max-tasks-per-matrix "$MAX_TASKS"

if [[ ! -s "$TASK_FILE" ]]; then
  echo "[信息] 未生成任务文件或任务为空：$TASK_FILE"
  echo "[结束] 没有需要处理的嵌入任务。"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

echo "[步骤] 处理任务，生成嵌入 (设备: $DEVICE, batch: $BATCH_SIZE)..."
uv run python script/process_matrix_tasks.py \
  --matrix-id "$MATRIX_ID" \
  --task-file "$TASK_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --config-file config.toml \
  --repo-id "$REPO_ID" \
  --batch-size "$BATCH_SIZE" \
  --engine "$ENGINE" \
  --device "$DEVICE"

echo "[步骤] 合并嵌入并上传到 Hugging Face..."
uv run python script/merge.py \
  --repo-id "$REPO_ID" \
  --artifacts-dir "$ARTIFACTS_DIR"

echo "[完成] 全部流程执行完毕。"
