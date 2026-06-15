#!/bin/bash

# Stop immediately on error to prevent cascading failures.
set -e

# ==========================================
# 1. Common settings and path definitions
# ==========================================
PROJECT_ROOT="$HOME/alpamayo-coc-autolabeler"
BASE_DATA_DIR="/mnt/share_drive/workspace-at/data/e2e-scene-shards-jpntaxi"
HF_CACHE_DIR="/mnt/nvme/hf_cache"
COC_CACHE_DIR="$HOME/coc_cache"

# Host-side parent directory where all outputs are organized by date.
BASE_OUTPUT_DIR="${PROJECT_ROOT}/batch_outputs"

DOCKER_IMAGE="coc_autolabeler:latest"
MODEL_NAME="qwen3.5_397b_fp8"

# ==========================================
# 2. Enumerate target date directories
# ==========================================
# Extract directory names matching "YYYY-MM-DD" format from BASE_DATA_DIR and sort them.
DATES=$(find "${BASE_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name "20*" -printf "%f\n" | sort)

echo "=========================================="
echo "Target dates for processing:"
echo "$DATES"
echo "=========================================="
echo ""


# ==========================================
# Step 1: Meta-Action generation for all dates
# ==========================================
echo ">>> Starting Step 1: Meta-Action Generation for all dates"
for DATE in $DATES; do
    echo "[Step 1] Processing: $DATE"
    TARGET_DATA_DIR="${BASE_DATA_DIR}/${DATE}"
    SAVE_DIR="${BASE_OUTPUT_DIR}/${DATE}/meta_actions"

    mkdir -p "${SAVE_DIR}"

    # Note: -it is omitted to support non-interactive batch processing.
    docker run --rm \
        -v ${PROJECT_ROOT}:/workspace \
        -v ${TARGET_DATA_DIR}:/dataset \
        -v ${COC_CACHE_DIR}:/coc_cache \
        -w /workspace/src \
        ${DOCKER_IMAGE} \
        python -m meta_action.data_labeling \
        --data_dir /dataset \
        --cache_dir /coc_cache \
        --save_dir /workspace/batch_outputs/${DATE}/meta_actions \
        --data_format webdataset \
        --meta_action_names all_ego
done
echo ">>> Step 1 completed!"
echo ""


# ==========================================
# Step 2: Keyframe selection for all dates
# ==========================================
echo ">>> Starting Step 2: Keyframe Selection for all dates"
for DATE in $DATES; do
    echo "[Step 2] Processing: $DATE"
    KEYFRAME_DIR="${BASE_OUTPUT_DIR}/${DATE}/keyframes"

    mkdir -p "${KEYFRAME_DIR}"

    docker run --rm \
        --user $(id -u):$(id -g) \
        -v ${PROJECT_ROOT}:/workspace \
        -w /workspace/src \
        ${DOCKER_IMAGE} \
        python -m coc_labeling.keyframe_auto_select \
        --meta_action_dir /workspace/batch_outputs/${DATE}/meta_actions/final_outputs \
        --output_dir /workspace/batch_outputs/${DATE}/keyframes
done
echo ">>> Step 2 completed!"
echo ""


# ==========================================
# Step 3: CoC Auto Labeling for all dates (model loaded once)
# ==========================================
echo ">>> Starting Step 3: CoC Auto Labeling (batch mode — model loaded once)"

docker run --rm \
    --user $(id -u):$(id -g) \
    --gpus all \
    --ipc=host \
    -e HF_TOKEN=${HF_TOKEN} \
    -e HF_HOME=/hf_cache \
    -e MODEL_CACHE_DIR=/hf_cache \
    -e HOME=/tmp \
    -v /etc/passwd:/etc/passwd:ro \
    -v ${HF_CACHE_DIR}:/hf_cache \
    -v ${PROJECT_ROOT}:/workspace \
    -v ${BASE_DATA_DIR}:/dataset_root \
    -v ${COC_CACHE_DIR}:/coc_cache \
    -w /workspace/src \
    ${DOCKER_IMAGE} \
    python -m coc_labeling.batch_data_labeling \
    --dates ${DATES} \
    --data_dir_root /dataset_root \
    --batch_outputs_root /workspace/batch_outputs \
    --coc_cache_dir /coc_cache \
    --model_name ${MODEL_NAME}

echo ">>> Step 3 completed!"
echo "All processing finished successfully!"
