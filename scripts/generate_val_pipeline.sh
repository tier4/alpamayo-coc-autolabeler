#!/bin/bash

# Pipeline for VALIDATION data annotation.
# Same 4-step pipeline as generate_all_pipeline.sh but pointed at the eval dataset.
set -e

# ==========================================
# 1. Common settings and path definitions
# ==========================================
PROJECT_ROOT="$HOME/alpamayo-coc-autolabeler"
BASE_DATA_DIR="/mnt/storage_rdma/datasets/tier4/e2e-val"
HF_CACHE_DIR="/mnt/nvme/hf_cache"
COC_CACHE_DIR="$HOME/coc_cache"

OUTPUT_SUBDIR="alpamayo_labels_val"
BASE_OUTPUT_DIR="$HOME/${OUTPUT_SUBDIR}"

DOCKER_IMAGE="coc_autolabeler:latest"
MODEL_NAME="qwen3.5_397b_fp8"

MAP_BASE="/mnt/storage_rdma/datasets/tier4/maps/1423"

# Date -> map version mapping (from generate_navigation_labels.py)
declare -A DATE_TO_MAP_VERSION=(
    ["2025-06-12"]="1423-20250523084638229189"
    ["2025-06-16"]="1423-20250613084329651701"
    ["2025-06-25"]="1423-20250619104812759171"
    ["2025-07-28"]="1423-20250630062747432546"
    ["2025-08-08"]="1423-20250630062747432546"
    ["2025-08-13"]="1423-20250630062747432546"
    ["2025-09-25"]="1423-20250905061011941236"
    ["2025-10-08"]="1423-20250905061011941236"
    ["2025-10-15"]="1423-20250905061011941236"
    ["2025-10-22"]="1423-20250905061011941236"
    ["2025-10-29"]="1423-20251024074044664283"
    ["2025-11-12"]="1423-20250905061011941236"
    ["2025-11-25"]="1423-20250905061011941236"
    ["2025-12-09"]="1423-20251209045313582007"
)

# ==========================================
# 2. Enumerate target date directories
# ==========================================
DATES=$(find "${BASE_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name "20*" -printf "%f\n" | sort)

echo "=========================================="
echo "Target dates for processing (validation):"
echo "$DATES"
echo "=========================================="
echo ""

# Pre-create all per-date output directories on the host to guarantee host-user ownership.
for DATE in $DATES; do
    mkdir -p "${BASE_OUTPUT_DIR}/${DATE}/meta_actions"
    mkdir -p "${BASE_OUTPUT_DIR}/${DATE}/keyframes"
done


# ==========================================
# Step 1: Meta-Action generation for all dates
# ==========================================
echo ">>> Starting Step 1: Meta-Action Generation for all dates"
for DATE in $DATES; do
    echo "[Step 1] Processing: $DATE"
    TARGET_DATA_DIR="${BASE_DATA_DIR}/${DATE}"

    MAP_VERSION="${DATE_TO_MAP_VERSION[$DATE]}"
    LANE_ARGS=""
    if [ -n "$MAP_VERSION" ]; then
        LANELET2_MAP="${MAP_BASE}/${MAP_VERSION}/lanelet2_map.osm"
        if [ -f "$LANELET2_MAP" ]; then
            LANE_ARGS="--use_lane --lanelet2_map /maps/lanelet2_map.osm"
            echo "  Using lanelet2 map: $LANELET2_MAP"
        else
            echo "  [WARN] lanelet2_map.osm not found: $LANELET2_MAP, skipping lane meta-actions"
        fi
    else
        echo "  [WARN] No map version for $DATE, skipping lane meta-actions"
    fi

    docker run --rm \
        --user $(id -u):$(id -g) \
        -v ${PROJECT_ROOT}:/workspace \
        -v ${TARGET_DATA_DIR}:/dataset \
        -v ${COC_CACHE_DIR}:/coc_cache \
        -v ${BASE_OUTPUT_DIR}:/outputs \
        ${MAP_VERSION:+-v ${MAP_BASE}/${MAP_VERSION}:/maps:ro} \
        -w /workspace/src \
        ${DOCKER_IMAGE} \
        python -m meta_action.data_labeling \
        --data_dir /dataset \
        --cache_dir /coc_cache \
        --save_dir /outputs/${DATE}/meta_actions \
        --data_format webdataset \
        --meta_action_names all_ego \
        ${LANE_ARGS}
done
echo ">>> Step 1 completed!"
echo ""


# ==========================================
# Step 2: Keyframe selection for all dates
# ==========================================
echo ">>> Starting Step 2: Keyframe Selection for all dates"
for DATE in $DATES; do
    echo "[Step 2] Processing: $DATE"

    docker run --rm \
        --user $(id -u):$(id -g) \
        -v ${PROJECT_ROOT}:/workspace \
        -v ${BASE_OUTPUT_DIR}:/outputs \
        -w /workspace/src \
        ${DOCKER_IMAGE} \
        python -m coc_labeling.keyframe_auto_select \
        --meta_action_dir /outputs/${DATE}/meta_actions/final_outputs \
        --output_dir /outputs/${DATE}/keyframes
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
    -v ${BASE_OUTPUT_DIR}:/outputs \
    -w /workspace/src \
    ${DOCKER_IMAGE} \
    python -m coc_labeling.batch_data_labeling \
    --dates ${DATES} \
    --data_dir_root /dataset_root \
    --batch_outputs_root /outputs \
    --coc_cache_dir /coc_cache \
    --model_name ${MODEL_NAME}

echo ">>> Step 3 completed!"
echo ""


# ==========================================
# Step 4: Unified label generation (nav text + CoC merge)
# ==========================================
echo ">>> Starting Step 4: Unified Label Generation (nav text + CoC merge)"

source "${PROJECT_ROOT}/.venv/bin/activate"
python "${PROJECT_ROOT}/scripts/create_unified_labels.py" \
    --dates ${DATES} \
    --batch_outputs_root "${BASE_OUTPUT_DIR}" \
    --data_dir_root "${BASE_DATA_DIR}"

echo ">>> Step 4 completed!"
echo "All processing finished successfully!"
