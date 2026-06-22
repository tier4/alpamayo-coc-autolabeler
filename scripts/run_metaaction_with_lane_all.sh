#!/bin/bash
set -e

PROJECT_ROOT="$HOME/alpamayo-coc-autolabeler"
MAP_BASE="/mnt/storage_rdma/datasets/tier4/maps/1423"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# ==========================================
# Map version lookup
# ==========================================
declare -A DATE_TO_MAP_VERSION=(
    # Train dates
    ["2025-11-19"]="1423-20250905061011941236"
    ["2025-12-09"]="1423-20251209045313582007"
    ["2025-12-10"]="1423-20250905061011941236"
    ["2025-12-17"]="1423-20251210022250515914"
    ["2025-12-18"]="1423-20251210022250515914"
    ["2025-12-23"]="1423-20251210022250515914"
    ["2025-12-24"]="1423-20251210022250515914"
    ["2025-12-25"]="1423-20251210022250515914"
    ["2026-01-06"]="1423-20251210022250515914"
    ["2026-01-07"]="1423-20251210022250515914"
    # Val dates
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
)

process_split() {
    local SPLIT_NAME=$1
    local BASE_DATA_DIR=$2
    local OUTPUT_DIR=$3

    DATES=$(find "${BASE_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name "20*" -printf "%f\n" | sort)

    echo "=========================================="
    echo "[${SPLIT_NAME}] Meta-Action Generation with Lane"
    echo "  Data:   ${BASE_DATA_DIR}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "=========================================="

    for DATE in $DATES; do
        MAP_VERSION="${DATE_TO_MAP_VERSION[$DATE]}"
        LANELET2_MAP="${MAP_BASE}/${MAP_VERSION}/lanelet2_map.osm"

        if [ -z "$MAP_VERSION" ] || [ ! -f "$LANELET2_MAP" ]; then
            echo "[${SPLIT_NAME}] [SKIP] $DATE: no map available"
            continue
        fi

        SAVE_DIR="${OUTPUT_DIR}/${DATE}/meta_actions"
        FINAL_DIR="${SAVE_DIR}/final_outputs"

        # Skip if already processed
        if [ -d "$FINAL_DIR" ] && [ "$(ls -A $FINAL_DIR 2>/dev/null)" ]; then
            EXISTING=$(ls $FINAL_DIR/*.txt 2>/dev/null | wc -l)
            echo "[${SPLIT_NAME}] [SKIP] $DATE: already has $EXISTING scenes"
            continue
        fi

        mkdir -p "$SAVE_DIR"

        echo "[${SPLIT_NAME}] [RUN] $DATE (map: $MAP_VERSION)"
        cd "${PROJECT_ROOT}/src"
        ${PYTHON} -m meta_action.data_labeling \
            --data_dir "${BASE_DATA_DIR}/${DATE}" \
            --cache_dir /tmp/coc_cache \
            --save_dir "${SAVE_DIR}" \
            --data_format webdataset \
            --meta_action_names all_ego \
            --use_lane \
            --lanelet2_map "${LANELET2_MAP}" \
            --num_workers 8
        echo "[${SPLIT_NAME}] [DONE] $DATE"
        echo ""
    done
}

# ==========================================
# Run train and val
# ==========================================
process_split "TRAIN" \
    "/mnt/storage_rdma/datasets/tier4/e2e" \
    "$HOME/alpamayo_labels_metaaction/train"

process_split "VAL" \
    "/mnt/storage_rdma/datasets/tier4/e2e-val" \
    "$HOME/alpamayo_labels_metaaction/val"

echo "=========================================="
echo "All processing finished!"
echo "  Train: $HOME/alpamayo_labels_metaaction/train/"
echo "  Val:   $HOME/alpamayo_labels_metaaction/val/"
echo "=========================================="
