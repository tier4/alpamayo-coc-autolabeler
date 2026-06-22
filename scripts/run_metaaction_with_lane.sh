#!/bin/bash
set -e

PROJECT_ROOT="$HOME/alpamayo-coc-autolabeler"
BASE_DATA_DIR="/mnt/storage_rdma/datasets/tier4/e2e-val"
OUTPUT_DIR="$HOME/alpamayo_labels_metaaction"
MAP_BASE="/mnt/storage_rdma/datasets/tier4/maps/1423"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

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

DATES=$(find "${BASE_DATA_DIR}" -mindepth 1 -maxdepth 1 -type d -name "20*" -printf "%f\n" | sort)

echo "=========================================="
echo "Meta-Action Generation with Lane (val data)"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

for DATE in $DATES; do
    MAP_VERSION="${DATE_TO_MAP_VERSION[$DATE]}"
    LANELET2_MAP="${MAP_BASE}/${MAP_VERSION}/lanelet2_map.osm"

    if [ ! -f "$LANELET2_MAP" ]; then
        echo "[SKIP] $DATE: lanelet2_map not found at $LANELET2_MAP"
        continue
    fi

    SAVE_DIR="${OUTPUT_DIR}/${DATE}/meta_actions"
    mkdir -p "$SAVE_DIR"

    echo "[RUN] $DATE (map: $MAP_VERSION)"
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
    echo "[DONE] $DATE"
    echo ""
done

echo "All dates processed! Output: ${OUTPUT_DIR}"
