#!/bin/bash
set -e

DATES=(
    "2025-12-09"
    "2025-12-17"
    "2025-12-18"
    "2025-12-23"
    "2025-12-24"
    "2025-12-25"
)

cd "$HOME/alpamayo-coc-autolabeler"
source .venv/bin/activate

for DATE in "${DATES[@]}"; do
    echo "=========================================="
    echo "[$DATE] Starting visualization"
    echo "=========================================="
    python scripts/visualize_keyframe_labels.py "$DATE"

    VIZ_DIR="batch_outputs/$DATE/keyframe_viz"
    MP4="$VIZ_DIR/keyframe_viz_${DATE}.mp4"
    FILELIST="$VIZ_DIR/filelist.txt"

    echo "[$DATE] Generating filelist.txt"
    (
        cd "$VIZ_DIR"
        > filelist.txt
        for f in $(ls *.png | sort); do
            echo "file '$f'" >> filelist.txt
            echo "duration 0.5" >> filelist.txt
        done
    )

    echo "[$DATE] Creating video"
    ffmpeg -y -f concat -safe 0 -i "$FILELIST" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -c:v libx264 -pix_fmt yuv420p -crf 23 \
        "$MP4"

    echo "[$DATE] Done -> $MP4"
done

echo "All dates completed."
