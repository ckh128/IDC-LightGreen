#!/usr/bin/env bash
# IMX500 / Raspberry Pi AI Camera video recorder for YOLO dataset collection.
#
# Usage:
#   ./record_imx500.sh                  # record until Ctrl+C, with preview
#   ./record_imx500.sh 300 1920 1080 30 # record for 5 minutes, 1080p, 30 fps
#
# Video files are never overwritten. They are stored under:
#   ~/yolo_videos/YYYY-MM-DD/

set -euo pipefail

DURATION_SECONDS="${1:-0}" # 0 = record until Ctrl+C
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
FPS="${4:-30}"
OUTPUT_ROOT="${OUTPUT_DIR:-$HOME/yolo_videos}"
RECORDING_DATE="$(date +%F)"
OUTPUT_DIR="$OUTPUT_ROOT/$RECORDING_DATE"

if ! command -v rpicam-vid >/dev/null 2>&1; then
    echo "rpicam-vid was not found. Install or update Raspberry Pi camera software." >&2
    exit 1
fi

# Preview requires an active Raspberry Pi desktop display (local HDMI, VNC,
# NoMachine, or Raspberry Pi Connect). Do not run with --nopreview.
mkdir -p "$OUTPUT_DIR"

# Nanoseconds plus a collision check make every recording name unique.
TIMESTAMP="$(date +%Y%m%d_%H%M%S_%N)"
OUTPUT_FILE="$OUTPUT_DIR/imx500_${TIMESTAMP}_${WIDTH}x${HEIGHT}_${FPS}fps.mp4"
COUNT=1
while [[ -e "$OUTPUT_FILE" ]]; do
    OUTPUT_FILE="$OUTPUT_DIR/imx500_${TIMESTAMP}_${COUNT}_${WIDTH}x${HEIGHT}_${FPS}fps.mp4"
    ((COUNT++))
done

echo "Recording: $OUTPUT_FILE"
echo "Resolution: ${WIDTH}x${HEIGHT}; FPS: $FPS; duration: $DURATION_SECONDS seconds (0 = until Ctrl+C)"
echo "A camera preview window should appear. Press Ctrl+C in this terminal to stop."

# On Raspberry Pi 5, an .mp4 output is written directly. The preview window
# lets you verify framing while recording.
rpicam-vid \
    --timeout "$((DURATION_SECONDS * 1000))" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --framerate "$FPS" \
    --output "$OUTPUT_FILE"

echo "Recording complete: $OUTPUT_FILE"
