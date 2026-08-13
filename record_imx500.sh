#!/usr/bin/env bash
# IMX500 (Raspberry Pi AI Camera) video recorder for building a YOLO dataset.
# Usage examples:
#   ./record_imx500.sh                         # record until Ctrl+C
#   ./record_imx500.sh 300 1920 1080 30        # record 5 minutes, 1080p, 30 fps

set -euo pipefail

DURATION_SECONDS="${1:-0}"  # 0 means record until Ctrl+C
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
FPS="${4:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/yolo_videos}"

if ! command -v rpicam-vid >/dev/null 2>&1; then
    echo "rpicam-vid를 찾지 못했습니다. Raspberry Pi OS의 카메라 도구를 설치/업데이트하세요." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="$OUTPUT_DIR/imx500_${TIMESTAMP}_${WIDTH}x${HEIGHT}_${FPS}fps.mp4"

echo "녹화 파일: $OUTPUT_FILE"
echo "해상도: ${WIDTH}x${HEIGHT}, FPS: $FPS, 시간: $DURATION_SECONDS 초 (0 = Ctrl+C까지)"

# The .mp4 extension makes rpicam-vid create an MP4 file directly on Raspberry Pi 5.
# --nopreview is important when recording through SSH/headless.
rpicam-vid \
    --timeout "$((DURATION_SECONDS * 1000))" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --framerate "$FPS" \
    --nopreview \
    --output "$OUTPUT_FILE"

echo "녹화 완료: $OUTPUT_FILE"
