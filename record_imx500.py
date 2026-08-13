#!/usr/bin/env python3
"""Record IMX500/AI Camera video for a YOLO dataset.

Examples:
  python3 record_imx500.py
  python3 record_imx500.py --duration 300 --width 1920 --height 1080 --fps 30

Videos are saved without overwriting old files in ~/yolo_videos/YYYY-MM-DD/.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from picamera2 import Picamera2, Preview
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
except ImportError:
    sys.exit(
        "Picamera2 is not installed. On Raspberry Pi OS run: "
        "sudo apt update && sudo apt install -y python3-picamera2 ffmpeg"
    )


def make_output_path(root: Path, width: int, height: int, fps: int) -> Path:
    """Return a new path; never reuse or overwrite an existing recording."""
    folder = root / datetime.now().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = folder / f"imx500_{stamp}_{width}x{height}_{fps}fps.mp4"
    number = 1
    while candidate.exists():
        candidate = folder / f"imx500_{stamp}_{number}_{width}x{height}_{fps}fps.mp4"
        number += 1
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record IMX500 video with a live preview.")
    parser.add_argument("--duration", type=float, default=0,
                        help="Recording time in seconds; 0 records until Ctrl+C (default: 0).")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "yolo_videos")
    parser.add_argument("--no-preview", action="store_true",
                        help="Record without the GUI preview (useful when no desktop is running).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration < 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        sys.exit("Duration must be >= 0; width, height, and fps must be positive.")

    output_path = make_output_path(args.output_dir, args.width, args.height, args.fps)
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height)},
        controls={"FrameDurationLimits": (int(1_000_000 / args.fps),) * 2},
    )
    picam2.configure(config)

    # QTGL displays the live camera image in the Raspberry Pi desktop session.
    if not args.no_preview:
        picam2.start_preview(Preview.QTGL)

    encoder = H264Encoder(bitrate=10_000_000)
    output = FfmpegOutput(str(output_path))
    stopping = False

    def stop_requested(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)

    print(f"Recording: {output_path}")
    print("Live preview is open. Press Ctrl+C in this terminal to stop recording.")
    picam2.start_recording(encoder, output)
    started = time.monotonic()

    try:
        while not stopping:
            if args.duration and time.monotonic() - started >= args.duration:
                break
            time.sleep(0.1)
    finally:
        picam2.stop_recording()
        if not args.no_preview:
            picam2.stop_preview()
        picam2.close()

    print(f"Recording complete: {output_path}")


if __name__ == "__main__":
    main()
