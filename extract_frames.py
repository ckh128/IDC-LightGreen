#!/usr/bin/env python3
"""Extract one JPG for every five frames in raw_videos.

Run from the project folder:
    python3 extract_frames.py

New, unlabelled images go to raw_images/review/. Move them to raw_images/keep/
or raw_images/drop/ after reviewing them. Existing JPGs are left untouched, so
the command is safe to run repeatedly after adding more videos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit(
        "OpenCV is required. Install it with: "
        "python3 -m pip install opencv-python"
    )


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".h264"}


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Extract every Nth frame as JPG.")
    parser.add_argument("--videos", type=Path, default=project / "raw_videos")
    parser.add_argument("--images", type=Path, default=project / "raw_images" / "review")
    parser.add_argument("--every", type=int, default=5, help="Extract every Nth frame (default: 5).")
    return parser.parse_args()


def extract(video: Path, destination: Path, every: int) -> tuple[int, int]:
    """Extract selected frames, returning (new_images, selected_frames)."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video.name}", file=sys.stderr)
        return 0, 0

    added = selected = frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every == 0:
            selected += 1
            # Preserve the source video name (including names such as d__...)
            # and the original frame index, so every output filename is stable.
            output = destination / f"{video.stem}_f{frame_index:06d}.jpg"
            if not output.exists():
                if not cv2.imwrite(str(output), frame):
                    capture.release()
                    raise RuntimeError(f"Could not write {output}")
                added += 1
        frame_index += 1
    capture.release()
    return added, selected


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        sys.exit("--every must be a positive integer.")
    if not args.videos.is_dir():
        sys.exit(f"Video folder does not exist: {args.videos}")

    args.images.mkdir(parents=True, exist_ok=True)
    videos = sorted(path for path in args.videos.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f"No videos found in: {args.videos}")
        return

    total_added = 0
    for video in videos:
        added, selected = extract(video, args.images, args.every)
        total_added += added
        print(f"{video.name}: {added} new JPG(s), {selected} selected frame(s)")
    print(f"Done. {total_added} new image(s) saved to: {args.images}")


if __name__ == "__main__":
    main()
