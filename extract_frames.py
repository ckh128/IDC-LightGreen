#!/usr/bin/env python3
"""Extract one JPG for every five frames in raw_videos.

Run from the project folder:
    python3 extract_frames.py

New, unlabelled images go to raw_images/in/. Run a model/review pass to move
them to raw_images/keep/, raw_images/review/, or raw_images/drop/. Existing JPGs are left untouched, so
the command is safe to run repeatedly after adding more videos.
"""

from __future__ import annotations

import argparse
import json
import re
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


def default_data_directory(name: str) -> Path:
    """Prefer a shared data folder next to the repository when it exists."""
    project = Path(__file__).resolve().parent
    shared = project.parent / name
    return shared if shared.is_dir() else project / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract every Nth frame as JPG.")
    parser.add_argument("--videos", type=Path, default=default_data_directory("raw_videos"))
    parser.add_argument("--images", type=Path, default=default_data_directory("raw_images") / "in")
    parser.add_argument("--every", type=int, default=5, help="Extract every Nth frame (default: 5).")
    return parser.parse_args()


def next_dataset_id(images_root: Path) -> int:
    """Choose the next dNN session id from all previously extracted JPGs."""
    ids = []
    for image in images_root.rglob("*.jpg"):
        match = re.match(r"d(\d+)_\d+\.jpg$", image.name)
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=0) + 1


def load_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        return set(json.loads(state_path.read_text(encoding="utf-8")).get("processed_videos", []))
    except (json.JSONDecodeError, OSError) as error:
        sys.exit(f"Cannot read extraction state {state_path}: {error}")


def save_state(state_path: Path, processed: set[str]) -> None:
    state_path.write_text(
        json.dumps({"processed_videos": sorted(processed)}, indent=2) + "\n",
        encoding="utf-8",
    )


def extract(video: Path, destination: Path, every: int, dataset_id: int, start: int) -> tuple[int, int, int]:
    """Extract selected frames, returning (new_images, selected_frames)."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video.name}", file=sys.stderr)
        return 0, 0, start

    added = selected = frame_index = 0
    sequence = start
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every == 0:
            selected += 1
            output = destination / f"d{dataset_id:02d}_{sequence:06d}.jpg"
            if not output.exists():
                if not cv2.imwrite(str(output), frame):
                    capture.release()
                    raise RuntimeError(f"Could not write {output}")
                added += 1
            sequence += 1
        frame_index += 1
    capture.release()
    return added, selected, sequence


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        sys.exit("--every must be a positive integer.")
    if not args.videos.is_dir():
        sys.exit(f"Video folder does not exist: {args.videos}")

    args.images.mkdir(parents=True, exist_ok=True)
    state_path = args.images.parent / ".extract_frames_state.json"
    processed = load_state(state_path)
    videos = sorted(
        path for path in args.videos.iterdir()
        if path.suffix.lower() in VIDEO_SUFFIXES and path.name not in processed
    )
    if not videos:
        print("No new videos to extract.")
        return

    dataset_id = next_dataset_id(args.images.parent)
    sequence = 1
    total_added = 0
    for video in videos:
        added, selected, sequence = extract(video, args.images, args.every, dataset_id, sequence)
        total_added += added
        print(f"{video.name}: {added} new JPG(s), {selected} selected frame(s)")
        processed.add(video.name)
        save_state(state_path, processed)
    print(f"Done. {total_added} new image(s) saved to: {args.images} (d{dataset_id:02d})")


if __name__ == "__main__":
    main()
