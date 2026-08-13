#!/usr/bin/env python3
"""Extract training and held-out temporal-probe frames from raw videos.

For every group of five source frames:
  - 5n     -> raw_images/in/    (candidate for training / active learning)
  - 5n + 1, 5n + 3 -> raw_images/probe/ (never training data; model checks only)

Each new batch receives the next dataset id: d03, d04, and so on.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required. Install it with: python3 -m pip install opencv-python")

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".h264"}


def default_data_directory(name: str) -> Path:
    project = Path(__file__).resolve().parent
    shared = project.parent / name
    return shared if shared.is_dir() else project / name


def next_dataset_id(images_root: Path) -> int:
    ids = []
    for image in images_root.rglob("*.jpg"):
        match = re.match(r"d(\d+)_\d+", image.name)
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=0) + 1


def load_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        return set(json.loads(state_path.read_text(encoding="utf-8-sig")).get("processed_videos", []))
    except (json.JSONDecodeError, OSError) as error:
        sys.exit(f"Cannot read extraction state {state_path}: {error}")


def save_state(state_path: Path, processed: set[str]) -> None:
    state_path.write_text(json.dumps({"processed_videos": sorted(processed)}, indent=2) + "\n", encoding="utf-8")


def write_jpg(path: Path, image: object) -> bool:
    if path.exists():
        return False
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")
    return True


def extract_video(
    video: Path, intake: Path, probe: Path, every: int, dataset_id: int, start: int
) -> tuple[int, int, int, list[tuple[str, str, int, int]]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"Skipping unreadable video: {video.name}", file=sys.stderr)
        return 0, 0, start, []

    intake_added = probe_added = frame_index = 0
    manifest: list[tuple[str, str, int, int]] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        remainder = frame_index % every
        sequence = start + frame_index // every
        prefix = f"d{dataset_id:02d}_{sequence:06d}"
        if remainder == 0:
            if write_jpg(intake / f"{prefix}.jpg", frame):
                intake_added += 1
        elif remainder in (1, 3):
            path = probe / f"{prefix}_p{remainder}.jpg"
            if write_jpg(path, frame):
                probe_added += 1
            manifest.append((path.name, video.name, frame_index, remainder))
        frame_index += 1
    capture.release()
    next_sequence = start + (frame_index + every - 1) // every
    return intake_added, probe_added, next_sequence, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract active-learning and probe frames.")
    parser.add_argument("--videos", type=Path, default=default_data_directory("raw_videos"))
    parser.add_argument("--images-root", type=Path, default=default_data_directory("raw_images"))
    parser.add_argument("--every", type=int, default=5, help="Frame-group size (default: 5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every != 5:
        sys.exit("This workflow uses groups of exactly five frames; keep --every 5.")
    if not args.videos.is_dir():
        sys.exit(f"Video folder does not exist: {args.videos}")

    intake, probe, meta = (args.images_root / name for name in ("in", "probe", "meta"))
    for folder in (intake, probe, meta):
        folder.mkdir(parents=True, exist_ok=True)
    state_path = args.images_root / ".extract_frames_state.json"
    processed = load_state(state_path)
    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and p.name not in processed)
    if not videos:
        print("No new videos to extract.")
        return

    dataset_id = next_dataset_id(args.images_root)
    sequence = 1
    all_manifest: list[tuple[str, str, int, int]] = []
    total_intake = total_probe = 0
    for video in videos:
        added_in, added_probe, sequence, manifest = extract_video(video, intake, probe, 5, dataset_id, sequence)
        total_intake += added_in
        total_probe += added_probe
        all_manifest.extend(manifest)
        processed.add(video.name)
        save_state(state_path, processed)
        print(f"{video.name}: {added_in} intake, {added_probe} probe JPG(s)")

    log = meta / f"d{dataset_id:02d}_probe_manifest.csv"
    with log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("probe_image", "source_video", "source_frame", "offset"))
        writer.writerows(all_manifest)
    print(f"Done: {total_intake} in/, {total_probe} probe/. Manifest: {log}")


if __name__ == "__main__":
    main()
