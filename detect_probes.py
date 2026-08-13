#!/usr/bin/env python3
"""Run YOLO on held-out probe images and make annotated probe videos.

Example:
  python3 detect_probes.py --weights runs/detect/train/weights/best.pt

Probe frames are never moved into training folders by this program.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    sys.exit("Install dependencies first: python3 -m pip install ultralytics opencv-python")


def default_data_directory(name: str) -> Path:
    project = Path(__file__).resolve().parent
    shared = project.parent / name
    return shared if shared.is_dir() else project / name


def sort_key(path: Path) -> tuple[int, int, int]:
    match = re.fullmatch(r"d(\d+)_(\d+)_p(\d+)\.jpg", path.name)
    if not match:
        return (999999, 999999, 999999)
    return tuple(map(int, match.groups()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate held-out temporal probe frames with YOLO.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO .pt weights file.")
    parser.add_argument("--images-root", type=Path, default=default_data_directory("raw_images"))
    parser.add_argument("--processed-images", type=Path, default=default_data_directory("outputs") / "images")
    parser.add_argument("--processed-videos", type=Path, default=default_data_directory("outputs") / "videos")
    parser.add_argument("--raw-archive", type=Path, default=default_data_directory("raw_images") / "d00",
                        help="Copy annotated probe JPGs to this review-only archive.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def make_video(images: list[Path], destination: Path, fps: float) -> None:
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {destination}")
    for path in images:
        image = cv2.imread(str(path))
        if image is not None:
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    writer.release()


def main() -> None:
    args = parse_args()
    probe = args.images_root / "probe"
    if not args.weights.is_file():
        sys.exit(f"Weights file does not exist: {args.weights}")
    images = sorted(probe.glob("d*_p*.jpg"), key=sort_key)
    if not images:
        print(f"No probe images found in: {probe}")
        return

    args.processed_images.mkdir(parents=True, exist_ok=True)
    args.processed_videos.mkdir(parents=True, exist_ok=True)
    args.raw_archive.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    grouped: dict[int, list[Path]] = {}
    archive_index = len(list(args.raw_archive.glob("d00_*.jpg"))) + 1
    for image in images:
        output = args.processed_images / image.name
        result = model.predict(source=str(image), conf=args.confidence, verbose=False)[0]
        if not cv2.imwrite(str(output), result.plot()):
            raise RuntimeError(f"Could not write {output}")
        archive = args.raw_archive / f"d00_{archive_index:06d}.jpg"
        while archive.exists():
            archive_index += 1
            archive = args.raw_archive / f"d00_{archive_index:06d}.jpg"
        shutil.copy2(output, archive)
        archive_index += 1
        session = sort_key(image)[0]
        grouped.setdefault(session, []).append(output)

    for session, frames in grouped.items():
        video = args.processed_videos / f"d{session:02d}_probe_detections.mp4"
        make_video(frames, video, args.fps)
        print(f"Created: {video}")
    print(f"Annotated {len(images)} probe images in: {args.processed_images}")


if __name__ == "__main__":
    main()
