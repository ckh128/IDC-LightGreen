"""Annotate D03-D06 temporal probe frames using a trained YOLO model."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROUND_IDS = ("d03", "d04", "d05", "d06")


def image_sort_key(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"d\d+_(\d+)_p([13])\.jpg", path.name)
    if not match:
        raise ValueError(f"Unexpected probe filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def make_video(images: list[Path], output: Path, fps: float) -> None:
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output}")
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    raw_images = args.project_root / "raw_images"
    output_root = args.project_root / "outputs" / "d02_r2_probe_detections"
    video_root = args.project_root / "outputs" / "d02_r2_probe_videos"
    if args.overwrite:
        for directory in (output_root, video_root):
            if directory.exists():
                shutil.rmtree(directory)
    output_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    total = 0
    for round_id in ROUND_IDS:
        images = sorted(
            [* (raw_images / round_id / "p1").glob("*.jpg"), * (raw_images / round_id / "p3").glob("*.jpg")],
            key=image_sort_key,
        )
        expected = len(list((raw_images / round_id / "p1").glob("*.jpg"))) + len(list((raw_images / round_id / "p3").glob("*.jpg")))
        if len(images) != expected or not images:
            raise RuntimeError(f"Missing p1 or p3 images for {round_id}")

        annotated_root = output_root / round_id
        annotated_root.mkdir(parents=True, exist_ok=True)
        annotated: list[Path] = []
        for start in range(0, len(images), args.batch):
            batch = images[start:start + args.batch]
            results = model.predict([str(path) for path in batch], conf=args.confidence, verbose=False)
            for source, result in zip(batch, results):
                destination = annotated_root / source.name
                if not cv2.imwrite(str(destination), result.plot()):
                    raise RuntimeError(f"Cannot write {destination}")
                annotated.append(destination)
            print(f"{round_id}: {min(start + len(batch), len(images))}/{len(images)}")

        video_path = video_root / f"{round_id}_d02_r2_probe.mp4"
        make_video(annotated, video_path, args.fps)
        print(f"Created {video_path}")
        total += len(annotated)
    print(f"Annotated {total} probe images")


if __name__ == "__main__":
    main()
