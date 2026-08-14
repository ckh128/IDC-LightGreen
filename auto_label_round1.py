"""Create editable YOLO pseudo-labels and preview images for one round's 5n frames."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    source_images = sorted(args.images.glob("*.jpg"))
    if not source_images:
        raise FileNotFoundError(f"No JPG images found: {args.images}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}")
        shutil.rmtree(args.output)

    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    previews_dir = args.output / "previews"
    for directory in (images_dir, labels_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    manifest_rows: list[dict[str, str | int | float]] = []
    for start in range(0, len(source_images), args.batch):
        batch = source_images[start:start + args.batch]
        results = model.predict([str(path) for path in batch], conf=args.confidence, verbose=False)
        for source, result in zip(batch, results):
            shutil.copy2(source, images_dir / source.name)
            lines: list[str] = []
            confidences: list[float] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    center_x, center_y, width, height = box.xywhn[0].tolist()
                    confidence = float(box.conf.item())
                    lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                    confidences.append(confidence)
            (labels_dir / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if not cv2.imwrite(str(previews_dir / source.name), result.plot()):
                raise RuntimeError(f"Cannot write preview: {source.name}")
            manifest_rows.append({
                "image": source.name,
                "boxes": len(lines),
                "min_confidence": round(min(confidences), 6) if confidences else "",
                "max_confidence": round(max(confidences), 6) if confidences else "",
            })
        print(f"{min(start + len(batch), len(source_images))}/{len(source_images)}")

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("image", "boxes", "min_confidence", "max_confidence"))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Created editable pseudo-labels for {len(source_images)} images: {args.output}")


if __name__ == "__main__":
    main()
