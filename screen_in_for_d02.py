#!/usr/bin/env python3
"""Screen D01 intake images with best_d01.pt for efficient D02 labelling.

The model's boxes are pseudo-labels, not ground truth.  Images are routed as:
  - drop: no detection (often terminal/empty/unusable frames; never deleted)
  - keep: at least one confident detection; YOLO pseudo-label is ready to verify
  - review: a low-confidence detection; inspect and correct in CVAT

The pseudo-label export can be imported into CVAT as a YOLO detection dataset.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=PROJECT / "models" / "d01" / "best_d01.pt")
    parser.add_argument("--source", type=Path, default=PROJECT / "raw_images" / "in")
    parser.add_argument("--raw-images", type=Path, default=PROJECT / "raw_images")
    parser.add_argument("--export", type=Path, default=PROJECT / "exports" / "d02" / "pseudolabel_yolo")
    parser.add_argument("--previews", type=Path, default=PROJECT / "outputs" / "images" / "d02_pseudolabel_previews")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--keep-confidence", type=float, default=0.75, help="Confidence required for keep.")
    return parser.parse_args()


def write_yolo_labels(path: Path, boxes: object, width: int, height: int) -> None:
    lines: list[str] = []
    for cls, xyxy in zip(boxes.cls.tolist(), boxes.xyxy.tolist(), strict=True):
        x1, y1, x2, y2 = xyxy
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    images = sorted(args.source.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No JPG images in {args.source}")

    folders = {name: args.raw_images / name for name in ("drop", "keep", "review")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    export_images, export_labels = args.export / "images", args.export / "labels"
    export_images.mkdir(parents=True, exist_ok=True)
    export_labels.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    rows: list[tuple[str, str, float, int]] = []
    for index, image_path in enumerate(images, start=1):
        result = model.predict(source=str(image_path), conf=args.confidence, verbose=False)[0]
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        best = 0.0 if not count else float(max(boxes.conf.tolist()))
        bucket = "drop" if not count else ("keep" if best >= args.keep_confidence else "review")
        destination = folders[bucket] / image_path.name
        if destination.exists():
            raise RuntimeError(f"Destination already exists: {destination}")
        shutil.move(str(image_path), str(destination))

        if count:
            preview = args.previews / image_path.name
            if not cv2.imwrite(str(preview), result.plot()):
                raise RuntimeError(f"Cannot write {preview}")
            original = cv2.imread(str(destination))
            if original is None:
                raise RuntimeError(f"Cannot read {destination}")
            height, width = original.shape[:2]
            shutil.copy2(destination, export_images / destination.name)
            write_yolo_labels(export_labels / f"{destination.stem}.txt", boxes, width, height)
        rows.append((destination.name, bucket, best, count))
        if index % 100 == 0 or index == len(images):
            print(f"{index}/{len(images)} screened")

    (args.export / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )
    manifest = args.export / "screening_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "route", "best_confidence", "box_count"))
        writer.writerows(rows)
    print(f"Done. Pseudo-label export: {args.export}; manifest: {manifest}")


if __name__ == "__main__":
    main()
