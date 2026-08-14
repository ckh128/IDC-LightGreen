#!/usr/bin/env python3
"""Prepare a small, representative high-confidence D02 review batch for CVAT.

Source images are never moved.  All D02 images are recorded in a manifest:
  - high: every detection has confidence >= --high-confidence
  - low: a detection exists, but at least one confidence is lower
  - empty: no detection (kept for a later negative/false-negative check)

Round 1 selects an evenly spaced --sample-rate fraction of high images.  Its
images and predicted boxes are bundled in a portable COCO ZIP for CVAT review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parent.parent
CLASSES = {0: "human", 1: "robot"}


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def evenly_spaced(items: list[dict], rate: float) -> list[dict]:
    count = max(1, round(len(items) * rate)) if items else 0
    if count >= len(items):
        return items
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)] if count > 1 else [items[len(items) // 2]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=PROJECT / "raw_images" / "in")
    parser.add_argument("--weights", type=Path, default=PROJECT / "models" / "d01" / "best_d01.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "labeling" / "d02" / "round1_high_confidence")
    parser.add_argument("--high-confidence", type=float, default=0.85)
    parser.add_argument("--sample-rate", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.sample_rate <= 1:
        raise SystemExit("--sample-rate must be in (0, 1]")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    paths = sorted(args.images.glob("d02_*.jpg"))
    if not paths or not args.weights.is_file():
        raise SystemExit("D02 images or D01 weights are missing")
    model = YOLO(str(args.weights))
    records: list[dict] = []
    for index, path in enumerate(paths, start=1):
        result = model.predict(source=str(path), conf=0.25, verbose=False)[0]
        boxes = [] if result.boxes is None else list(zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True))
        category = "empty" if not boxes else ("high" if min(box[2] for box in boxes) >= args.high_confidence else "low")
        height, width = result.orig_shape
        records.append({"path": path, "name": path.name, "width": width, "height": height, "category": category, "boxes": boxes})
        if index % 100 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} classified")

    high = [record for record in records if record["category"] == "high"]
    selected = evenly_spaced(high, args.sample_rate)
    image_dir = args.output / "images"
    annotation_dir = args.output / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    coco = {"info": {"description": "D02 round-1 high-confidence pseudo-label sample"}, "licenses": [],
            "categories": [{"id": key + 1, "name": value, "supercategory": "object"} for key, value in CLASSES.items()],
            "images": [], "annotations": []}
    annotation_id = 1
    for image_id, record in enumerate(selected, start=1):
        link_or_copy(record["path"], image_dir / record["name"])
        coco["images"].append({"id": image_id, "file_name": record["name"], "width": record["width"], "height": record["height"]})
        for cls, xyxy, confidence in record["boxes"]:
            x1, y1, x2, y2 = xyxy
            width, height = x2 - x1, y2 - y1
            coco["annotations"].append({"id": annotation_id, "image_id": image_id, "category_id": int(cls) + 1,
                                        "bbox": [round(x1, 3), round(y1, 3), round(width, 3), round(height, 3)],
                                        "area": round(width * height, 3), "iscrowd": 0, "segmentation": [],
                                        "score": round(float(confidence), 6)})
            annotation_id += 1
    (annotation_dir / "instances_default.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    with (args.output / "triage_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "bucket", "box_count", "minimum_confidence", "selected_round1"))
        selected_names = {record["name"] for record in selected}
        for record in records:
            scores = [box[2] for box in record["boxes"]]
            writer.writerow((record["name"], record["category"], len(scores), f"{min(scores):.6f}" if scores else "", record["name"] in selected_names))
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(args.output))
    counts = {bucket: sum(record["category"] == bucket for record in records) for bucket in ("high", "low", "empty")}
    print(f"high={counts['high']} low={counts['low']} empty={counts['empty']} selected={len(selected)} boxes={len(coco['annotations'])}")
    print(f"Created {archive}")


if __name__ == "__main__":
    main()
