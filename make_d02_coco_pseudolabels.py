#!/usr/bin/env python3
"""Create CVAT COCO annotations for newly extracted D02 5n frames.

The images remain in raw_images/in.  This creates only the annotation ZIP;
upload the D02 images to a CVAT task first, then import this ZIP as COCO 1.0.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parent.parent
CLASSES = {0: "human", 1: "robot"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=PROJECT / "raw_images" / "in")
    parser.add_argument("--weights", type=Path, default=PROJECT / "models" / "d01" / "best_d01.pt")
    parser.add_argument("--output", type=Path, default=PROJECT / "exports" / "d02" / "d02_coco_annotations.zip")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    images = sorted(args.images.glob("d02_*.jpg"))
    if not images:
        raise SystemExit(f"No D02 JPG images found in {args.images}")
    if not args.weights.is_file():
        raise SystemExit(f"Model weights not found: {args.weights}")

    coco = {
        "info": {"description": "D02 5n-frame pseudo-labels from best_d01.pt"},
        "licenses": [],
        "categories": [{"id": key + 1, "name": value, "supercategory": "object"} for key, value in CLASSES.items()],
        "images": [],
        "annotations": [],
    }
    model = YOLO(str(args.weights))
    annotation_id = 1
    for image_id, path in enumerate(images, start=1):
        result = model.predict(source=str(path), conf=args.confidence, verbose=False)[0]
        height, width = result.orig_shape
        coco["images"].append({"id": image_id, "file_name": path.name, "width": width, "height": height})
        if result.boxes is not None:
            for cls, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(), strict=True):
                x1, y1, x2, y2 = xyxy
                box_width, box_height = x2 - x1, y2 - y1
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [round(x1, 3), round(y1, 3), round(box_width, 3), round(box_height, 3)],
                    "area": round(box_width * box_height, 3),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                annotation_id += 1
        if image_id % 100 == 0 or image_id == len(images):
            print(f"{image_id}/{len(images)} images")

    export_root = args.output.with_suffix("")
    if export_root.exists():
        shutil.rmtree(export_root)
    annotation_dir = export_root / "annotations"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "instances_default.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    shutil.make_archive(str(args.output.with_suffix("")), "zip", export_root)
    print(f"Created {args.output}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


if __name__ == "__main__":
    main()
