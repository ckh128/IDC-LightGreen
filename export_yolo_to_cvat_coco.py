#!/usr/bin/env python3
"""Convert a YOLO image/label folder into a CVAT-importable COCO annotation ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Folder containing images/, labels/, data.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Output ZIP path")
    args = parser.parse_args()

    names = {0: "human", 1: "robot"}
    images_dir, labels_dir = args.source / "images", args.source / "labels"
    export_root = args.output.with_suffix("")
    if export_root.exists():
        shutil.rmtree(export_root)
    annotation_dir = export_root / "annotations"
    annotation_dir.mkdir(parents=True)

    coco = {
        "info": {"description": "D02 pseudo-labels exported for CVAT review"},
        "licenses": [],
        "categories": [{"id": index + 1, "name": name, "supercategory": "object"} for index, name in names.items()],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    for image_id, image_path in enumerate(sorted(images_dir.glob("*.jpg")), start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        coco["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, center_x, center_y, box_width, box_height = map(float, line.split()[:5])
            x = (center_x - box_width / 2) * width
            y = (center_y - box_height / 2) * height
            w, h = box_width * width, box_height * height
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id) + 1,
                "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                "area": round(w * h, 3),
                "iscrowd": 0,
                "segmentation": [],
            })
            annotation_id += 1

    (annotation_dir / "instances_default.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    archive_base = args.output.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", export_root)
    print(f"Created {args.output}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


if __name__ == "__main__":
    main()
