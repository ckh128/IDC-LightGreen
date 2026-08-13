#!/usr/bin/env python3
"""Build a leak-resistant YOLO dataset from D02 COCO pseudo-labels.

This is intentionally a *test* dataset: labels are the D01 model predictions
and have not been manually corrected.  Empty-label images are preserved as
negative examples.  Consecutive video frames are kept in 30-frame windows
before assigning train/val/test, reducing temporal leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.20, 0.10)


def yolo_line(box: list[float], width: int, height: int, class_id: int) -> str:
    x, y, w, h = box
    return f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=PROJECT / "raw_images" / "in")
    parser.add_argument("--coco", type=Path, default=PROJECT / "exports" / "d02" / "d02_coco_annotations.zip")
    parser.add_argument("--output", type=Path, default=PROJECT / "datasets" / "d02_pseudolabel_test")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    if args.window < 1:
        raise SystemExit("--window must be positive")

    with zipfile.ZipFile(args.coco) as archive:
        coco = json.loads(archive.read("annotations/instances_default.json"))
    annotations: dict[int, list[dict]] = {}
    for item in coco["annotations"]:
        annotations.setdefault(item["image_id"], []).append(item)
    records = []
    for image in coco["images"]:
        path = args.images / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing D02 image: {path}")
        records.append((image, path, annotations.get(image["id"], [])))
    records.sort(key=lambda item: int(item[1].stem.rsplit("_", 1)[1]))

    windows = [records[index:index + args.window] for index in range(0, len(records), args.window)]
    random.Random(42).shuffle(windows)
    targets = [len(records) * ratio for ratio in RATIOS]
    assigned = {split: [] for split in SPLITS}
    for window in windows:
        split = max(range(len(SPLITS)), key=lambda index: targets[index] - len(assigned[SPLITS[index]]))
        assigned[SPLITS[split]].extend(window)

    rows: list[tuple[str, str, int]] = []
    for split, items in assigned.items():
        images_dir, labels_dir = args.output / "images" / split, args.output / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image, source, boxes in items:
            shutil.copy2(source, images_dir / source.name)
            lines = [yolo_line(box["bbox"], image["width"], image["height"], box["category_id"] - 1) for box in boxes]
            (labels_dir / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append((source.name, split, len(boxes)))
    (args.output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "split", "box_count"))
        writer.writerows(sorted(rows))
    print(" ".join(f"{split}={len(assigned[split])}" for split in SPLITS))
    print(f"Created {args.output} from {len(records)} D02 images")


if __name__ == "__main__":
    main()
