#!/usr/bin/env python3
"""Create a compact, diverse D02 batch for fully manual CVAT labelling.

No model boxes are used. Consecutive frames with a small dHash distance are
treated as visually redundant. If the remaining set is still large, evenly
spaced representatives are selected so both source videos and their time
segments remain represented.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path

import cv2


PROJECT = Path(__file__).resolve().parent.parent


def dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def evenly_spaced(items: list[tuple[Path, int]], count: int) -> list[tuple[Path, int]]:
    if len(items) <= count:
        return items
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=PROJECT / "raw_images" / "in")
    parser.add_argument("--output", type=Path, default=PROJECT / "labeling" / "d02" / "manual_round1")
    parser.add_argument("--duplicate-distance", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    if args.max_images < 1:
        raise SystemExit("--max-images must be positive")

    images = sorted(args.images.glob("d02_*.jpg"))
    if not images:
        raise SystemExit(f"No D02 images in {args.images}")
    candidates: list[tuple[Path, int]] = []
    records: list[tuple[str, int, str, bool]] = []
    previous_hash: int | None = None
    for index, path in enumerate(images, start=1):
        current_hash = dhash(path)
        distance = 64 if previous_hash is None else (current_hash ^ previous_hash).bit_count()
        duplicate = previous_hash is not None and distance <= args.duplicate_distance
        if not duplicate:
            candidates.append((path, distance))
        records.append((path.name, distance, "near_duplicate" if duplicate else "candidate", False))
        previous_hash = current_hash
    selected = evenly_spaced(candidates, args.max_images)
    selected_names = {path.name for path, _ in selected}
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True)
    for path, _ in selected:
        link_or_copy(path, image_dir / path.name)
    with (args.output / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "dhash_distance_from_previous", "filter_result", "selected_manual_round1"))
        for name, distance, status, _ in records:
            writer.writerow((name, distance, status, name in selected_names))
    archive = args.output / "cvat_images.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for path in sorted(image_dir.glob("*.jpg")):
            handle.write(path, arcname=path.name)
    print(f"total={len(images)} non_duplicate={len(candidates)} selected={len(selected)}")
    print(f"Created {archive}")


if __name__ == "__main__":
    main()
