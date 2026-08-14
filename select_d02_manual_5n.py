#!/usr/bin/env python3
"""Select evenly spaced D02 5n images for the first manual-label round."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT / "raw_images" / "in")
    parser.add_argument("--output", type=Path, default=PROJECT / "labeling" / "d02" / "manual_5n_round1")
    parser.add_argument("--count", type=int, default=90)
    args = parser.parse_args()
    images = sorted(args.source.glob("d02_*.jpg"))
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    if not images or args.count < 1:
        raise SystemExit("No D02 images or invalid count")
    count = min(args.count, len(images))
    selected = [images[round(index * (len(images) - 1) / (count - 1))] for index in range(count)] if count > 1 else [images[len(images) // 2]]
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True)
    for image in selected:
        link_or_copy(image, image_dir / image.name)
    with (args.output / "selection_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "selected_for_manual_5n"))
        selected_names = {image.name for image in selected}
        writer.writerows((image.name, image.name in selected_names) for image in images)
    archive = args.output / "cvat_images.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
        for image in selected:
            handle.write(image_dir / image.name, arcname=image.name)
    print(f"source={len(images)} selected={len(selected)} zip={archive}")


if __name__ == "__main__":
    main()
