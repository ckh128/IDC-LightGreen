#!/usr/bin/env python3
"""Bundle COCO annotations and their images into one portable ZIP dataset."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.annotations_zip) as source:
        annotation = source.read("annotations/instances_default.json")
    images = sorted(args.images.glob("d02_*.jpg"))
    if not images:
        raise SystemExit(f"No D02 images found in {args.images}")

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        target.writestr("annotations/instances_default.json", annotation)
        for index, image in enumerate(images, start=1):
            target.write(image, arcname=f"images/{image.name}")
            if index % 100 == 0 or index == len(images):
                print(f"{index}/{len(images)} images")
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
