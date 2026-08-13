#!/usr/bin/env python3
"""Route near-duplicate intake images to drop/ or review/ conservatively.

The script compares each image with the immediately preceding image in its
dataset session. A dHash distance of 0-2 is a very strong duplicate signal and
is moved to drop. Distance 3-8 is visually similar but ambiguous, so it is
moved to review. All other images remain in in/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required. Install it with: python3 -m pip install opencv-python")


def dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    shared = project.parent / "raw_images"
    root = shared if shared.is_dir() else project / "raw_images"
    parser = argparse.ArgumentParser(description="Route visually redundant frames.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--session", default="d01", help="Filename prefix to process (default: d01).")
    parser.add_argument("--drop-distance", type=int, default=2)
    parser.add_argument("--review-distance", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Report decisions without moving files.")
    return parser.parse_args()


def main() -> None:
    options = args()
    if not 0 <= options.drop_distance < options.review_distance:
        sys.exit("Require: 0 <= --drop-distance < --review-distance")

    source = options.root / "in"
    drop = options.root / "drop"
    review = options.root / "review"
    meta = options.root / "meta"
    if not source.is_dir():
        sys.exit(f"Intake folder does not exist: {source}")
    images = sorted(source.glob(f"{options.session}_*.jpg"))
    if len(images) < 2:
        print(f"Need at least two matching images in {source}")
        return

    decisions: list[tuple[str, str, int, str]] = []
    previous_hash = dhash(images[0])
    previous_name = images[0].name
    for image in images[1:]:
        current_hash = dhash(image)
        score = distance(previous_hash, current_hash)
        destination = "keep-in"
        if score <= options.drop_distance:
            destination = "drop"
        elif score <= options.review_distance:
            destination = "review"
        decisions.append((image.name, previous_name, score, destination))
        previous_hash = current_hash
        previous_name = image.name

    counts = {name: sum(item[3] == name for item in decisions) for name in ("drop", "review", "keep-in")}
    print(f"Session {options.session}: {len(images)} input images")
    print(f"drop={counts['drop']}, review={counts['review']}, remain in={counts['keep-in'] + 1}")
    if options.dry_run:
        return

    drop.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    for name, _previous, _score, destination in decisions:
        if destination == "keep-in":
            continue
        image = source / name
        target = (drop if destination == "drop" else review) / name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        image.rename(target)
    log = meta / f"{options.session}_duplicate_routing.csv"
    with log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "compared_with", "dhash_distance", "decision"))
        writer.writerows(decisions)
    print(f"Moved files and wrote: {log}")


if __name__ == "__main__":
    main()
