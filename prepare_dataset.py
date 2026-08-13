#!/usr/bin/env python3
"""Create a leak-resistant YOLO train/val/test dataset from labelled pairs.

Input:  ../data/images/{train,val} and ../data/labels/{train,val}
Output: ../processed_data/{images,labels}/{train,val,test} plus data.yaml

Adjacent numbered source frames stay together in one split, preventing near
identical video frames from leaking from training into validation/test.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


def contiguous_groups(pairs: list[tuple[Path, Path]]) -> list[list[tuple[Path, Path]]]:
    """Group frames whose numeric filename portions are consecutive."""
    ordered = sorted(pairs, key=lambda pair: int(re.search(r"(\d+)$", pair[0].stem).group(1)))
    groups: list[list[tuple[Path, Path]]] = []
    for pair in ordered:
        number = int(re.search(r"(\d+)$", pair[0].stem).group(1))
        if not groups:
            groups.append([pair])
            continue
        previous = int(re.search(r"(\d+)$", groups[-1][-1][0].stem).group(1))
        if number == previous + 1:
            groups[-1].append(pair)
        else:
            groups.append([pair])
    return groups


def collect_pairs(source: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image in source.glob("images/*/*"):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = source / "labels" / image.parent.name / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}: {label}")
        if not re.search(r"\d+$", image.stem):
            raise ValueError(f"Image name must end with a frame number: {image.name}")
        pairs.append((image, label))
    if not pairs:
        raise FileNotFoundError(f"No labelled image pairs under {source}")
    return pairs


def assign_groups(groups: list[list[tuple[Path, Path]]], total: int) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(42)
    rng.shuffle(groups)
    target = {split: total * ratio for split, ratio in RATIOS.items()}
    assigned = {split: [] for split in SPLITS}
    for group in groups:
        split = max(SPLITS, key=lambda name: target[name] - len(assigned[name]))
        assigned[split].extend(group)
    return assigned


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits.")
    parser.add_argument("--source", type=Path, default=project / "data")
    parser.add_argument("--output", type=Path, default=project / "processed_data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(f"Output exists and is not empty: {args.output}. Use --overwrite to replace it.")
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)

    pairs = collect_pairs(args.source)
    ordered_pairs = sorted(pairs, key=lambda pair: int(re.search(r"(\d+)$", pair[0].stem).group(1)))
    output_names = {image: f"d01_{index:06d}{image.suffix.lower()}" for index, (image, _label) in enumerate(ordered_pairs, start=1)}
    grouped = contiguous_groups(pairs)
    assigned = assign_groups(grouped, len(pairs))
    for split, items in assigned.items():
        for kind in ("images", "labels"):
            (args.output / kind / split).mkdir(parents=True, exist_ok=True)
        for image, label in items:
            output_image = output_names[image]
            shutil.copy2(image, args.output / "images" / split / output_image)
            shutil.copy2(label, args.output / "labels" / split / f"{Path(output_image).stem}.txt")

    (args.output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )
    print("Dataset created:")
    for split in SPLITS:
        print(f"  {split}: {len(assigned[split])} images")


if __name__ == "__main__":
    main()
