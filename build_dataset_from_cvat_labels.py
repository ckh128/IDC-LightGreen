#!/usr/bin/env python3
"""Build a YOLO dataset using CVAT-exported labels and locally stored images.

CVAT may export only annotation TXT files. For every label, this script searches
the project for an image with the identical stem. If more than one match exists,
the most recently modified image is selected. Output filenames are standardized
as d01_000001.jpg (and matching .txt files).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Name must end with a frame number: {path.name}")
    return int(match.group(1))


def contiguous_groups(items: list[tuple[Path, Path]]) -> list[list[tuple[Path, Path]]]:
    groups: list[list[tuple[Path, Path]]] = []
    for item in sorted(items, key=lambda pair: frame_number(pair[0])):
        if not groups or frame_number(item[0]) != frame_number(groups[-1][-1][0]) + 1:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def assign(groups: list[list[tuple[Path, Path]]], count: int) -> dict[str, list[tuple[Path, Path]]]:
    random.Random(42).shuffle(groups)
    targets = {name: count * ratio for name, ratio in RATIOS.items()}
    result = {name: [] for name in SPLITS}
    for group in groups:
        destination = max(SPLITS, key=lambda name: targets[name] - len(result[name]))
        result[destination].extend(group)
    return result


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Pair CVAT labels with local images and split a YOLO dataset.")
    parser.add_argument("--labels-zip", type=Path, default=project / "raw_data" / "d01_raw_data_001.zip")
    parser.add_argument("--output", type=Path, default=project / "processed_data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.labels_zip.is_file():
        sys.exit(f"CVAT ZIP not found: {args.labels_zip}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        sys.exit(f"Output exists: {args.output}. Use --overwrite to replace it.")

    with tempfile.TemporaryDirectory(prefix="cvat_labels_") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(args.labels_zip) as archive:
            archive.extractall(extracted)
        labels = sorted((extracted / "labels").rglob("*.txt"))
        if not labels:
            sys.exit("No label TXT files found in CVAT ZIP.")

        index: dict[str, list[Path]] = {}
        ignored = {args.output.resolve(), extracted.resolve()}
        for image in project.rglob("*"):
            if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
                continue
            if any(parent in ignored for parent in image.parents):
                continue
            index.setdefault(image.stem, []).append(image)

        pairs: list[tuple[Path, Path]] = []
        missing: list[str] = []
        choices: list[tuple[str, str, int]] = []
        for label in labels:
            candidates = index.get(label.stem, [])
            if not candidates:
                missing.append(label.name)
                continue
            image = max(candidates, key=lambda path: path.stat().st_mtime)
            pairs.append((label, image))
            choices.append((label.name, str(image), len(candidates)))

        if missing:
            (project / "raw_data" / "d01_missing_images.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
            sys.exit(f"Missing {len(missing)} image(s). See raw_data/d01_missing_images.txt")
        if args.output.exists():
            shutil.rmtree(args.output)

        numbered = {label: f"d01_{number:06d}.jpg" for number, (label, _image) in enumerate(sorted(pairs, key=lambda pair: frame_number(pair[0])), start=1)}
        dataset = assign(contiguous_groups(pairs), len(pairs))
        for split, items in dataset.items():
            image_dir, label_dir = args.output / "images" / split, args.output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for label, image in items:
                output_image = numbered[label]
                shutil.copy2(image, image_dir / output_image)
                shutil.copy2(label, label_dir / f"{Path(output_image).stem}.txt")
        (args.output / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: human\n  1: robot\n", encoding="utf-8"
        )
        with (args.output / "image_source_manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("cvat_label", "selected_image", "same_name_candidates"))
            writer.writerows(choices)
        print(f"Built {len(pairs)} labelled pairs from CVAT labels.")
        for split in SPLITS:
            print(f"  {split}: {len(dataset[split])}")


if __name__ == "__main__":
    main()
