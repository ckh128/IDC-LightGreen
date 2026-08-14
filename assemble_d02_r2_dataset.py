#!/usr/bin/env python3
"""Combine D02-R1 manual labels with D02-R2 CVAT-corrected labels."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent


def copy_dataset(source: Path, label_source: str, output: Path, rows: list[tuple[str, str, str]]) -> None:
    for split in ("train", "val", "test"):
        for image in sorted((source / "images" / split).glob("*.jpg")):
            label = source / "labels" / split / f"{image.stem}.txt"
            target_image, target_label = output / "images" / split / image.name, output / "labels" / split / label.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if target_image.exists() or target_label.exists():
                raise FileExistsError(f"Filename collision: {image.name}")
            shutil.copy2(image, target_image)
            shutil.copy2(label, target_label)
            rows.append((image.name, split, label_source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, default=PROJECT / "datasets" / "d02_r1")
    parser.add_argument("--corrected", type=Path, default=PROJECT / "datasets" / "d02_r2_cvat")
    parser.add_argument("--output", type=Path, default=PROJECT / "datasets" / "d02_r2")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    rows: list[tuple[str, str, str]] = []
    copy_dataset(args.manual, "manual_r1", args.output, rows)
    copy_dataset(args.corrected, "cvat_corrected_r2", args.output, rows)
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(("image", "split", "label_source")); writer.writerows(rows)
    print(" ".join(f"{split}={sum(row[1] == split for row in rows)}" for split in ("train", "val", "test")))


if __name__ == "__main__":
    main()
