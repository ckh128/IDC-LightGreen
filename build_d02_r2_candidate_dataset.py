#!/usr/bin/env python3
"""Prepare a D02-R2 training candidate from manual R1 labels plus pseudo-labels.

Pseudo-labels are added only to train/.  Manually labelled R1 val/test images
remain the evaluation sets, so pseudo-labels never become validation truth.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent


def copy_pair(image: Path, label: Path, output: Path, split: str) -> None:
    image_dir, label_dir = output / "images" / split, output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / image.name)
    shutil.copy2(label, label_dir / label.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, default=PROJECT / "datasets" / "d02_r1")
    parser.add_argument("--pseudo", type=Path, default=PROJECT / "auto_labeling" / "d02_r2_candidates" / "pseudo")
    parser.add_argument("--output", type=Path, default=PROJECT / "datasets" / "d02_r2_candidate")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    rows = []
    for split in ("train", "val", "test"):
        for image in sorted((args.manual / "images" / split).glob("*.jpg")):
            label = args.manual / "labels" / split / f"{image.stem}.txt"
            copy_pair(image, label, args.output, split)
            rows.append((image.name, split, "manual"))
    for image in sorted((args.pseudo / "images").glob("*.jpg")):
        label = args.pseudo / "labels" / f"{image.stem}.txt"
        copy_pair(image, label, args.output, "train")
        rows.append((image.name, "train", "pseudo"))
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(("image", "split", "label_source")); writer.writerows(rows)
    print(f"manual={sum(row[2]=='manual' for row in rows)} pseudo_train={sum(row[2]=='pseudo' for row in rows)} train={sum(row[1]=='train' for row in rows)} val={sum(row[1]=='val' for row in rows)} test={sum(row[1]=='test' for row in rows)}")


if __name__ == "__main__":
    main()
