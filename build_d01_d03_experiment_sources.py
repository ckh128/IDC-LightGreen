"""Build the D03 final dataset and one Colab source archive for five experiments."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


def source_key(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"d03_(\d+)(?:_p(\d+))?\.txt", name)
    if not match:
        raise ValueError(f"Unexpected D03 CVAT label name: {name}")
    return int(match.group(1)), int(match.group(2) or 0)


def output_map(dataset: Path, prefix: str) -> dict[str, str]:
    rows = list(csv.DictReader((dataset / "image_source_manifest.csv").open(encoding="utf-8")))
    rows.sort(key=lambda row: source_key(row["cvat_label"]))
    mapping: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        output = f"{prefix}_{index:06d}.jpg"
        mapping[output] = Path(row["selected_image"]).name
    return mapping


def d03_records(dataset: Path, prefix: str) -> list[tuple[Path, Path, str, str]]:
    mapping = output_map(dataset, prefix)
    records: list[tuple[Path, Path, str, str]] = []
    for split in SPLITS:
        for image in (dataset / "images" / split).glob("*.jpg"):
            source = mapping.get(image.name)
            if source is None:
                raise RuntimeError(f"No source mapping for {image}")
            label = dataset / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            frame = source_key(source.replace(".jpg", ".txt"))[0]
            records.append((image, label, source, f"d03_{frame:06d}"))
    return records


def make_d03_final(project: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    r1 = d03_records(project / "datasets" / "d03_r1", "d03")
    r2 = d03_records(project / "datasets" / "d03_r2", "d03r2")
    groups: dict[str, list[tuple[Path, Path, str, str]]] = defaultdict(list)
    for record in r1 + r2:
        groups[record[3]].append(record)
    grouped = list(groups.values())
    random.Random(42).shuffle(grouped)
    targets = {split: len(r1 + r2) * ratio for split, ratio in RATIOS.items()}
    assigned = {split: [] for split in SPLITS}
    for group in grouped:
        split = max(SPLITS, key=lambda item: targets[item] - len(assigned[item]))
        assigned[split].extend(group)
    for split, records in assigned.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label, source, _group in records:
            new_stem = f"{image.parent.parent.parent.name}_{image.stem}"
            shutil.copy2(image, image_dir / f"{new_stem}.jpg")
            shutil.copy2(label, label_dir / f"{new_stem}.txt")
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )
    print("D03 final", " ".join(f"{split}={len(assigned[split])}" for split in SPLITS))


def copy_dataset(source: Path, destination: Path) -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            origin = source / kind / split
            target = destination / kind / split
            target.mkdir(parents=True, exist_ok=True)
            for path in origin.iterdir():
                shutil.copy2(path, target / path.name)
    (destination / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project_root
    d03_final = project / "datasets" / "d03_final"
    make_d03_final(project, d03_final, args.overwrite)

    staging = project / "exports" / "experiments" / "d01_d03_sources"
    archive = project / "exports" / "experiments" / "d01_d03_sources_colab.zip"
    if staging.exists():
        shutil.rmtree(staging)
    if archive.exists():
        archive.unlink()
    sources = {"d01": project / "datasets", "d02": project / "datasets" / "d02_r2", "d03": d03_final}
    for name, dataset in sources.items():
        copy_dataset(dataset, staging / name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in staging.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    print(archive)


if __name__ == "__main__":
    main()
