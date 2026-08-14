#!/usr/bin/env python3
"""Move the later portion of a D02 frame session into D03 without losing time mapping."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
IN_PATTERN = re.compile(r"d02_(\d+)\.jpg")
PROBE_PATTERN = re.compile(r"d02_(\d+)_p([13])\.jpg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT / "raw_images")
    parser.add_argument("--split-after", type=int, default=1300)
    args = parser.parse_args()
    intake = args.root / "in"
    d02_probe, d03_probe = args.root / "probe" / "d02", args.root / "probe" / "d03"
    meta = args.root / "meta"
    d03_probe.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    changes: list[tuple[str, str, str, int]] = []

    for path in sorted(intake.glob("d02_*.jpg")):
        match = IN_PATTERN.fullmatch(path.name)
        if not match or int(match.group(1)) <= args.split_after:
            continue
        sequence = int(match.group(1)) - args.split_after
        target = intake / f"d03_{sequence:06d}.jpg"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        path.rename(target)
        changes.append(("in", path.name, target.name, sequence))

    for path in sorted(d02_probe.glob("d02_*_p*.jpg")):
        match = PROBE_PATTERN.fullmatch(path.name)
        if not match or int(match.group(1)) <= args.split_after:
            continue
        sequence, offset = int(match.group(1)) - args.split_after, match.group(2)
        target = d03_probe / f"d03_{sequence:06d}_p{offset}.jpg"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        path.rename(target)
        changes.append(("probe", path.name, target.name, sequence))

    log = meta / f"d02_to_d03_split_{args.split_after:06d}.csv"
    with log.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("kind", "old_name", "new_name", "d03_source_sequence"))
        writer.writerows(changes)
    print(f"Moved {sum(row[0] == 'in' for row in changes)} intake and {sum(row[0] == 'probe' for row in changes)} probe images")
    print(f"Mapping: {log}")


if __name__ == "__main__":
    main()
