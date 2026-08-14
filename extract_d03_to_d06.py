"""Extract newly added recordings into balanced D03-D06 frame batches."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


ROUND_IDS = ("d03", "d04", "d05", "d06")
OFFSETS = {0: "5n", 1: "p1", 2: "p2", 3: "p3", 4: "p4"}


def frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise RuntimeError(f"Cannot read frame count: {video}")
    return count


def make_assignments(videos: list[Path]) -> list[tuple[Path, int]]:
    groups: list[tuple[Path, int]] = []
    for video in videos:
        groups.extend((video, group_index) for group_index in range(frame_count(video) // 5))
    return groups


def extract(videos: list[Path], output_root: Path, manifest_path: Path) -> None:
    groups = make_assignments(videos)
    if not groups:
        raise RuntimeError("No complete 5-frame groups found")

    for round_id in ROUND_IDS:
        for offset_name in OFFSETS.values():
            (output_root / round_id / offset_name).mkdir(parents=True, exist_ok=True)

    base, remainder = divmod(len(groups), len(ROUND_IDS))
    round_for_group: list[str] = []
    for index, round_id in enumerate(ROUND_IDS):
        round_for_group.extend([round_id] * (base + (1 if index < remainder else 0)))

    counters: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, str | int]] = []
    current_video: Path | None = None
    capture: cv2.VideoCapture | None = None

    try:
        for global_index, ((video, group_index), round_id) in enumerate(zip(groups, round_for_group), start=1):
            if video != current_video:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(video))
                current_video = video
            counters[round_id] += 1
            sample_index = counters[round_id]
            capture.set(cv2.CAP_PROP_POS_FRAMES, group_index * 5)
            for offset, offset_name in OFFSETS.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read {video.name}, frame {group_index * 5 + offset + 1}")
                suffix = "" if offset == 0 else f"_{offset_name}"
                filename = f"{round_id}_{sample_index:06d}{suffix}.jpg"
                destination = output_root / round_id / offset_name / filename
                if not cv2.imwrite(str(destination), frame):
                    raise RuntimeError(f"Cannot write {destination}")
                rows.append({
                    "round": round_id,
                    "sample_index": sample_index,
                    "category": offset_name,
                    "image": str(destination.relative_to(output_root.parent)),
                    "source_video": video.name,
                    "source_frame_1based": group_index * 5 + offset + 1,
                    "global_group": global_index,
                })
    finally:
        if capture is not None:
            capture.release()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Complete groups: {len(groups)}")
    for round_id in ROUND_IDS:
        print(f"{round_id}: {counters[round_id]} samples, {counters[round_id] * 5} images")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_root = args.project_root / "raw_videos"
    output_root = args.project_root / "raw_images"
    manifest_path = output_root / "meta" / "d03_to_d06_manifest.csv"
    videos = sorted(video_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9].mp4"), key=lambda item: item.name)
    if not videos:
        raise FileNotFoundError(f"No newly numbered MP4s in {video_root}")

    targets = [output_root / round_id for round_id in ROUND_IDS]
    existing = [target for target in targets if target.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Targets already exist: " + ", ".join(map(str, existing)))
    for target in existing:
        shutil.rmtree(target)

    print("Videos:")
    for video in videos:
        print(f"  {video.name}: {frame_count(video)} frames")
    extract(videos, output_root, manifest_path)


if __name__ == "__main__":
    main()
