#!/usr/bin/env python3
"""Extract only D01's held-out 5n+1 and 5n+3 probe frames.

This intentionally never touches raw_images/in.  It is useful for bringing
older D01 videos into the probe workflow after their intake frames exist.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2


PROJECT = Path(__file__).resolve().parent.parent
VIDEOS = PROJECT / "raw_videos"
PROBE = PROJECT / "raw_images" / "probe"
MANIFEST = PROJECT / "raw_images" / "meta" / "d01_probe_manifest.csv"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".h264"}


def main() -> None:
    PROBE.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    sequence = 1
    created = 0
    rows: list[tuple[str, str, int, int]] = []

    for video in sorted(VIDEOS.iterdir()):
        if video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            offset = frame_index % 5
            if offset in (1, 3):
                filename = f"d01_{sequence + frame_index // 5:06d}_p{offset}.jpg"
                destination = PROBE / filename
                if not destination.exists():
                    if not cv2.imwrite(str(destination), frame):
                        raise RuntimeError(f"Cannot write {destination}")
                    created += 1
                rows.append((filename, video.name, frame_index, offset))
            frame_index += 1
        capture.release()
        sequence += (frame_index + 4) // 5
        print(f"{video.name}: {frame_index} frames processed")

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("probe_image", "source_video", "source_frame", "offset"))
        writer.writerows(rows)
    print(f"Created {created} probe images; {len(rows)} total. Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
