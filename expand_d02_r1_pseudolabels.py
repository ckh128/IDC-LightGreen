#!/usr/bin/env python3
"""After best_d02_r1.pt is supplied, prepare D02-R2 pseudo-label candidates.

For every curated D02 5n image still in raw_images/in:
  - extract the matching 5n+2 and 5n+4 frames from the original videos;
  - predict YOLO labels and retain only non-empty, high-confidence candidates;
  - render 5n+1 and 5n+3 as probe images, never as training labels.
The output is a *candidate* dataset. Review a sample before R2 training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parent.parent


def extract_frames(videos: list[Path], groups: set[int], offsets: set[int], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    sequence, paths = 1, []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            group, offset = sequence + frame_index // 5, frame_index % 5
            if group in groups and offset in offsets:
                path = destination / f"d02_{group:06d}_p{offset}.jpg"
                if not path.exists() and not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Cannot write {path}")
                paths.append(path)
            frame_index += 1
        capture.release()
        sequence += (frame_index + 4) // 5
    return sorted(paths)


def write_yolo(path: Path, boxes: object, width: int, height: int) -> None:
    lines = []
    for cls, xyxy in zip(boxes.cls.tolist(), boxes.xyxy.tolist(), strict=True):
        x1, y1, x2, y2 = xyxy
        lines.append(f"{int(cls)} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=PROJECT / "models" / "d02_r1" / "best_d02_r1.pt")
    parser.add_argument("--images-root", type=Path, default=PROJECT / "raw_images")
    parser.add_argument("--videos", type=Path, default=PROJECT / "raw_videos")
    parser.add_argument("--output", type=Path, default=PROJECT / "auto_labeling" / "d02_r2_candidates")
    parser.add_argument("--confidence", type=float, default=0.80)
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"D02-R1 weights not found: {args.weights}")
    source = sorted((args.images_root / "in").glob("d02_*.jpg"))
    groups = {int(path.stem.rsplit("_", 1)[1]) for path in source}
    if not groups:
        raise SystemExit("No curated D02 5n images in raw_images/in")
    output = args.output
    p2p4 = extract_frames(sorted(args.videos.glob("d02_*")), groups, {2, 4}, output / "source_5n_plus_2_4")
    probe_source = args.images_root / "probe" / "d02"
    probes = [path for path in probe_source.glob("d02_*_p*.jpg") if int(path.stem.split("_")[1]) in groups]
    model = YOLO(str(args.weights))
    pseudo_images, pseudo_labels = output / "pseudo" / "images", output / "pseudo" / "labels"
    probe_render = output / "probe_detections"
    pseudo_images.mkdir(parents=True, exist_ok=True); pseudo_labels.mkdir(parents=True, exist_ok=True); probe_render.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in p2p4:
        result = model.predict(source=str(path), conf=0.25, verbose=False)[0]
        boxes = result.boxes
        best = 0.0 if boxes is None or not len(boxes) else max(boxes.conf.tolist())
        keep = boxes is not None and len(boxes) and min(boxes.conf.tolist()) >= args.confidence
        if keep:
            height, width = result.orig_shape
            cv2.imwrite(str(pseudo_images / path.name), result.orig_img)
            write_yolo(pseudo_labels / f"{path.stem}.txt", boxes, width, height)
        rows.append((path.name, "pseudo_keep" if keep else "review_or_exclude", f"{best:.6f}", 0 if boxes is None else len(boxes)))
    for path in sorted(probes):
        result = model.predict(source=str(path), conf=0.25, verbose=False)[0]
        cv2.imwrite(str(probe_render / path.name), result.plot())
    with (output / "pseudo_manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(("image", "route", "best_confidence", "box_count")); writer.writerows(rows)
    (output / "pseudo" / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames:\n  0: human\n  1: robot\n", encoding="utf-8")
    print(f"curated_5n={len(groups)} pseudo_source={len(p2p4)} pseudo_kept={sum(row[1]=='pseudo_keep' for row in rows)} probes={len(probes)}")


if __name__ == "__main__":
    main()
