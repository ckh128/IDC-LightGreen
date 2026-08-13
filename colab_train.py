"""Google Colab training cell source.

In Colab, first upload or mount Drive containing processed_data, then set
DATASET below and run this file as notebook cells (or copy its commands).
"""

# !pip install -q ultralytics
# from ultralytics import YOLO
# DATASET = "/content/drive/MyDrive/idc-cv/processed_data/data.yaml"
# model = YOLO("yolo11n.pt")
# results = model.train(data=DATASET, epochs=100, imgsz=640, batch=16,
#                       project="/content/drive/MyDrive/idc-cv/runs", name="d01")
# print(results.save_dir / "weights" / "best.pt")
