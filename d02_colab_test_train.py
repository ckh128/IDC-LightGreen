"""Paste these cells into Google Colab to train the unreviewed D02 test model."""

# Cell 1: upload d02_pseudolabel_test.zip in the Colab file picker.
# from google.colab import files
# uploaded = files.upload()

# Cell 2: unpack and train.  This makes labels.jpg, train_batch*.jpg,
# results.png, weights/best.pt, and a downloadable result ZIP.
# !pip -q install ultralytics
# import shutil
# from pathlib import Path
# from ultralytics import YOLO
#
# zip_name = next(name for name in uploaded if name.endswith('.zip'))
# shutil.unpack_archive(zip_name, '/content/d02_pseudolabel_test')
# model = YOLO('yolo11n.pt')
# results = model.train(
#     data='/content/d02_pseudolabel_test/data.yaml',
#     epochs=30, imgsz=640, batch=16,
#     project='/content/runs/detect', name='d02_pseudolabel_test',
# )
# run_dir = Path(results.save_dir)
# print('best model:', run_dir / 'weights' / 'best.pt')
# shutil.make_archive('/content/d02_pseudolabel_test_results', 'zip', run_dir)
# files.download('/content/d02_pseudolabel_test_results.zip')
