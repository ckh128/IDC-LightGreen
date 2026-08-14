"""Paste these two cells into Colab to train the manually labelled D02-R1 model."""

# Cell 1: select d02_r1_manual.zip in the browser file picker.
# from google.colab import files
# uploaded = files.upload()

# Cell 2: train and download the complete result folder.
# !pip -q install ultralytics
# import shutil
# from pathlib import Path
# from ultralytics import YOLO
# from google.colab import files
# zip_name = next(name for name in uploaded if name.endswith('.zip'))
# shutil.unpack_archive(zip_name, '/content/d02_r1')
# dataset_root = next(Path('/content/d02_r1').rglob('data.yaml')).parent
# Path('/content/d02_r1_colab.yaml').write_text(
#     f"path: {dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n"
# )
# model = YOLO('yolo11n.pt')
# results = model.train(data='/content/d02_r1_colab.yaml', epochs=100, imgsz=640,
#                       batch=16, project='/content/runs/detect', name='d02_r1')
# run_dir = Path(results.save_dir)
# shutil.make_archive('/content/d02_r1_results', 'zip', run_dir)
# files.download('/content/d02_r1_results.zip')
