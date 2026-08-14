"""D02-R2 Colab training cells."""

# !pip -q install ultralytics
# import shutil
# import zipfile
# from pathlib import Path
# from ultralytics import YOLO
# from google.colab import files
# uploaded = files.upload()
# zip_path = next(Path('/content').glob('*.zip'))
# extract_dir = Path('/content/d02_r2_data')
# shutil.unpack_archive(zip_path, extract_dir)
# dataset_root = next(extract_dir.rglob('data.yaml')).parent
# colab_yaml = Path('/content/d02_r2_colab.yaml')
# colab_yaml.write_text(f"path: {dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n")
# model = YOLO('yolo11n.pt')
# results = model.train(data=str(colab_yaml), epochs=100, imgsz=640, batch=16, device=0,
#                       project='/content/runs/detect', name='d02_r2', exist_ok=True)
# run_dir = Path(results.save_dir)
# shutil.make_archive('/content/d02_r2_results', 'zip', run_dir)
# files.download('/content/d02_r2_results.zip')
