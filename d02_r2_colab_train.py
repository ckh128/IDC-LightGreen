"""Run after manually uploading d02_r2_manual_corrected.zip to Colab's /content sidebar."""

# !pip -q install ultralytics
# import shutil
# import zipfile
# from pathlib import Path
# from ultralytics import YOLO
# from google.colab import files
#
# zip_files = sorted(Path('/content').glob('*.zip'), key=lambda path: path.stat().st_mtime, reverse=True)
# dataset_zip = next(
#     path for path in zip_files
#     if any(name == 'data.yaml' or name.endswith('/data.yaml') for name in zipfile.ZipFile(path).namelist())
# )
# extract_dir = Path('/content/d02_r2_data')
# if extract_dir.exists():
#     shutil.rmtree(extract_dir)
# shutil.unpack_archive(dataset_zip, extract_dir)
# dataset_root = next(extract_dir.rglob('data.yaml')).parent
# colab_yaml = Path('/content/d02_r2_colab.yaml')
# colab_yaml.write_text(f"path: {dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: human\n  1: robot\n")
# model = YOLO('yolo11n.pt')
# results = model.train(data=str(colab_yaml), epochs=100, imgsz=640, batch=16, device=0,
#                       project='/content/runs/detect', name='d02_r2', exist_ok=True)
# run_dir = Path(results.save_dir)
# shutil.make_archive('/content/d02_r2_results', 'zip', run_dir)
# files.download('/content/d02_r2_results.zip')
