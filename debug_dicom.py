import SimpleITK as sitk
import os
from pathlib import Path

target_dir = r"d:\Github\knee-implant-pipeline\data\inputs\CT Scan data for Ethosh\AB 72 Y Male-Left Knee\A20250206171445\P00001\S0001"
reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(target_dir)
print(f"Series IDs found: {series_ids}")

if series_ids:
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(target_dir, sid)
        print(f"Series {sid} has {len(files)} files.")
else:
    print("No Series IDs found.")
    # Check if files exist at all
    print(f"Files in dir: {os.listdir(target_dir)[:10]}")
