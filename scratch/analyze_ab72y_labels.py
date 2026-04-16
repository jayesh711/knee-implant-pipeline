import nibabel as nib
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data")

def analyze_labels(volume_name):
    seg_file = DATA_DIR / "segmentations" / "phase1" / f"{volume_name}.nii"
    if not seg_file.exists():
        seg_file = DATA_DIR / "segmentations" / "phase1" / f"{volume_name}.nii.gz"
    
    if not seg_file.exists():
        print(f"File not found: {seg_file}")
        return

    print(f"Analyzing: {seg_file}")
    img = nib.load(str(seg_file))
    data = img.get_fdata()
    
    zooms = img.header.get_zooms()
    voxel_vol = np.prod(zooms) / 1000.0 # in cc
    
    labels, counts = np.unique(data, return_counts=True)
    
    print(f"{'LabelID':<10} | {'Count':<10} | {'Volume (cc)':<10}")
    print("-" * 35)
    for label, count in zip(labels, counts):
        if label == 0: continue
        vol = count * voxel_vol
        print(f"{int(label):<10} | {count:<10} | {vol:<10.2f}")

if __name__ == "__main__":
    analyze_labels("AB_72_Y_Male-Right_Knee")
