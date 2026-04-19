import nibabel as nib
import numpy as np
from pathlib import Path

seg_path = r"d:\Github\knee-implant-pipeline\data\segmentations\phase1\AB_72Y_Male_Left.nii"
img = nib.load(seg_path)
data = img.get_fdata()
unique_labels = np.unique(data)
spacing = img.header.get_zooms()
voxel_vol = np.prod(spacing) / 1000.0 # cc

print(f"Segmentation: {seg_path}")
print(f"Shape: {data.shape}")
print(f"Spacing: {spacing}")
print("-" * 30)
print(f"{'LabelID':<10} | {'Volume (cc)':<12}")
print("-" * 30)

for lab in unique_labels:
    if lab == 0: continue
    vol = np.sum(data == lab) * voxel_vol
    print(f"{int(lab):<10} | {vol:<12.2f}")
