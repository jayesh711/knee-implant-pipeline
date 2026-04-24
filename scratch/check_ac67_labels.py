import nibabel as nib
import numpy as np
from pathlib import Path

path = r"d:\knee-implant-pipeline\knee-implant-pipeline\data\segmentations\phase1\AC_67_male.nii"
img = nib.load(path)
data = img.get_fdata()
unique_labels = np.unique(data)
print(f"Unique labels in {path}: {unique_labels}")
