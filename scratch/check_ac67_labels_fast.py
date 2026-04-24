import nibabel as nib
import numpy as np

path = r"d:\knee-implant-pipeline\knee-implant-pipeline\data\segmentations\phase1\AC_67_male.nii"
img = nib.load(path)
proxy = img.dataobj
# Sample every 5th voxel in each dimension to save time
sample = proxy[::5, ::5, ::5]
unique_labels = np.unique(sample)
print(f"Unique labels in {path} (sampled): {unique_labels}")
