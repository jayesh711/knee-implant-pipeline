import nibabel as nib
import numpy as np

path = r"d:\knee-implant-pipeline\knee-implant-pipeline\data\segmentations\phase1\AB_72_Male_Right_Knee.nii"
img = nib.load(path)
data = img.get_fdata()
unique_labels = np.unique(data)
print(f"Unique labels in segmentation: {unique_labels}")
