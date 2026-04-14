import nibabel as nib
import numpy as np
from pathlib import Path

seg_path = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data\segmentations\phase1\AB_72_Y_Male-Right_Knee.nii")
if seg_path.exists():
    img = nib.load(str(seg_path))
    data = img.get_fdata()
    unique_labels = np.unique(data)
    print(f"Unique labels in segmentation: {unique_labels}")
    for label in unique_labels:
        count = np.sum(data == label)
        print(f"Label {label}: {count} voxels")
else:
    print("Segmentation file not found.")
