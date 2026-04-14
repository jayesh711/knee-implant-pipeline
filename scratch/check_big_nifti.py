import nibabel as nib
import numpy as np
from pathlib import Path

seg_path = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data\NIfTI\AB_72_Y_Male-Right_Knee.nii.gz")
if seg_path.exists():
    img = nib.load(str(seg_path))
    data = img.get_fdata()
    unique_labels = np.unique(data)
    print(f"Unique labels in Big NIfTI: {unique_labels}")
else:
    print("Big NIfTI file not found.")
