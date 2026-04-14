import nibabel as nib
from pathlib import Path

ct_path = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data\NIfTI\AB_72_Y_Male-Right_Knee_raw.nii.gz")
if ct_path.exists():
    img = nib.load(str(ct_path))
    print(f"Shape: {img.shape}")
    print(f"Affine:\n{img.affine}")
    print(f"Zooms: {img.header.get_zooms()}")
