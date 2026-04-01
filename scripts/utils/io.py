import SimpleITK as sitk
import numpy as np
import nibabel as nib
from pathlib import Path

def load_dicom_series(dicom_dir: str):
    """Load a DICOM folder and return (sitk_image, np_array, spacing)."""
    reader = sitk.ImageSeriesReader()
    names  = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(names)
    img    = reader.Execute()
    arr    = sitk.GetArrayFromImage(img)   # (z, y, x)
    return img, arr, img.GetSpacing()      # spacing = (sx, sy, sz) mm

def load_nifti(path: str):
    img     = sitk.ReadImage(str(path))
    arr     = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return img, arr, spacing

def save_nifti(arr: np.ndarray, ref_img, out_path: str):
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(out_path))
    print(f"Saved: {out_path}")
