import nibabel as nib
import numpy as np
from skimage import measure
from pathlib import Path

DATA = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data")
volume_name = "AB_72_Y_Male-Right_Knee"
raw_volume = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"

if raw_volume.exists():
    img = nib.load(str(raw_volume))
    ds_factor = 2
    # Load 2x downsampled for speed
    print("Loading downsampled volume...")
    mask_ds = (img.dataobj[::ds_factor, ::ds_factor, ::ds_factor] > 250).astype(np.uint8)
    
    print("Labeling components...")
    labels_ds = measure.label(mask_ds)
    props = measure.regionprops(labels_ds)
    
    bone_props = [p for p in props if p.area > 5000]
    bone_props.sort(key=lambda x: x.centroid[2], reverse=True)
    
    print(f"Found {len(bone_props)} major bone structures:")
    for i, p in enumerate(bone_props):
        z_vox = p.centroid[2] * ds_factor
        z_world = img.affine[2, 2] * z_vox + img.affine[2, 3]
        print(f"  Component {i}: Area={p.area}, Z-centroid (voxel)={z_vox:.1f}, Z-world={z_world:.1f}")
else:
    print("Raw volume not found.")
