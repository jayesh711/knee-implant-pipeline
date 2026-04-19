import nibabel as nib
import numpy as np
from skimage import measure
from pathlib import Path

DATA = Path(r"d:\Github\knee-implant-pipeline\data")
volume_name = "AB_72Y_Male_Left"
HU_METAL_MIN = 3000

def debug_fallback():
    raw_volume = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    img = nib.load(str(raw_volume))
    affine = img.affine
    
    ds_factor = 2
    raw_ds = np.asarray(img.dataobj[::ds_factor, ::ds_factor, ::ds_factor])
    mask_ds = ((raw_ds > 250) & (raw_ds < HU_METAL_MIN)).astype(np.uint8)
    
    labels_ds = measure.label(mask_ds)
    props = measure.regionprops(labels_ds)
    
    print(f"Total components found: {len(props)}")
    props.sort(key=lambda x: x.area, reverse=True)
    
    for i, p in enumerate(props[:10]):
        z_min, z_max = p.bbox[2], p.bbox[5]
        z_len = (z_max - z_min) * ds_factor
        centroid_z = p.centroid[2] * ds_factor
        print(f"Component {i}: Area={p.area}, Z_Centroid={centroid_z:.1f}, Z_Range=[{z_min*ds_factor}, {z_max*ds_factor}], Vertical_Len={z_len}")

if __name__ == "__main__":
    debug_fallback()
