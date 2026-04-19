import nibabel as nib
import numpy as np
from skimage import measure
from pathlib import Path

DATA = Path(r"d:\Github\knee-implant-pipeline\data")
volume_name = "AB_72Y_Male_Left"

def analyze_components_after_ceiling():
    raw_volume = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    img = nib.load(str(raw_volume))
    data = img.get_fdata()
    
    # Use the Z-range from the successful fallback (Z < 924, Z > 200)
    # Applying the 350 - 1800 HU range
    mask = ((data > 350) & (data < 1800)).astype(np.uint8)
    
    # Crop to tibia search region
    mask[:, :, 924:] = 0
    mask[:, :, :200] = 0
    
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    
    print(f"Components found after 1800 HU ceiling: {len(props)}")
    props.sort(key=lambda x: x.area, reverse=True)
    
    for i, p in enumerate(props[:10]):
        z_len = p.bbox[5] - p.bbox[2]
        width = max(p.bbox[3]-p.bbox[0], p.bbox[4]-p.bbox[1])
        aspect = z_len / width if width > 0 else 0
        print(f"Component {i}: Area={p.area}, Centroid={p.centroid}, Z_Len={z_len}, Aspect={aspect:.2f}")

if __name__ == "__main__":
    analyze_components_after_ceiling()
