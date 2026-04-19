import nibabel as nib
import numpy as np
from skimage import measure
from pathlib import Path

DATA = Path(r"d:\Github\knee-implant-pipeline\data")
volume_name = "AB_72Y_Male_Left"

def analyze_tibia_region():
    raw_volume = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    img = nib.load(str(raw_volume))
    data = img.get_fdata()
    
    # Let's look at the region where the rod allegedly is
    # In the previous run, the tibia fallback selected Z-centroid ~607
    # The search range was below the femur (Z < ~924)
    
    # Threshold for bone-like intensity
    mask = (data > 350).astype(np.uint8)
    
    # Crop to the tibia search region (roughly)
    mask[:, :, 924:] = 0
    mask[:, :, :200] = 0 # remove feet
    
    labels = measure.label(mask)
    props = measure.regionprops(labels, intensity_image=data)
    
    print(f"Components in Tibia region (Z < 924): {len(props)}")
    props.sort(key=lambda x: x.area, reverse=True)
    
    for i, p in enumerate(props[:5]):
        mean_hu = p.mean_intensity
        max_hu = p.max_intensity
        z_len = p.bbox[5] - p.bbox[2]
        # Aspect ratio: Z length / max(X width, Y width)
        width = max(p.bbox[3]-p.bbox[0], p.bbox[4]-p.bbox[1])
        aspect_ratio = z_len / width if width > 0 else 0
        
        print(f"Component {i}: Area={p.area}, Mean_HU={mean_hu:.1f}, Max_HU={max_hu:.1f}, Z_Len={z_len:.1f}, Aspect={aspect_ratio:.2f}")

if __name__ == "__main__":
    analyze_tibia_region()
