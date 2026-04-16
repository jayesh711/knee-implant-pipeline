import nibabel as nib
import numpy as np
from pathlib import Path
from config import DATA

def analyze_tibia_artifacts(volume_name):
    raw_vol_path = DATA / "NIfTI" / f"{volume_name}_prepped.nii.gz"
    if not raw_vol_path.exists():
        raw_vol_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    
    img = nib.load(str(raw_vol_path))
    data = img.get_fdata()
    
    # We know the Tibia is at Z < -568 (World) or index < 432 (ds=2)
    # Let's look at the density distribution in the tibia region
    # From previous check: Tibia centroid was 310.9 (ds=2)
    
    ds = 2
    tibia_data = data[::ds, ::ds, :432] # Bottom half approx
    
    print(f"Analyzing HU distribution in Tibia region (ds=2)...")
    
    # Check for high density regions that might be the 'back rod'
    # Bone is ~250-1000. Rod is likely higher.
    for h in [1000, 1200, 1500, 1800, 2000]:
        count = np.sum(tibia_data > h)
        print(f"  Voxels > {h} HU: {count}")
    
    # Let's find the max local density in the tibia region
    print(f"  Max HU in Tibia region: {np.max(tibia_data)}")

if __name__ == "__main__":
    analyze_tibia_artifacts("AB_72_Y_Male-Right_Knee")
