import nibabel as nib
import numpy as np
from skimage import measure, morphology
from pathlib import Path
from config import DATA, HU_METAL_MIN

def test_separation(volume_name):
    raw_vol_path = DATA / "NIfTI" / f"{volume_name}_prepped.nii.gz"
    if not raw_vol_path.exists():
        raw_vol_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    
    img = nib.load(str(raw_vol_path))
    data = img.get_fdata()
    
    # Try different metal thresholds
    thresholds = [2000, 2500, 3000]
    
    print(f"Testing component separation for {volume_name}...")
    
    # Downsample for speed
    ds = 2
    data_ds = data[::ds, ::ds, ::ds]
    
    for t in thresholds:
        print(f"\nMetal Threshold: {t}")
        # Bone mask: 250 < HU < t
        mask = ((data_ds > 250) & (data_ds < t)).astype(np.uint8)
        
        # Method 1: Raw connectivity
        labels = measure.label(mask)
        print(f"  Raw components: {labels.max()}")
        
        # Method 2: Erosion (to break rod bridges)
        eroded = morphology.binary_erosion(mask, morphology.ball(1))
        labels_eroded = measure.label(eroded)
        print(f"  Eroded components: {labels_eroded.max()}")
        
        if labels_eroded.max() > 0:
            # Check size of largest component
            counts = np.bincount(labels_eroded.flat)
            if len(counts) > 1:
                print(f"  Largest component size: {np.max(counts[1:])}")

if __name__ == "__main__":
    test_separation("AB_72_Y_Male-Right_Knee")
