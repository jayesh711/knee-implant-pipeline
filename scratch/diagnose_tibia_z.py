import nibabel as nib
import numpy as np

patient = "AB_72_Y_Male-Right_Knee"
seg_path = f"data/segmentations/phase1/{patient}.nii"

print(f"Loading {seg_path}...")
img = nib.load(seg_path)
data = img.get_fdata()

tibia_candidates = [2, 26, 81] 
for tid in tibia_candidates:
    mask = (data == tid)
    coords = np.argwhere(mask)
    if coords.size > 0:
        z_min = coords[:, 2].min()
        z_max = coords[:, 2].max()
        z_mean = coords[:, 2].mean()
        print(f"Label {tid}: Z-range {z_min} to {z_max} (mean {z_mean:.1f}), Count: {coords.shape[0]}")
        
    else:
        print(f"Label {tid}: Not found")
