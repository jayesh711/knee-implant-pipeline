import nibabel as nib
import numpy as np

patient = "AB_72_Y_Male-Right_Knee"
seg_path = f"data/segmentations/phase1/{patient}.nii"
img = nib.load(seg_path)
data = img.get_fdata()

for fid in [25, 76]:
    mask = (data == fid)
    coords = np.argwhere(mask)
    if coords.size > 0:
        print(f"Femur Label {fid} Z-range: {coords[:, 2].min()} to {coords[:, 2].max()} (mean {coords[:, 2].mean():.1f})")
