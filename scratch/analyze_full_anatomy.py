import nibabel as nib
import numpy as np
from skimage import measure

patient = "AB_72_Y_Male-Right_Knee"
seg_path = f"data/segmentations/phase1/{patient}.nii"

print(f"Loading {seg_path}...")
img = nib.load(seg_path)
data = img.get_fdata()

# Get all labels
unique_labels = np.unique(data).astype(int)
print(f"Labels found: {unique_labels}")

# Analyze each label's Z-centroid and Area
results = []
for label_id in unique_labels:
    if label_id == 0: continue
    mask = (data == label_id)
    coords = np.argwhere(mask)
    if coords.size > 0:
        z_mean = coords[:, 2].mean()
        z_min = coords[:, 2].min()
        z_max = coords[:, 2].max()
        area = coords.shape[0]
        results.append({'id': label_id, 'z_mean': z_mean, 'z_min': z_min, 'z_max': z_max, 'area': area})

# Sort by Z-mean (Superior to Inferior)
# We suspect higher Z = Superior in this scan
results.sort(key=lambda x: x['z_mean'], reverse=True)

print("\nAnatomical Z-Profile (Sorted Superior to Inferior?):")
for r in results:
    print(f"Label {r['id']:3}: Z {r['z_min']:4}-{r['z_max']:4} (mean {r['z_mean']:6.1f}), Area: {r['area']:8}")

# Determine 'Superior' vs 'Inferior' by looking at known pairs if possible
# TotalSegmentator Right Femur is usually 45 or 76 depending on version.
# Tibia is usually 47 or 81.
# Usually femur is above tibia.
# In my previous run: 
# Label 76: Z=1338
# Label 81: Z=1754
# If label 81 is Hip and 76 is Femur, then Higher Z = Superior.
