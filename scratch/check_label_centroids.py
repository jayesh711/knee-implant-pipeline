import nibabel as nib
import numpy as np
from skimage import measure

patient = "AB_72_Y_Male-Right_Knee"
seg_path = f"data/segmentations/phase1/{patient}.nii"

print(f"Loading {seg_path}...")
img = nib.load(seg_path)
data = img.get_fdata()

# Identify all major components in the segmentation
# TotalSegmentator labels are often 1-117
labels = np.unique(data).astype(int)
print(f"Unique labels: {labels}")

def get_major_components(label_id):
    mask = (data == label_id)
    lbl, num = measure.label(mask, return_num=True)
    if num == 0: return []
    props = measure.regionprops(lbl)
    # Filter for components > 5000 voxels
    return [p for p in props if p.area > 5000]

# Check label 76 (Assumed Femur)
f_comps = get_major_components(76)
for i, p in enumerate(f_comps):
    print(f"Femur (76) Component {i}: Centroid {p.centroid}, Area {p.area}")

# Check label 81 (Assumed Tibia)
t_comps = get_major_components(81)
for i, p in enumerate(t_comps):
    print(f"Tibia (81) Component {i}: Centroid {p.centroid}, Area {p.area}")

# Check other labels present
for l in [18, 20, 21, 22, 25, 26]:
    if l in labels:
        comps = get_major_components(l)
        for i, p in enumerate(comps):
            print(f"Label {l} Component {i}: Centroid {p.centroid}, Area {p.area}")
