import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

patient = "AB_72_Y_Male-Right_Knee"
seg_path = f"data/segmentations/phase1/{patient}.nii"

print(f"Loading {seg_path}...")
img = nib.load(seg_path)
data = img.get_fdata()

# Tibia is label 81
tibia_mask = (data == 81)
z_counts = np.sum(tibia_mask, axis=(0, 1))

# Femur is label 76
femur_mask = (data == 76)
fz_counts = np.sum(femur_mask, axis=(0, 1))

# Find ranges
t_z_indices = np.where(z_counts > 0)[0]
f_z_indices = np.where(fz_counts > 0)[0]

print(f"Femur Z-range: {f_z_indices.min()} to {f_z_indices.max()}")
print(f"Tibia Z-range: {t_z_indices.min()} to {t_z_indices.max()}")

# Look at Tibia profile to find ankle
# Foot is likely at the high Z end (since Tibia mean is 1754 and Femur is 1338)
for i in range(len(t_z_indices)):
    z = t_z_indices[i]
    count = z_counts[z]
    if i % 20 == 0 or i == len(t_z_indices) - 1:
        print(f"Z: {z}, area: {count} voxels")

# Find 'Ankle' - a narrowing in the lower half
lower_half_start = int(t_z_indices.min() + (t_z_indices.max() - t_z_indices.min()) * 0.5)
narrowest_z = lower_half_start + np.argmin(z_counts[lower_half_start:])
print(f"Narrowest point in lower half: Z={narrowest_z}, area={z_counts[narrowest_z]}")
