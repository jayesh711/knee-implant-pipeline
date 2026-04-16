import nibabel as nib
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data")

def check_z_ranges(volume_name):
    seg_file = DATA_DIR / "segmentations" / "phase1" / f"{volume_name}.nii"
    if not seg_file.exists():
        seg_file = DATA_DIR / "segmentations" / "phase1" / f"{volume_name}.nii.gz"
    
    if not seg_file.exists():
        print(f"File not found: {seg_file}")
        return

    img = nib.load(str(seg_file))
    data = img.get_fdata()
    affine = img.affine
    
    unique_labels = np.unique(data).astype(int)
    
    print(f"{'Label':<10} | {'Z-Min (World)':<15} | {'Z-Max (World)':<15} | {'Z-Center':<15}")
    print("-" * 60)
    
    for label in unique_labels:
        if label == 0: continue
        indices = np.argwhere(data == label)
        if len(indices) == 0: continue
        
        z_voxels = indices[:, 2]
        z_min_vox = np.min(z_voxels)
        z_max_vox = np.max(z_voxels)
        
        # Approximate world Z (assuming no large rotation for simplicity of diagnosis)
        z_min_world = (affine[2, 2] * z_min_vox) + affine[2, 3]
        z_max_world = (affine[2, 2] * z_max_vox) + affine[2, 3]
        
        # Handle negative affine[2,2]
        if z_min_world > z_max_world:
            z_min_world, z_max_world = z_max_world, z_min_world
            
        print(f"{label:<10} | {z_min_world:<15.1f} | {z_max_world:<15.1f} | {(z_min_world+z_max_world)/2:<15.1f}")

if __name__ == "__main__":
    check_z_ranges("AB_72_Y_Male-Right_Knee")
