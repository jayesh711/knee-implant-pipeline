import nibabel as nib
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data")

def check_centroids(volume_name):
    seg_file = DATA_DIR / "segmentations" / "phase1" / f"{volume_name}.nii"
    if not seg_file.exists():
        seg_file = DATA_DIR / "segmentations" / "phase1" / f"{volume_name}.nii.gz"
    
    if not seg_file.exists():
        print(f"File not found: {seg_file}")
        return

    print(f"Analyzing Centroids: {seg_file}")
    img = nib.load(str(seg_file))
    data = img.get_fdata()
    affine = img.affine
    
    unique_labels = np.unique(data).astype(int)
    
    print(f"{'LabelID':<10} | {'Centroid (Voxel)':<20} | {'Centroid (World)':<20} | {'Volume (cc)':<10}")
    print("-" * 75)
    
    zooms = img.header.get_zooms()
    voxel_vol = np.prod(zooms) / 1000.0
    
    for label in unique_labels:
        if label == 0: continue
        mask = (data == label)
        count = np.sum(mask)
        if count == 0: continue
        
        indices = np.argwhere(mask)
        centroid_voxel = indices.mean(axis=0)
        
        # Convert to world coordinates
        centroid_world = (affine[:3, :3] @ centroid_voxel) + affine[:3, 3]
        
        vol = count * voxel_vol
        print(f"{label:<10} | {str(np.round(centroid_voxel, 1)):<20} | {str(np.round(centroid_world, 1)):<20} | {vol:<10.2f}")

if __name__ == "__main__":
    check_centroids("AB_72_Y_Male-Right_Knee")
