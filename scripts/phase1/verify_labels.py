import nibabel as nib
import numpy as np
from pathlib import Path
from config import DATA

def verify_labels(volume_name="S0001"):
    # Try multiple common output paths
    potential_paths = [
        DATA / "segmentations" / "phase1" / volume_name / "multilabel.nii.gz",
        DATA / "segmentations" / "phase1" / f"{volume_name}.nii",
        DATA / "segmentations" / "phase1" / f"{volume_name}.nii.gz"
    ]
    
    seg_path = None
    for p in potential_paths:
        if p.exists():
            seg_path = p
            break
            
    if not seg_path:
        print(f"Error: No segmentation file found for {volume_name} in {DATA / 'segmentations' / 'phase1'}")
        return
        
    print(f"--- Verifying labels for {volume_name} in {seg_path.name} ---")
    seg = nib.load(str(seg_path))
    data = seg.get_fdata()
    
    unique_labels = np.unique(data).astype(int)
    print(f"Unique Label IDs found: {unique_labels}")
    print(f"Total non-zero voxels: {int((data > 0).sum())}")
    
    # Common bone labels in TotalSegmentator:
    # 24: Femur left, 25: Femur right, 26: Tibia left, 27: Tibia right
    bone_labels = {24: "Femur Left", 25: "Femur Right", 26: "Tibia Left", 27: "Tibia Right"}
    
    print("\nLabel Statistics:")
    for label_id in unique_labels:
        count = int((data == label_id).sum())
        name = bone_labels.get(label_id, "Unknown")
        print(f"  ID {label_id:2}: {count:10,} voxels ({name})")
        if label_id in bone_labels:
            found_bones = True
            
    if not found_bones:
        print("  Warning: No common knee bone labels (24-27) found in the scan.")
    
    print("-" * 30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify labels in NIfTI segmentation")
    parser.add_argument("--name", type=str, default="S0001", help="Volume name to verify")
    
    args = parser.parse_args()
    verify_labels(args.name)
