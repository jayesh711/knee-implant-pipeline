import nibabel as nib
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import binary_dilation
from pathlib import Path
import argparse
from config import DATA

def ground_truth_reconstruction(name="S0001", threshold=200):
    """
    Generate a 'Pure Signal' baseline reconstruction from raw HU data.
    Now refined to mask out non-bone noise (scanner bed, etc.) using AI segments.
    """
    ct_path = DATA / "NIfTI" / f"{name}_raw.nii.gz"
    if not ct_path.exists():
        ct_path = DATA / "NIfTI" / f"{name}.nii.gz"
        if not ct_path.exists():
            print(f"Error: CT Scan {name} not found in {DATA / 'NIfTI'}")
            return

    print(f"--- Generating REFINED GROUND TRUTH for {name} ---")
    print(f"Using Threshold: {threshold} HU (Solid Bone Level)")
    
    img = nib.load(str(ct_path))
    data = img.get_fdata()
    affine = img.affine
    
    # NEW: Bone-Proximal Filter (Masking noise)
    # Search for the segmentation file to guide the noise removal
    seg_potential = [
        DATA / "segmentations" / "phase1" / f"{name}.nii",
        DATA / "segmentations" / "phase1" / f"{name}.nii.gz",
        DATA / "segmentations" / "phase1" / name / "multilabel.nii.gz"
    ]
    
    clean_mask = None
    for p in seg_potential:
        if p.exists():
            print(f"  Found segmentation for noise suppression: {p.name}")
            seg_img = nib.load(str(p))
            seg_data = seg_img.get_fdata()
            # Create a broad bone mask
            # Dilation increased to 100 (~50mm) to ensure missing bone signal isn't cut off
            clean_mask = (seg_data > 0).astype(np.uint8)
            clean_mask = binary_dilation(clean_mask, iterations=100) 
            break

    if clean_mask is not None:
        print("  Applying generous bone-proximal masking (Full-Leg Restore)...")
        data[clean_mask == 0] = -1000 # Mask scanner bed but keep bone surroundings
    
    # 1. Simple Thresholding
    mask = (data > threshold).astype(np.uint8)
    
    # 2. Connected Component Analysis (Keep largest structures)
    print("  Isolating main bone structures...")
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    
    final_mask = np.zeros_like(mask)
    for r in regions[:15]: # Keep up to 15 largest components
        final_mask[labels == r.label] = 1
    
    if np.sum(final_mask) == 0:
        print("Warning: No bone detected at this threshold!")
        return
        
    # 3. Marching Cubes (Voxel Space)
    verts, faces, _, _ = measure.marching_cubes(final_mask, level=0.5)
    
    # 4. Affine Transform (Voxel to World)
    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
    
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
    
    output_dir = DATA / "meshes"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_ground_truth.stl"
    
    mesh.export(str(out_path))
    print(f"SUCCESS: Refined Ground Truth output to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001")
    parser.add_argument("--threshold", type=int, default=200, help="HU Threshold for bone")
    args = parser.parse_args()
    ground_truth_reconstruction(args.name, args.threshold)
