import os
import trimesh
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.spatial import KDTree
import argparse

def calculate_surface_agreement(mesh, volume_data, affine):
    """
    Sample HU values from the volume at each mesh vertex.
    Returns the mean HU and the % of vertices in the 'bone-edge' range.
    """
    # Convert mesh vertices from world coordinates to voxel coordinates
    inv_affine = np.linalg.inv(affine)
    homog_verts = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
    voxel_coords = (inv_affine @ homog_verts.T).T[:, :3]
    
    # Clip to volume dimensions
    dims = volume_data.shape
    v_coords = np.clip(voxel_coords, 0, np.array(dims)-1).astype(int)
    
    # Sample HU values
    hu_values = volume_data[v_coords[:, 0], v_coords[:, 1], v_coords[:, 2]]
    
    # Anatomical Agreement Metric:
    # High quality bone boundary is typically between 300 and 1300 HU.
    bone_edge_mask = (hu_values >= 300) & (hu_values <= 1300)
    agreement_pct = (np.sum(bone_edge_mask) / len(hu_values)) * 100
    
    return np.mean(hu_values), agreement_pct

def compare_two_meshes(mesh_pipe, mesh_ref):
    """
    Calculate geometric distance between two meshes.
    Performs ICP registration to handle coordinate system shifts.
    """
    # Try registration (ICP) - find transform from ref to pipe
    print("  Aligning meshes (ICP Registration)...")
    transform, cost = trimesh.registration.mesh_other(mesh_ref, mesh_pipe, samples=1000)
    
    # Apply transform to ref
    mesh_ref_aligned = mesh_ref.copy()
    mesh_ref_aligned.apply_transform(transform)
    
    # Points to sample
    points_pipe = mesh_pipe.vertices
    points_ref = mesh_ref_aligned.vertices
    
    # Distance from pipe to ref
    tree = KDTree(points_ref)
    dist_p2r, _ = tree.query(points_pipe)
    
    # Metrics
    rms = np.sqrt(np.mean(dist_p2r**2))
    mean_dist = np.mean(dist_p2r)
    hausdorff = np.max(dist_p2r)
    
    return {
        "RMS": rms,
        "MeanDist": mean_dist,
        "Hausdorff": hausdorff
    }

def main():
    parser = argparse.ArgumentParser(description="Clinical Mesh Comparison and Validation")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory containing Cuvis STLs")
    parser.add_argument("--name", type=str, required=True, help="Patient name (e.g. AB_72Y_Male_Left)")
    args = parser.parse_args()
    
    base_path = Path(r"d:\Github\knee-implant-pipeline")
    ref_dir = Path(args.ref_dir)
    meshes_out = base_path / "data" / "meshes"
    raw_vol_path = base_path / "data" / "NIfTI" / f"{args.name}_raw.nii.gz"
    
    print(f"\nComparing reconstruction quality for: {args.name}\n")
    
    # Load raw CT volume for truth verification
    print(f"Loading raw CT: {raw_vol_path.name}")
    img_raw = nib.load(str(raw_vol_path))
    data_raw = img_raw.get_fdata()
    affine = img_raw.affine
    
    results = {}
    
    for bone in ["femur", "tibia"]:
        print(f"\n--- {bone.upper()} ANALYSIS ---")
        
        # Paths
        pipe_path = meshes_out / f"{args.name}_{bone}.stl"
        ref_path = ref_dir / f"{bone.capitalize()}.stl"
        
        if not pipe_path.exists() or not ref_path.exists():
            print(f"  Warning: Missing mesh for {bone}. Skipping...")
            continue
            
        # Load Meshes
        mesh_pipe = trimesh.load(str(pipe_path))
        mesh_ref = trimesh.load(str(ref_path))
        
        # 1. Geometric Comparison (Pipe vs Ref)
        geom = compare_two_meshes(mesh_pipe, mesh_ref)
        print(f"  Geometric Offset (Ours vs Cuvis):")
        print(f"    - Avg Distance (RMS): {geom['RMS']:.3f} mm")
        print(f"    - Max Deviation:      {geom['Hausdorff']:.3f} mm")
        
        # 2. Anatomical Ground-Truth Agreement (Mesh vs DICOM)
        print("  Sampling Anatomical Accuracy (DICOM HU Agreement)...")
        pipe_hu, pipe_agree = calculate_surface_agreement(mesh_pipe, data_raw, affine)
        ref_hu, ref_agree = calculate_surface_agreement(mesh_ref, data_raw, affine)
        
        print(f"  Anatomical Score (How well mesh follows bone edge):")
        print(f"    - PIPELINE:  {pipe_agree:.1f}% Agreement (Avg HU: {pipe_hu:.1f})")
        print(f"    - CUVIS:     {ref_agree:.1f}% Agreement (Avg HU: {ref_hu:.1f})")
        
        results[bone] = {
            "geom": geom,
            "pipe_agree": pipe_agree,
            "ref_agree": ref_agree
        }
        
    # VERDICT
    print("\n" + "="*50)
    print(" FINAL VERDICT")
    print("="*50)
    for bone, res in results.items():
        diff = res['pipe_agree'] - res['ref_agree']
        winner = "PIPELINE (Ours)" if diff > 0 else "CUVIS Software"
        abs_diff = abs(diff)
        print(f"{bone.upper()}: {winner} is better by {abs_diff:.1f}% accuracy score.")
    print("="*50)

if __name__ == "__main__":
    main()
