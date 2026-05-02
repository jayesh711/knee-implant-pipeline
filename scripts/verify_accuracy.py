import trimesh
import numpy as np
from pathlib import Path
import argparse
import sys

def calculate_accuracy(generated_mesh_path, reference_mesh_path):
    if not generated_mesh_path.exists():
        return f"Error: Generated mesh {generated_mesh_path.name} not found."
    if not reference_mesh_path.exists():
        return f"Error: Reference mesh {reference_mesh_path.name} not found."

    print(f"Comparing {generated_mesh_path.name} vs {reference_mesh_path.name}...")
    
    # Load meshes
    gen = trimesh.load(str(generated_mesh_path))
    ref = trimesh.load(str(reference_mesh_path))

    # 1. Volume Comparison
    gen_vol = gen.volume / 1000.0 # to cc
    ref_vol = ref.volume / 1000.0 # to cc
    vol_diff = abs(gen_vol - ref_vol)
    vol_error_pct = (vol_diff / ref_vol * 100) if ref_vol > 0 else 0

    # 2. Surface Distance (Memory Efficient KDTree with ICP Alignment)
    from scipy.spatial import cKDTree
    
    # --- UPGRADE: ICP Alignment (V3) ---
    # To compare meshes from different coordinate systems (NIfTI vs DICOM Patient),
    # we use ICP to align the Gen mesh to the Ref mesh first.
    print( "  Aligning meshes (ICP)...")
    
    # ICP requires an initial guess if they are very far apart. 
    # Use centroid alignment as a starter.
    gen.vertices -= gen.centroid
    gen.vertices += ref.centroid
    
    matrix, transformed_gen_verts, cost = trimesh.registration.icp(gen.vertices, ref.vertices, threshold=1.0, max_iterations=50)
    
    # Re-sample from the transformed mesh
    samples, _ = trimesh.sample.sample_surface_even(trimesh.Trimesh(transformed_gen_verts, gen.faces), 5000)
    
    tree = cKDTree(ref.vertices)
    distances, _ = tree.query(samples, k=1)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    tree = cKDTree(ref.vertices)
    distances, _ = tree.query(samples, k=1)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    max_dist = np.max(distances)
    rmse = np.sqrt(np.mean(distances**2))

    return {
        "mean_dist_mm": mean_dist,
        "rmse_mm": rmse,
        "max_dist_mm": max_dist,
        "vol_gen_cc": gen_vol,
        "vol_ref_cc": ref_vol,
        "vol_error_pct": vol_error_pct
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Patient/Volume name")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory containing reference STLs")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    meshes_dir = Path("data/meshes")
    
    results = {}
    for bone in ["femur", "tibia"]:
        gen_path = meshes_dir / f"{args.name}_{bone}.stl"
        # Match Cuvis naming (Femur.stl / Tibia.stl)
        ref_path = ref_dir / f"{bone.capitalize()}.stl"
        
        acc = calculate_accuracy(gen_path, ref_path)
        if isinstance(acc, str):
            print(acc)
        else:
            results[bone] = acc
            print(f"\n--- {bone.upper()} ACCURACY REPORT ---")
            print(f"Mean Surface Distance: {acc['mean_dist_mm']:.3f} mm")
            print(f"RMSE:                   {acc['rmse_mm']:.3f} mm")
            print(f"Max Distance:           {acc['max_dist_mm']:.3f} mm")
            print(f"Volume Accuracy:        {100 - acc['vol_error_pct']:.1f}%")
            print("-" * 30)

if __name__ == "__main__":
    main()
