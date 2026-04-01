import trimesh
import argparse
from pathlib import Path
import os

def check_mesh_quality(mesh_path):
    print(f"--- Analyzing Mesh Quality: {mesh_path.name} ---")
    if not mesh_path.exists():
        print(f"Error: {mesh_path} not found.")
        return None
        
    mesh = trimesh.load(str(mesh_path))
    
    # 1. Watertight check (Critical for haptic boundaries)
    is_watertight = mesh.is_watertight
    
    # 2. Manifold check
    is_manifold = mesh.is_edge_manifold
    
    # 3. Volume calculation (mm^3)
    volume = mesh.volume
    
    # 4. Surface area (mm^2)
    surface_area = mesh.area
    
    # 5. Euler characteristic (should be 2 for a closed sphere-like surface)
    euler = mesh.euler_number
    
    # 6. Triangle count
    tri_count = len(mesh.faces)
    
    # 7. Non-manifold edges
    non_manifold_edges = mesh.edges_unique[mesh.edges_unique_inverse][mesh.is_edge_manifold]
    
    results = {
        "name": mesh_path.name,
        "is_watertight": is_watertight,
        "is_manifold": is_manifold,
        "volume": volume,
        "surface_area": surface_area,
        "euler": euler,
        "triangles": tri_count,
        "status": "PASS" if is_watertight and tri_count < 200000 else "FAIL"
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to mesh (.stl/.obj)")
    args = parser.parse_args()
    
    res = check_mesh_quality(Path(args.mesh_path))
    
    if res:
        print("\n" + "="*30)
        print(" MESH QUALITY REPORT")
        print("="*30)
        print(f"Name:           {res['name']}")
        print(f"Watertight:     {res['is_watertight']}")
        print(f"Manifold:       {res['is_manifold']}")
        print(f"Euler Number:   {res['euler']} (Target: 2)")
        print(f"Volume:         {res['volume']:.2f} mm^3")
        print(f"Surface Area:   {res['surface_area']:.2f} mm^2")
        print(f"Triangles:      {res['triangles']}")
        print(f"FINAL STATUS:   {res['status']}")
        print("="*30)
