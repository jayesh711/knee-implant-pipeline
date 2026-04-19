import trimesh
import numpy as np
from pathlib import Path

def analyze_mesh_character(path_ours, path_ref):
    mesh_ours = trimesh.load(path_ours)
    mesh_ref = trimesh.load(path_ref)
    
    print(f"Analysis for: {Path(path_ours).name} vs {Path(path_ref).name}")
    
    # 1. Complexity
    print(f"OURS:  Vertices={len(mesh_ours.vertices)}, Faces={len(mesh_ours.faces)}")
    print(f"CUVIS: Vertices={len(mesh_ref.vertices)}, Faces={len(mesh_ref.faces)}")
    
    # 2. Curvature / Smoothness (Standard deviation of vertex normals or face adjacencies)
    # A rougher mesh has higher variation in face normals locally
    def get_roughness(mesh):
        # Sample average angle between adjacent face normals
        try:
            adj = mesh.face_adjacency
            # trimesh already provides face_adjacency_angles
            angles = mesh.face_adjacency_angles
            return np.mean(angles), np.std(angles)
        except:
            return 0, 0

    rough_mu_o, rough_std_o = get_roughness(mesh_ours)
    rough_mu_r, rough_std_r = get_roughness(mesh_ref)
    
    print(f"Roughness (Avg Adj Angle): OURS={np.degrees(rough_mu_o):.2f} | CUVIS={np.degrees(rough_mu_r):.2f}")
    print(f"Roughness (Std Dev):       OURS={np.degrees(rough_std_o):.2f} | CUVIS={np.degrees(rough_std_r):.2f}")

if __name__ == "__main__":
    analyze_mesh_character(
        r"d:\Github\knee-implant-pipeline\data\meshes\AB_72Y_Male_Left_femur.stl",
        r"E:\Cuvis_Software\OneDrive_1_4-19-2026\Femur.stl"
    )
    analyze_mesh_character(
        r"d:\Github\knee-implant-pipeline\data\meshes\AB_72Y_Male_Left_tibia.stl",
        r"E:\Cuvis_Software\OneDrive_1_4-19-2026\Tibia.stl"
    )
