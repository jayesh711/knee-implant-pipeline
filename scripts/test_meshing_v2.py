import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_fill_holes
from skimage import measure
from pathlib import Path
import os

def extract_mesh_v2(mask, affine, sigma=0.8, target_faces=150000):
    """
    High-quality mesh extraction using Anti-Aliased Marching Cubes 
    via Distance Transform + Gaussian Blur.
    """
    print(f"    Step 1: Filling holes and preparing mask...")
    mask = binary_fill_holes(mask).astype(np.uint8)
    
    print(f"    Step 2: Computing Distance Transform (EDT)...")
    # EDT: Positive inside, Negative outside (or vice versa)
    # distance_transform_edt computes distance to nearest zero.
    # So we do it for mask and ~mask.
    dt_pos = distance_transform_edt(mask)
    dt_neg = distance_transform_edt(1 - mask)
    dt = dt_pos - dt_neg
    
    if sigma > 0:
        print(f"    Step 3: Applying Gaussian Blur (sigma={sigma})...")
        dt = gaussian_filter(dt, sigma=sigma)
    
    print(f"    Step 4: Marching Cubes at Level 0...")
    verts, faces, normals, values = measure.marching_cubes(dt, level=0)
    
    # Voxel -> World
    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
    
    print(f"    Step 5: PyMeshLab Post-Processing...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    
    # 5a. Remove artifacts
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    # ms.apply_filter("meshing_remove_isolated_pieces_wrt_diameter", mincomponentdiag=10.0) # percentage as float
    
    # 5b. Isotropic Remeshing (Crucial for quality)
    print(f"    Step 5b: Isotropic Explicit Remeshing...")
    # Target length approx 0.8mm for 120k-150k faces on a femur
    ms.apply_filter("meshing_isotropic_explicit_remeshing", targetlen=pymeshlab.PureValue(0.8), iterations=3)
    
    # 5c. Taubin Smoothing (Very gentle after EDT+Blur)
    print(f"    Step 5c: Taubin Smoothing...")
    ms.apply_coord_taubin_smoothing(stepsmoothnum=20, lambda_=0.5, mu=-0.53)
    
    # 5d. Final Decimation if needed
    current_faces = ms.current_mesh().face_number()
    if current_faces > target_faces:
        print(f"    Step 5d: Decimating {current_faces} -> {target_faces}...")
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=target_faces)
    
    final_m = ms.current_mesh()
    return trimesh.Trimesh(vertices=final_m.vertex_matrix(), faces=final_m.face_matrix())

# Test on AB_72Y_Male_Left
DATA_DIR = Path("d:/Github/knee-implant-pipeline/data")
seg_path = DATA_DIR / "segmentations/phase1/AB_72Y_Male_Left_clinical.nii.gz"
out_dir = DATA_DIR / "meshes/tests"
os.makedirs(out_dir, exist_ok=True)

if seg_path.exists():
    print(f"Loading {seg_path.name}...")
    img = nib.load(str(seg_path))
    data = img.get_fdata().astype(np.uint8)
    affine = img.affine
    
    # Femur is label 1 in Precision
    femur_mask = (data == 1).astype(np.uint8)
    
    if np.any(femur_mask):
        print("Extracting V2 High Quality Mesh...")
        mesh_v2 = extract_mesh_v2(femur_mask, affine)
        out_path = out_dir / "AB_72Y_Male_Left_femur_V2.stl"
        mesh_v2.export(str(out_path))
        print(f"Saved to {out_path}")
    else:
        print("Femur mask empty!")
else:
    print(f"File not found: {seg_path}")
